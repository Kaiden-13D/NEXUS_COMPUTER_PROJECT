import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from matplotlib import pyplot as plt
import copy
import random

# --- 1. 하이퍼파라미터 및 설정 ---
class Args:
    def __init__(self):
        self.epochs = 30  # 전체 통신 라운드
        self.local_epochs = 2 # 각 클라이언트가 로컬에서 학습할 에폭 수
        self.num_clients = 100 # <--- 변경: 100명의 클라이언트
        self.num_zones = 6   # <--- 추가: 6개의 존
        self.batch_size = 64
        self.lr = 0.01
        self.momentum = 0.5
        self.seed = 42

args = Args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# --- 2. 모델 정의 ---
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# --- 3. 데이터 준비 및 Non-IID 분배 ---
def get_data_loaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # --- 추가: 100명의 클라이언트를 6개의 Zone에 할당 ---
    client_zone_mapping = {i: random.randint(0, args.num_zones - 1) for i in range(args.num_clients)}

    # Non-IID 데이터 분배: 각 클라이언트가 2개의 숫자 라벨만 갖도록 설정
    num_labels_per_client = 2
    labels = train_dataset.targets
    client_data_indices = {i: np.empty(0, dtype=np.int64) for i in range(args.num_clients)}
    
    # 각 라벨별 데이터 인덱스
    label_indices = [np.where(labels == i)[0] for i in range(10)]

    # 각 클라이언트에게 라벨 할당
    client_labels = {}
    for client_id in range(args.num_clients):
        client_labels[client_id] = np.random.choice(10, num_labels_per_client, replace=False)

    # 라벨에 따라 데이터 분배
    for client_id, labels_for_client in client_labels.items():
        for label in labels_for_client:
            # 해당 라벨을 가진 클라이언트 수를 기반으로 데이터 분할
            num_clients_with_label = sum(1 for cid in range(args.num_clients) if label in client_labels[cid])
            num_samples = len(label_indices[label]) // num_clients_with_label
            
            # 클라이언트에게 샘플 할당 (중복 방지를 위해 간단한 방식으로 처리)
            # 이 방식은 완벽히 균등하진 않지만 Non-IID를 효과적으로 시뮬레이션
            rand_idx = np.random.choice(len(label_indices[label]), num_samples, replace=False)
            client_data_indices[client_id] = np.concatenate((client_data_indices[client_id], label_indices[label][rand_idx]))
    
    client_loaders = []
    for i in range(args.num_clients):
        subset = Subset(train_dataset, client_data_indices[i])
        loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True)
        client_loaders.append(loader)

    print("--- Client & Zone & Data Distribution (Sample) ---")
    zone_counts = {z:0 for z in range(args.num_zones)}
    for cid, zid in client_zone_mapping.items():
        zone_counts[zid] += 1

    print(f"Zone별 클라이언트 수: {zone_counts}")

    for i in range(3):
        labels_in_loader = set()
        if len(client_loaders[i].dataset) > 0:
            for _, targets in client_loaders[i]:
                labels_in_loader.update(targets.numpy())
        print(f"Client {i} (Zone {client_zone_mapping[i]}) has labels: {sorted(list(labels_in_loader))}, data size: {len(client_loaders[i].dataset)}")

    return client_loaders, test_loader, client_zone_mapping

# --- 4. 클라이언트 및 서버 로직 ---
def client_update(client_model, optimizer, train_loader, local_epochs):
    """클라이언트 측 학습 로직"""
    client_model.train()
    for epoch in range(local_epochs):
        for data, target in train_loader:
            if len(data) == 0: continue
            optimizer.zero_grad()
            output = client_model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
    return client_model.state_dict()

def server_aggregate(global_model, client_models):
    """서버 측 모델 병합 로직 (FedAvg)"""
    if not client_models: # 모델이 없는 경우
        return global_model # 기존 모델 그대로 반환
    
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[i][k].float() for i in range(len(client_models))], 0).mean(0)
    global_model.load_state_dict(global_dict)
    return global_model

def test(model, test_loader):
    """글로벌 모델 성능 평가"""
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy

# --- 5. 메인 실행 루프 ---
if __name__ == '__main__':
    print("Initializing Hierarchical Federated Learning Simulation...")
    
    # 데이터 로더 및 존 매핑 정보 준비
    client_loaders, test_loader, client_zone_mapping = get_data_loaders()

    # 글로벌 모델 초기화
    global_model = Net()
    
    test_losses = []
    accuracies = []

    print("\nStarting Training...\n")
    # 연합 학습 라운드 시작
    for epoch in range(args.epochs):
        
        # --- 1. 1차 집계: Zone별로 모델 학습 및 병합 ---
        zone_models = []
        for z_id in range(args.num_zones):
            # 현재 존에 속한 클라이언트들 찾기
            clients_in_zone = [c_id for c_id, z in client_zone_mapping.items() if z == z_id]
            
            if not clients_in_zone: # 존에 클라이언트가 없으면 건너뛰기
                continue

            local_client_models = []
            for c_id in clients_in_zone:
                # 중요: 매번 글로벌 모델을 복사해서 클라이언트에 전달
                local_model = copy.deepcopy(global_model)
                optimizer = optim.SGD(local_model.parameters(), lr=args.lr, momentum=args.momentum)
                
                # 클라이언트가 로컬 데이터로 모델 학습
                updated_weights = client_update(local_model, optimizer, client_loaders[c_id], args.local_epochs)
                local_client_models.append(updated_weights)
            
            # Zone 대표 모델 생성 (1차 병합)
            zone_model_agg = copy.deepcopy(global_model) # 병합을 위한 틀
            zone_model_agg = server_aggregate(zone_model_agg, local_client_models)
            zone_models.append(zone_model_agg.state_dict())

        # --- 2. 2차 집계: Zone 모델들을 병합하여 글로벌 모델 업데이트 ---
        global_model = server_aggregate(global_model, zone_models)

        # --- 3. 글로벌 모델 성능 평가 ---
        test_loss, accuracy = test(global_model, test_loader)
        test_losses.append(test_loss)
        accuracies.append(accuracy)
        
        print(f"Round {epoch+1}/{args.epochs} -> Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%")

    # --- 6. 결과 시각화 ---
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(args.epochs), test_losses, marker='o')
    plt.title("Test Loss vs. Communication Rounds")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Test Loss")
    
    plt.subplot(1, 2, 2)
    plt.plot(range(args.epochs), accuracies, marker='o', color='r')
    plt.title("Accuracy vs. Communication Rounds")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Accuracy (%)")
    
    plt.tight_layout()
    plt.show()

    print("\nHierarchical Federated Learning Simulation Finished.")
    # 최종 모델 저장
    torch.save(global_model.state_dict(), "hierarchical_federated_model.pt")
    print("Final global model saved as 'hierarchical_federated_model.pt'")

