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

# --- 1. Hyperparameters and Settings ---
class Args:
    # Args 클래스는 시뮬레이션에 사용될 모든 하이퍼파라미터와 설정을 포함합니다.
    def __init__(self):
        # __init__ 메소드는 Args 클래스의 인스턴스가 생성될 때 하이퍼파라미터를 초기화합니다.
        self.epochs = 30 # 전체 통신 라운드(에포크)의 수입니다.
        self.local_epochs = 2 # 각 클라이언트가 로컬에서 모델을 훈련하는 에포크 수입니다.
        self.num_clients = 100 # 시뮬레이션에 참여하는 총 클라이언트(UE)의 수입니다.
        self.num_zones = 6 # UAV가 관리하는 총 존(zone)의 수입니다.
        self.num_global_clusters = 3 # 전체 시스템에 걸쳐 생성할 글로벌 클러스터(K)의 수입니다.
        self.batch_size = 64 # 훈련 및 테스트 시 사용되는 데이터의 배치 크기입니다.
        self.lr = 0.01 # 학습률(learning rate)입니다.
        self.momentum = 0.5 # SGD 옵티마이저에 사용될 모멘텀 값입니다.
        self.seed = 42 # 재현성을 위해 사용되는 랜덤 시드입니다.
        self.log_interval = 10 # 로그를 출력하는 간격입니다.

args = Args() # Args 클래스의 인스턴스를 생성하여 하이퍼파라미터를 설정합니다.
torch.manual_seed(args.seed) # PyTorch의 랜덤 시드를 고정하여 결과를 재현 가능하게 합니다.
np.random.seed(args.seed) # NumPy의 랜덤 시드를 고정합니다.
random.seed(args.seed) # Python의 내장 random 모듈의 시드를 고정합니다.

# --- 2. Model Definition ---
class Net(nn.Module):
    # Net 클래스는 MNIST 데이터셋을 위한 간단한 컨볼루션 신경망(CNN)을 정의합니다.
    def __init__(self):
        # __init__ 메소드는 신경망의 각 계층을 초기화합니다.
        super(Net, self).__init__() # 부모 클래스인 nn.Module의 __init__을 호출합니다.
        self.conv1 = nn.Conv2d(1, 20, 5, 1) # 첫 번째 2D 컨볼루션 계층: 1개 입력 채널, 20개 출력 채널, 5x5 커널, 스트라이드 1
        self.conv2 = nn.Conv2d(20, 50, 5, 1) # 두 번째 2D 컨볼루션 계층: 20개 입력 채널, 50개 출력 채널, 5x5 커널, 스트라이드 1
        self.fc1 = nn.Linear(4*4*50, 500) # 첫 번째 완전 연결 계층(fully connected layer): 4*4*50 입력, 500 출력
        self.fc2 = nn.Linear(500, 10) # 두 번째 완전 연결 계층: 500 입력, 10 출력 (10개 클래스)

    def forward(self, x):
        # forward 메소드는 입력 데이터 x가 신경망을 통과하는 과정을 정의합니다.
        x = F.relu(self.conv1(x)) # 첫 번째 컨볼루션 계층을 통과하고 ReLU 활성화 함수를 적용합니다.
        x = F.max_pool2d(x, 2, 2) # 2x2 맥스 풀링을 적용하여 특성 맵의 크기를 줄입니다.
        x = F.relu(self.conv2(x)) # 두 번째 컨볼루션 계층을 통과하고 ReLU 활성화 함수를 적용합니다.
        x = F.max_pool2d(x, 2, 2) # 다시 2x2 맥스 풀링을 적용합니다.
        x = x.view(-1, 4*4*50) # 텐서를 1차원으로 펼쳐서 완전 연결 계층에 입력할 수 있도록 준비합니다.
        x = F.relu(self.fc1(x)) # 첫 번째 완전 연결 계층을 통과하고 ReLU 활성화 함수를 적용합니다.
        x = self.fc2(x) # 최종 완전 연결 계층을 통과하여 클래스별 점수를 얻습니다.
        return F.log_softmax(x, dim=1) # log_softmax를 적용하여 확률 분포 형태의 출력을 반환합니다.

# --- 3. Entity Classes (UE, UAV, Satellite) ---

class UE:
    # UE 클래스는 계층 구조의 가장 낮은 레벨에 있는 사용자 장비(클라이언트)를 나타냅니다.
    # 자신의 개인 데이터로 모델을 훈련하는 역할을 합니다.
    def __init__(self, id, data_loader):
        # __init__ 메소드는 UE 객체를 초기화합니다.
        self.id = id # 클라이언트의 고유 ID입니다.
        self.data_loader = data_loader # 클라이언트의 로컬 데이터셋을 위한 DataLoader입니다.
        self.model = Net() # 클라이언트가 사용할 로컬 모델입니다.

    def train(self, cluster_model_state, local_epochs, lr, momentum):
        # train 메소드는 클라이언트의 모델을 로컬에서 몇 에포크 동안 훈련합니다.
        # 할당된 클러스터 모델의 상태에서 훈련을 시작합니다.
        self.model.load_state_dict(copy.deepcopy(cluster_model_state)) # 클러스터 모델의 가중치를 복사하여 로컬 모델을 초기화합니다.
        self.model.train() # 모델을 훈련 모드로 설정합니다.
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum) # SGD 옵티마이저를 설정합니다.
        
        for epoch in range(local_epochs):
            # 지정된 로컬 에포크 수만큼 훈련을 반복합니다.
            for data, target in self.data_loader:
                # 데이터 로더에서 배치 단위로 데이터를 가져옵니다.
                if len(data) == 0: continue # 데이터가 비어있으면 건너뜁니다.
                optimizer.zero_grad() # 옵티마이저의 그래디언트를 초기화합니다.
                output = self.model(data) # 모델에 데이터를 입력하여 예측을 수행합니다.
                loss = F.nll_loss(output, target) # Negative Log Likelihood Loss를 계산합니다.
                loss.backward() # 역전파를 통해 그래디언트를 계산합니다.
                optimizer.step() # 옵티마이저를 사용하여 모델의 가중치를 업데이트합니다.
        
        return self.model.state_dict() # 훈련이 완료된 로컬 모델의 가중치를 반환합니다.

    def evaluate_model_loss(self, model_state):
        # evaluate_model_loss 메소드는 주어진 모델이 클라이언트의 로컬 데이터에서 얼마나 손실을 내는지 계산합니다.
        # 이는 동적 클러스터링에서 클라이언트가 가장 적합한 클러스터를 찾는 '인터뷰' 과정의 핵심입니다.
        eval_model = Net() # 평가를 위한 새로운 모델 객체를 생성합니다.
        eval_model.load_state_dict(copy.deepcopy(model_state)) # 평가할 모델의 가중치를 로드합니다.
        eval_model.eval() # 모델을 평가 모드로 설정합니다.
        total_loss = 0 # 총 손실을 저장할 변수입니다.
        total_samples = 0 # 총 샘플 수를 저장할 변수입니다.
        with torch.no_grad():
            # 그래디언트 계산을 비활성화하여 평가 속도를 높입니다.
            for data, target in self.data_loader:
                # 데이터 로더에서 배치 단위로 데이터를 가져옵니다.
                if len(data) == 0: continue # 데이터가 비어있으면 건너뜁니다.
                output = eval_model(data) # 모델 예측을 수행합니다.
                loss = F.nll_loss(output, target, reduction='sum').item() # 배치 손실을 합산하여 계산합니다.
                total_loss += loss # 총 손실에 더합니다.
                total_samples += len(data) # 총 샘플 수에 더합니다.
        
        if total_samples == 0:
            # 샘플이 없는 경우 무한대 손실을 반환합니다.
            return float('inf')
        return total_loss / total_samples # 평균 손실을 계산하여 반환합니다.

class UAV:
    # UAV 클래스는 이제 위성과 클라이언트 간의 단순 중계기 역할을 합니다.
    # 클러스터링이나 모델 집계는 더 이상 수행하지 않습니다.
    def __init__(self, id):
        # __init__ 메소드는 UAV 객체를 초기화합니다.
        self.id = id # UAV의 고유 ID입니다.
        self.clients = [] # 이 UAV(존)에 속한 클라이언트 목록입니다.

    def add_client(self, client):
        # add_client 메소드는 UAV에 클라이언트를 추가합니다.
        self.clients.append(client)

class Satellite:
    # Satellite 클래스는 이제 K개의 글로벌 모델을 관리하고, 전역 클라이언트 클러스터링을 담당합니다.
    def __init__(self, num_zones, num_global_clusters):
        # __init__ 메소드는 Satellite 객체를 초기화합니다.
        self.global_models = {k: Net() for k in range(num_global_clusters)} # K개의 글로벌 모델을 딕셔너리 형태로 관리합니다.
        self.zones = [UAV(i) for i in range(num_zones)] # UAV는 이제 클러스터 정보를 갖지 않습니다.
        self.all_clients = [] # 시뮬레이션의 모든 클라이언트를 저장하는 리스트입니다.

    def distribute_clients(self, clients, client_zone_mapping):
        # distribute_clients 메소드는 클라이언트들을 해당 존에 분배하고, 모든 클라이언트 목록을 저장합니다.
        self.all_clients = clients
        for client_id, zone_id in client_zone_mapping.items():
            self.zones[zone_id].add_client(clients[client_id])

    def assign_clients_to_global_clusters(self, args):
        # assign_clients_to_global_clusters는 모든 클라이언트를 K개의 글로벌 클러스터 중 하나에 할당합니다.
        assignments = {k: [] for k in range(args.num_global_clusters)}
        print(f"Assigning {len(self.all_clients)} clients to {args.num_global_clusters} global clusters...")
        for client in self.all_clients:
            losses = []
            # 각 클라이언트는 K개의 글로벌 모델 각각에 대해 손실을 계산하여 '인터뷰'합니다.
            for k in range(args.num_global_clusters):
                model_state = self.global_models[k].state_dict()
                loss = client.evaluate_model_loss(model_state)
                losses.append(loss)
            # 가장 낮은 손실을 보인 클러스터에 클라이언트를 할당합니다.
            best_cluster_id = np.argmin(losses)
            assignments[best_cluster_id].append(client)
        
        dist_str = ", ".join([f"GC{k}: {len(clients)} clients" for k, clients in assignments.items()])
        print(f"Global cluster distribution: {dist_str}")
        return assignments

    def _aggregate_weights(self, client_weights):
        # _aggregate_weights는 FedAvg를 위한 헬퍼 함수입니다.
        if not client_weights:
            return None
        agg_weights = copy.deepcopy(client_weights[0])
        for key in agg_weights.keys():
            agg_weights[key] = torch.stack([cw[key].float() for cw in client_weights], 0).mean(0)
        return agg_weights

    def train_round(self, epoch, args):
        # train_round는 이제 글로벌 클러스터링 기반의 훈련을 수행합니다.
        print(f"--- Round {epoch+1}/{args.epochs} ---")
        
        # 1단계: 모든 클라이언트를 K개의 글로벌 클러스터 중 하나에 동적으로 할당합니다.
        client_assignments = self.assign_clients_to_global_clusters(args)

        # 2단계 & 3단계: 각 글로벌 클러스터별로 훈련을 진행하고 결과를 집계합니다.
        for k, assigned_clients in client_assignments.items():
            if not assigned_clients:
                print(f"  Global Cluster {k} has no clients, skipping.")
                continue

            print(f"  Training Global Cluster {k} with {len(assigned_clients)} clients...")
            cluster_model_state = self.global_models[k].state_dict()
            local_client_updates = []
            
            for client in assigned_clients:
                # 각 클라이언트는 자신이 속한 글로벌 클러스터의 모델로 훈련을 시작합니다.
                updated_weights = client.train(cluster_model_state, args.local_epochs, args.lr, args.momentum)
                local_client_updates.append(updated_weights)
            
            # 4단계: 해당 클러스터의 클라이언트 모델들을 집계하여 특정 글로벌 모델을 업데이트합니다.
            if local_client_updates:
                aggregated_weights = self._aggregate_weights(local_client_updates)
                self.global_models[k].load_state_dict(aggregated_weights)

    def test(self, test_loader):
        # test 메소드는 K개의 모델을 사용하여 앙상블 예측을 수행하고 성능을 평가합니다.
        for model in self.global_models.values():
            model.eval() # 모든 모델을 평가 모드로 설정합니다.

        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                # 각 데이터 포인트에 대해 K개의 모델로부터 예측을 모두 얻습니다.
                outputs = [model(data) for model in self.global_models.values()]
                
                # 각 데이터 포인트에 대해 가장 낮은 손실(가장 확실한 예측)을 보이는 모델을 선택합니다.
                losses = [F.nll_loss(output, target, reduction='none') for output in outputs]
                stacked_losses = torch.stack(losses, dim=1)
                best_model_indices = torch.argmin(stacked_losses, dim=1)
                
                # 최종 예측과 손실을 계산합니다.
                final_output = torch.stack([outputs[i][j] for j, i in enumerate(best_model_indices)])
                test_loss += F.nll_loss(final_output, target, reduction='sum').item()
                
                pred = final_output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        return test_loss, accuracy

# --- 4. Data Preparation ---
def get_data_and_entities(args):
    # get_data_and_entities 함수는 데이터셋을 로드하고, 클라이언트를 생성하며, 데이터 분포를 설정합니다.
    transform = transforms.Compose([
        transforms.ToTensor(), # 이미지를 PyTorch 텐서로 변환합니다.
        transforms.Normalize((0.1307,), (0.3081,)) # MNIST 데이터셋의 평균과 표준편차로 정규화합니다.
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform) # 훈련 데이터셋을 로드하거나 다운로드합니다.
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform) # 테스트 데이터셋을 로드하거나 다운로드합니다.
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False) # 테스트 데이터셋을 위한 DataLoader를 생성합니다.
    
    client_zone_mapping = {i: random.randint(0, args.num_zones - 1) for i in range(args.num_clients)} # 각 클라이언트를 랜덤하게 존에 할당합니다.
    
    # 매우 불균등한(non-IID) 데이터 분포를 생성합니다: 각 클라이언트는 2개의 숫자 클래스만 가집니다.
    num_labels_per_client = 2 # 클라이언트당 할당될 라벨(숫자)의 수입니다.
    labels = train_dataset.targets # 훈련 데이터셋의 모든 라벨을 가져옵니다.
    client_data_indices = {i: np.empty(0, dtype=np.int64) for i in range(args.num_clients)} # 각 클라이언트에게 할당될 데이터의 인덱스를 저장할 딕셔너리입니다.
    label_indices = [np.where(labels == i)[0] for i in range(10)] # 각 라벨(0-9)에 해당하는 데이터의 인덱스를 리스트로 저장합니다.
    client_labels = {i: np.random.choice(10, num_labels_per_client, replace=False) for i in range(args.num_clients)} # 각 클라이언트에게 2개의 고유한 라벨을 랜덤하게 할당합니다.

    for client_id, labels_for_client in client_labels.items():
        # 각 클라이언트와 할당된 라벨에 대해 반복합니다.
        for label in labels_for_client:
            # 클라이언트에게 할당된 각 라벨에 대해 반복합니다.
            num_clients_with_label = sum(1 for cid in range(args.num_clients) if label in client_labels[cid]) # 해당 라벨을 가진 총 클라이언트 수를 계산합니다.
            num_samples = len(label_indices[label]) // num_clients_with_label # 각 클라이언트에게 할당할 샘플 수를 계산합니다 (데이터를 균등하게 분배).
            rand_idx = np.random.choice(len(label_indices[label]), num_samples, replace=False) # 해당 라벨의 데이터 중에서 랜덤하게 샘플 인덱스를 선택합니다.
            client_data_indices[client_id] = np.concatenate((client_data_indices[client_id], label_indices[label][rand_idx])) # 선택된 샘플 인덱스를 클라이언트의 데이터 인덱스 목록에 추가합니다.
    
    clients = [] # UE 객체를 저장할 리스트입니다.
    for i in range(args.num_clients):
        # 모든 클라이언트에 대해 반복합니다.
        subset = Subset(train_dataset, client_data_indices[i]) # 클라이언트에게 할당된 인덱스를 사용하여 Subset을 생성합니다.
        loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True) # 해당 Subset을 위한 DataLoader를 생성합니다.
        clients.append(UE(i, loader)) # 생성된 UE 객체를 리스트에 추가합니다.

    print("--- Client & Zone & Data Distribution (Sample) ---") # 클라이언트, 존, 데이터 분포 정보를 출력합니다.
    zone_counts = {z:0 for z in range(args.num_zones)} # 존별 클라이언트 수를 저장할 딕셔너리입니다.
    for cid, zid in client_zone_mapping.items():
        # 클라이언트-존 매핑에 따라 반복합니다.
        zone_counts[zid] += 1 # 해당 존의 클라이언트 수를 1 증가시킵니다.
    print(f"Zone client counts: {zone_counts}") # 존별 클라이언트 수 분포를 출력합니다.

    return clients, test_loader, client_zone_mapping # 생성된 클라이언트 리스트, 테스트 로더, 클라이언트-존 매핑을 반환합니다.

# --- 5. Main Execution Loop ---
if __name__ == '__main__':
    # 이 스크립트가 직접 실행될 때 main 로직을 수행합니다.
    print("Initializing Hierarchical Federated Learning Simulation...") # 시뮬레이션 시작을 알립니다.
    
    clients, test_loader, client_zone_mapping = get_data_and_entities(args) # 데이터와 엔티티(클라이언트)를 준비합니다.

    satellite = Satellite(args.num_zones, args.num_global_clusters) # Satellite 객체를 생성합니다.
    satellite.distribute_clients(clients, client_zone_mapping) # 클라이언트들을 각 존에 분배합니다.
    
    test_losses = [] # 각 라운드 후의 테스트 손실을 저장할 리스트입니다.
    accuracies = [] # 각 라운드 후의 정확도를 저장할 리스트입니다.

    print("\nStarting Training...\n") # 훈련 시작을 알립니다.
    for epoch in range(args.epochs):
        # 지정된 전체 에포크 수만큼 훈련 라운드를 반복합니다.
        satellite.train_round(epoch, args) # 한 라운드의 훈련을 수행합니다.
        
        test_loss, accuracy = satellite.test(test_loader) # 현재 글로벌 모델의 성능을 테스트합니다.
        test_losses.append(test_loss) # 테스트 손실을 리스트에 추가합니다.
        accuracies.append(accuracy) # 정확도를 리스트에 추가합니다.
        
        print(f"Round {epoch+1}/{args.epochs} -> Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%\n") # 현재 라운드의 결과를 출력합니다.

    # --- 6. Results Visualization ---
    plt.figure(figsize=(10, 5)) # 그래프를 그릴 Figure 객체를 생성합니다 (가로 10, 세로 5 인치).
    plt.subplot(1, 2, 1) # 1행 2열의 첫 번째 서브플롯을 선택합니다.
    plt.plot(range(args.epochs), test_losses, marker='o') # 통신 라운드에 따른 테스트 손실을 그립니다.
    plt.title("Test Loss vs. Communication Rounds") # 그래프의 제목을 설정합니다.
    plt.xlabel("Communication Rounds") # x축의 라벨을 설정합니다.
    plt.ylabel("Test Loss") # y축의 라벨을 설정합니다.
    
    plt.subplot(1, 2, 2) # 1행 2열의 두 번째 서브플롯을 선택합니다.
    plt.plot(range(args.epochs), accuracies, marker='o', color='r') # 통신 라운드에 따른 정확도를 그립니다.
    plt.title("Accuracy vs. Communication Rounds") # 그래프의 제목을 설정합니다.
    plt.xlabel("Communication Rounds") # x축의 라벨을 설정합니다.
    plt.ylabel("Accuracy (%)") # y축의 라벨을 설정합니다.
    
    plt.tight_layout() # 서브플롯들이 겹치지 않도록 레이아웃을 조정합니다.
    plt.show() # 그래프를 화면에 표시합니다.

    print("\nHierarchical Federated Learning Simulation Finished.") # 시뮬레이션 종료를 알립니다.
    # K개의 글로벌 모델을 state_dict 딕셔너리로 저장합니다.
    model_states = {k: model.state_dict() for k, model in satellite.global_models.items()}
    torch.save(model_states, "hierarchical_federated_models.pt")
    print("Final global models saved as 'hierarchical_federated_models.pt'") # 모델이 저장되었음을 알립니다.
