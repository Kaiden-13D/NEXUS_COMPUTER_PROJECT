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
    # UAV 클래스는 계층 구조의 중간 계층인 UAV(존 서버)를 나타냅니다.
    # 자신의 존 내 모든 클라이언트의 훈련을 조율하고, 로컬 클러스터링을 수행하는 중요한 역할을 합니다.
    def __init__(self, id, num_clusters):
        # __init__ 메소드는 UAV 객체를 초기화합니다.
        self.id = id # UAV의 고유 ID입니다.
        self.clients = [] # 이 UAV에 속한 클라이언트 목록입니다.
        self.num_clusters = num_clusters # 이 UAV가 관리할 클러스터의 수입니다.
        self.cluster_models = {c: Net() for c in range(num_clusters)} # 각 클러스터를 위한 로컬 모델들을 딕셔너리 형태로 유지합니다.

    def add_client(self, client):
        # add_client 메소드는 UAV에 클라이언트를 추가합니다.
        self.clients.append(client)

    def update_cluster_models_from_global(self, global_model_state):
        # update_cluster_models_from_global 메소드는 위성으로부터 새로운 글로벌 모델을 받아 모든 로컬 클러스터 모델을 재설정합니다.
        # 이는 중요한 '동기화' 단계입니다.
        for c_model in self.cluster_models.values():
            # 모든 클러스터 모델에 대해 반복합니다.
            c_model.load_state_dict(copy.deepcopy(global_model_state)) # 글로벌 모델의 가중치를 복사하여 클러스터 모델을 초기화합니다.

    def assign_clients_to_clusters(self):
        # assign_clients_to_clusters 메소드는 동적 로컬 클러스터링의 핵심입니다.
        # 각 클라이언트는 UAV의 현재 클러스터 모델 중 자신의 개인 데이터에 대해 최소 손실을 내는 클러스터에 할당됩니다.
        # 이 그룹화는 일시적이며, 변화에 적응하기 위해 매 라운드마다 재평가됩니다.
        assignments = {c: [] for c in range(self.num_clusters)} # 클러스터 할당 결과를 저장할 딕셔너리를 초기화합니다.
        if not self.clients:
            # 클라이언트가 없으면 빈 할당을 반환합니다.
            return assignments

        print(f"  Zone {self.id}: Assigning {len(self.clients)} clients to {self.num_clusters} clusters...") # 클라이언트 할당 시작을 알립니다.
        for client in self.clients:
            # 각 클라이언트에 대해 반복합니다.
            losses = [] # 각 클러스터 모델에 대한 손실을 저장할 리스트입니다.
            # 각 클라이언트는 모든 후보 클러스터 모델을 '인터뷰'합니다.
            for cluster_id in range(self.num_clusters):
                # 각 클러스터 ID에 대해 반복합니다.
                cluster_model_state = self.cluster_models[cluster_id].state_dict() # 해당 클러스터 모델의 가중치를 가져옵니다.
                loss = client.evaluate_model_loss(cluster_model_state) # 클라이언트의 데이터로 손실을 평가합니다.
                losses.append(loss) # 계산된 손실을 리스트에 추가합니다.
            
            # 클라이언트는 자신의 데이터를 가장 잘 '이해하는' 클러스터에 할당됩니다.
            best_cluster_id = np.argmin(losses) # 손실이 가장 작은 클러스터의 인덱스를 찾습니다.
            assignments[best_cluster_id].append(client) # 해당 클러스터에 클라이언트를 할당합니다.
        
        dist_str = ", ".join([f"C{c}: {len(clients)} clients" for c, clients in assignments.items()]) # 클러스터별 클라이언트 수 분포를 문자열로 만듭니다.
        print(f"  Zone {self.id}: Cluster distribution: {dist_str}") # 클러스터 분포를 출력합니다.

        return assignments # 최종 클라이언트 할당 결과를 반환합니다.

    def train_zone(self, local_epochs, lr, momentum):
        # train_zone 메소드는 존 내에서 한 라운드의 전체 훈련을 조율합니다.
        # 1. 이 라운드를 위해 클라이언트를 클러스터에 할당합니다.
        client_assignments = self.assign_clients_to_clusters()
        
        updated_cluster_weights = {} # 업데이트된 클러스터 가중치를 저장할 딕셔너리입니다.

        # 2. 각 클러스터에 대해, 할당된 클라이언트들로 훈련을 시작합니다.
        for cluster_id, assigned_clients in client_assignments.items():
            # 클러스터 할당 결과에 대해 반복합니다.
            if not assigned_clients:
                # 클러스터에 할당된 클라이언트가 없으면, 모델은 변경되지 않습니다.
                updated_cluster_weights[cluster_id] = self.cluster_models[cluster_id].state_dict() # 현재 모델 가중치를 그대로 저장합니다.
                continue # 다음 클러스터로 넘어갑니다.

            cluster_model_state = self.cluster_models[cluster_id].state_dict() # 현재 클러스터 모델의 가중치를 가져옵니다.
            local_client_updates = [] # 로컬 클라이언트들의 업데이트된 가중치를 저장할 리스트입니다.
            
            for client in assigned_clients:
                # 할당된 각 클라이언트에 대해 훈련을 수행합니다.
                updated_weights = client.train(cluster_model_state, local_epochs, lr, momentum) # 클라이언트 훈련을 호출합니다.
                local_client_updates.append(updated_weights) # 업데이트된 가중치를 리스트에 추가합니다.
            
            # 3. 이 클러스터의 결과를 집계합니다.
            # 이 집계는 클라이언트들이 유사성에 따라 그룹화되었기 때문에 더 안정적입니다.
            if local_client_updates:
                # 로컬 업데이트가 있는 경우에만 집계합니다.
                aggregated_weights = self._aggregate_weights(local_client_updates) # 가중치를 평균내어 집계합니다.
                self.cluster_models[cluster_id].load_state_dict(aggregated_weights) # 집계된 가중치로 클러스터 모델을 업데이트합니다.
                updated_cluster_weights[cluster_id] = aggregated_weights # 업데이트된 가중치를 저장합니다.
        
        return updated_cluster_weights # 모든 클러스터의 업데이트된 가중치를 반환합니다.

    def get_zone_summary_model(self):
        # get_zone_summary_model 메소드는 "존 요약"을 수행합니다.
        # 모든 전문화된 로컬 클러스터 모델들의 지식을 단일 평균 모델로 정제합니다.
        # 이 요약본이 위성으로 전송됩니다; 위성은 개별 클러스터 모델을 직접 보지 않습니다.
        if not self.cluster_models:
            # 클러스터 모델이 없으면 None을 반환합니다.
            return None
        
        cluster_model_states = [model.state_dict() for model in self.cluster_models.values()] # 모든 클러스터 모델의 가중치를 리스트로 가져옵니다.
        zone_summary_state = self._aggregate_weights(cluster_model_states) # 클러스터 모델들의 가중치를 평균내어 존 요약 가중치를 만듭니다.
        
        zone_summary_model = Net() # 존 요약 모델을 위한 새로운 Net 객체를 생성합니다.
        zone_summary_model.load_state_dict(zone_summary_state) # 요약 가중치를 모델에 로드합니다.
        return zone_summary_model # 존 요약 모델을 반환합니다.

    def _aggregate_weights(self, client_weights):
        # _aggregate_weights는 FedAvg(Federated Averaging)를 위한 헬퍼 함수입니다.
        if not client_weights:
            # 가중치 리스트가 비어있으면 None을 반환합니다.
            return None
        
        agg_weights = copy.deepcopy(client_weights[0]) # 첫 번째 클라이언트의 가중치를 기준으로 집계할 가중치를 초기화합니다.
        for k in agg_weights.keys():
            # 가중치 딕셔너리의 모든 키(계층)에 대해 반복합니다.
            agg_weights[k] = torch.stack([cw[k].float() for cw in client_weights], 0).mean(0) # 모든 클라이언트의 해당 계층 가중치를 쌓아서 평균을 계산합니다.
        return agg_weights # 집계된 가중치를 반환합니다.

class Satellite:
    # Satellite 클래스는 계층 구조의 최상위에 있는 중앙 글로벌 서버를 나타냅니다.
    # 전체 훈련 과정을 관리하고, 모든 존으로부터 요약본을 집계하며, 단일 글로벌 모델을 유지합니다.
    def __init__(self, num_zones, num_clusters_per_zone):
        # __init__ 메소드는 Satellite 객체를 초기화합니다.
        self.global_model = Net() # 중앙 글로벌 모델입니다.
        self.zones = [UAV(i, num_clusters_per_zone) for i in range(num_zones)] # 지정된 수만큼 UAV(존) 객체를 생성하여 리스트에 저장합니다.

    def distribute_clients(self, clients, client_zone_mapping):
        # distribute_clients 메소드는 클라이언트들을 해당 존에 분배합니다.
        for client_id, zone_id in client_zone_mapping.items():
            # 클라이언트-존 매핑에 따라 반복합니다.
            self.zones[zone_id].add_client(clients[client_id]) # 해당 존에 클라이언트를 추가합니다.

    def train_round(self, epoch, args):
        # train_round 메소드는 한 번의 전체 계층적 훈련 라운드를 실행합니다.
        print(f"--- Round {epoch+1}/{args.epochs} ---") # 현재 라운드 번호를 출력합니다.
        
        # 1단계: 동기화. 단일 글로벌 모델을 모든 UAV에 방송합니다.
        # UAV들은 이를 사용하여 로컬 클러스터 모델을 재설정합니다.
        global_model_state = self.global_model.state_dict() # 현재 글로벌 모델의 가중치를 가져옵니다.
        for zone in self.zones:
            # 모든 존에 대해 반복합니다.
            zone.update_cluster_models_from_global(global_model_state) # 각 존의 클러스터 모델들을 글로벌 모델로 업데이트합니다.

        # 2단계: 로컬 훈련 및 클러스터링.
        # 모든 존이 클라이언트 할당 및 훈련 단계를 수행하도록 합니다.
        zone_summary_models = [] # 각 존의 요약 모델을 저장할 리스트입니다.
        for zone in self.zones:
            # 모든 존에 대해 반복합니다.
            zone.train_zone(args.local_epochs, args.lr, args.momentum) # 존 내 훈련을 시작합니다.
            # 3단계: 수집. 모든 UAV로부터 "존 요약"을 수집합니다.
            summary_model = zone.get_zone_summary_model() # 존 요약 모델을 가져옵니다.
            if summary_model:
                # 요약 모델이 있는 경우에만 추가합니다.
                zone_summary_models.append(summary_model.state_dict()) # 요약 모델의 가중치를 리스트에 추가합니다.
        
        # 4단계: 글로벌 집계.
        # 수집된 존 요약들을 평균내어 단일 글로벌 모델을 업데이트합니다.
        if zone_summary_models:
            # 존 요약 모델이 있는 경우에만 집계합니다.
            self._aggregate_global_model(zone_summary_models) # 글로벌 모델을 집계합니다.

    def _aggregate_global_model(self, zone_model_weights):
        # _aggregate_global_model 메소드는 존 모델들을 집계하여 글로벌 모델을 업데이트합니다 (FedAvg).
        global_dict = self.global_model.state_dict() # 현재 글로벌 모델의 가중치 딕셔너리를 가져옵니다.
        for k in global_dict.keys():
            # 가중치 딕셔너리의 모든 키(계층)에 대해 반복합니다.
            global_dict[k] = torch.stack([zone_weights[k].float() for zone_weights in zone_model_weights], 0).mean(0) # 모든 존의 해당 계층 가중치를 쌓아서 평균을 계산합니다.
        self.global_model.load_state_dict(global_dict) # 집계된 가중치로 글로벌 모델을 업데이트합니다.

    def test(self, test_loader):
        # test 메소드는 테스트 데이터셋에서 글로벌 모델의 성능을 평가합니다.
        self.global_model.eval() # 모델을 평가 모드로 설정합니다.
        test_loss = 0 # 테스트 손실을 저장할 변수입니다.
        correct = 0 # 정확하게 예측된 샘플 수를 저장할 변수입니다.
        with torch.no_grad():
            # 그래디언트 계산을 비활성화합니다.
            for data, target in test_loader:
                # 테스트 데이터 로더에서 배치 단위로 데이터를 가져옵니다.
                output = self.global_model(data) # 모델 예측을 수행합니다.
                test_loss += F.nll_loss(output, target, reduction='sum').item() # 배치 손실을 합산하여 누적합니다.
                pred = output.argmax(dim=1, keepdim=True) # 가장 높은 확률을 가진 클래스를 예측값으로 선택합니다.
                correct += pred.eq(target.view_as(pred)).sum().item() # 예측이 정답과 일치하는 경우 카운트를 증가시킵니다.
        test_loss /= len(test_loader.dataset) # 전체 테스트 데이터셋에 대한 평균 손실을 계산합니다.
        accuracy = 100. * correct / len(test_loader.dataset) # 정확도를 백분율로 계산합니다.
        return test_loss, accuracy # 테스트 손실과 정확도를 반환합니다.

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

    satellite = Satellite(args.num_zones, args.num_clusters_per_zone) # Satellite 객체를 생성합니다.
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
    torch.save(satellite.global_model.state_dict(), "hierarchical_federated_model.pt") # 최종 글로벌 모델의 가중치를 파일에 저장합니다.
    print("Final global model saved as 'hierarchical_federated_model.pt'") # 모델이 저장되었음을 알립니다.
