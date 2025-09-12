# fedavg_mnist_label_clusters.py
import numpy as np, random, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ===== Config =====
SEED = 42
K = 6                            # 총 클라이언트(=드론) 수
CLIENTS_PER_CLUSTER = [2, 2, 2]  # 각 클러스터에 몇 개의 클라이언트?
N_CLUSTERS = len(CLIENTS_PER_CLUSTER)
E = 1                            # 로컬 에폭
ROUNDS = 20                      # 통신 라운드 수
PARTICIPANTS_PER_ROUND = K       # 라운드당 참여 클라이언트 수 (여기선 전원 참여)
BATCH = 64
LR = 0.05
MOMENTUM = 0.9
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 라벨 기반 클러스터 정의
CLUSTER_LABEL_MAP = {
    0: [0, 1, 2, 3],
    1: [4, 5, 6],
    2: [7, 8, 9],
}

# ===== Reproducibility =====
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ===== Model =====
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32*7*7, 64)
        self.fc2 = nn.Linear(64, 10)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))  # 28->14
        x = F.relu(F.max_pool2d(self.conv2(x), 2))  # 14->7
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def init_model():
    return SmallCNN().to(DEVICE)

def get_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

def set_state(model, state):
    model.load_state_dict(state, strict=True)

def average_states(weighted_states):
    # weighted_states: list of (n_k, state_dict)
    total = sum(n for n, _ in weighted_states)
    keys = weighted_states[0][1].keys()
    out = {}
    for k in keys:
        out[k] = sum(n * s[k] for n, s in weighted_states) / total
    return out

# ===== Data =====
def load_mnist():
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = datasets.MNIST(root="./data", train=True, download=True, transform=tmf)  # typo 보호
    test_set  = datasets.MNIST(root="./data", train=False, download=True, transform=tmf)
    return train_set, test_set

# 위 오타 수정 버전
def load_mnist():
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test_set  = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)
    return train_set, test_set

# ===== Partitioning: label-based clusters -> per-client splits =====
def build_clients_by_label_clusters(train_set, clients_per_cluster, cluster_label_map):
    """
    각 클러스터의 라벨 집합에 해당하는 train 인덱스를 모은 뒤,
    클러스터 내부에서 clients_per_cluster[c] 개수만큼 균등 분할하여
    client_idx(길이 K의 리스트)와 각 클라의 클러스터 번호 cluster_of(길이 K 배열)를 만든다.
    """
    targets = np.array(train_set.targets)
    client_idx = []
    cluster_of = []
    for c, n_clients in enumerate(clients_per_cluster):
        labels = cluster_label_map[c]
        idxs = np.where(np.isin(targets, labels))[0]
        np.random.shuffle(idxs)
        splits = np.array_split(idxs, n_clients)
        for s in splits:
            client_idx.append(np.array(s, dtype=int))
            cluster_of.append(c)
    cluster_of = np.array(cluster_of)
    return client_idx, cluster_of

# ===== Local Train / Evaluate =====
def local_train(state, subset, epochs=E):
    model = init_model()
    set_state(model, state)
    dl = DataLoader(subset, batch_size=BATCH, shuffle=True, num_workers=0)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
    return get_state(model)

@torch.no_grad()
def evaluate(state, test_set):
    """Global test acc (전체 MNIST 테스트셋 기준)"""
    model = init_model()
    set_state(model, state)
    model.eval()
    dl = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=0)
    correct, total, loss_sum = 0, 0, 0.0
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    for xb, yb in dl:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        loss_sum += loss_fn(logits, yb).item()
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
    acc = correct / total
    loss = loss_sum / total
    return acc, loss

@torch.no_grad()
def evaluate_on_labels(state, test_set, labels):
    """In-Cluster acc: 특정 라벨만 필터링한 테스트셋 기준 정확도"""
    targets = np.array(test_set.targets)
    mask = np.isin(targets, labels)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return float("nan"), float("nan")
    subset = Subset(test_set, idx)
    return evaluate(state, subset)

# ===== Main FL Loop =====
def main():
    # 1) 데이터 로드
    train_set, test_set = load_mnist()

    # 2) 클러스터 라벨 기준으로 클라이언트 분할
    client_idx, cluster_of = build_clients_by_label_clusters(
        train_set,
        CLIENTS_PER_CLUSTER,
        CLUSTER_LABEL_MAP
    )
    assert len(client_idx) == K, f"Expected K={K}, got {len(client_idx)}"
    assert len(cluster_of) == K

    client_subsets = [Subset(train_set, idx) for idx in client_idx]

    # 3) 클러스터별 전역 모델 초기화
    global_states = [get_state(init_model()) for _ in range(N_CLUSTERS)]

    # 4) 라운드 루프
    print(f"[INFO] Device={DEVICE} | K={K}, ROUNDS={ROUNDS}, E={E}, clusters={N_CLUSTERS}")
    print(f"[INFO] Cluster label map: {CLUSTER_LABEL_MAP}")

    for t in range(1, ROUNDS + 1):
        # 이번 라운드 참여자 선택 (전원 참여 또는 일부 참여)
        if PARTICIPANTS_PER_ROUND >= K:
            selected = np.arange(K)
        else:
            selected = np.random.choice(K, size=PARTICIPANTS_PER_ROUND, replace=False)

        # 클러스터별 업데이트 모으기
        weighted_states_by_cluster = [[] for _ in range(N_CLUSTERS)]

        for k in selected:
            c = cluster_of[k]
            subset = client_subsets[k]
            n_k = len(subset)
            if n_k == 0:
                continue
            new_state = local_train(global_states[c], subset, epochs=E)
            weighted_states_by_cluster[c].append((n_k, new_state))

        # 클러스터별 집계(FedAvg)
        for c in range(N_CLUSTERS):
            if len(weighted_states_by_cluster[c]) > 0:
                global_states[c] = average_states(weighted_states_by_cluster[c])

        # ===== 평가: 클러스터별 Global Acc, In-Cluster Acc 모두 출력 =====
        global_accs = []
        in_cluster_accs = []
        for c in range(N_CLUSTERS):
            acc_g, _ = evaluate(global_states[c], test_set)
            global_accs.append(acc_g)

            labels = CLUSTER_LABEL_MAP[c]
            acc_ic, _ = evaluate_on_labels(global_states[c], test_set, labels)
            in_cluster_accs.append(acc_ic)

        g_line = " | ".join([f"C{c}:{global_accs[c]*100:5.2f}%" for c in range(N_CLUSTERS)])
        ic_line = " | ".join([f"C{c}:{in_cluster_accs[c]*100:5.2f}%" for c in range(N_CLUSTERS)])
        print(f"Round {t:02d} | Global Acc    | {g_line}")
        print(f"          | In-Cluster Acc | {ic_line}")

    # 5) 최종 결과
    print("\n[DONE] Final accuracies:")
    for c in range(N_CLUSTERS):
        acc_g, loss_g = evaluate(global_states[c], test_set)
        acc_ic, loss_ic = evaluate_on_labels(global_states[c], test_set, CLUSTER_LABEL_MAP[c])
        print(f"  - Cluster {c} (labels {CLUSTER_LABEL_MAP[c]}): "
              f"Global Acc={acc_g*100:.2f}%  (loss={loss_g:.4f}) | "
              f"In-Cluster Acc={acc_ic*100:.2f}%  (loss={loss_ic:.4f})")

if __name__ == "__main__":
    main()
