# fedavg_mnist_hierarchical_clustered.py
import numpy as np, random, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ===== Config =====
SEED = 42
NUM_CLIENTS = 100
NUM_ZONES = 6
E = 2                  # 로컬 에폭
ROUNDS = 30            # 통신 라운드 수
PARTICIPANTS_PER_ROUND = NUM_CLIENTS   # 전원 참여 (원하면 샘플링)
BATCH = 64
LR = 0.01
MOMENTUM = 0.9
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 클러스터 정의 (라벨 기반)
CLUSTER_LABEL_MAP = {
    0: [0, 1, 2, 3],
    1: [4, 5, 6],
    2: [7, 8, 9],
}
N_CLUSTERS = len(CLUSTER_LABEL_MAP)

# ===== Reproducibility =====
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)

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

def average_states_weighted(weighted_states):
    """
    weighted_states: list of (n_k, state_dict)
    데이터 수(n_k) 가중 FedAvg
    """
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
    train_set = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test_set  = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)
    return train_set, test_set

# ===== Partitioning =====
def assign_zones_random(num_clients, num_zones, seed=42):
    rng = np.random.RandomState(seed)
    return {cid: int(rng.randint(0, num_zones)) for cid in range(num_clients)}

def auto_clients_per_cluster(num_clients, n_clusters):
    """
    num_clients를 n_clusters로 최대한 균등 분할.
    예) 100, 3 -> [34, 33, 33]
    """
    base = num_clients // n_clusters
    rem = num_clients % n_clusters
    arr = [base] * n_clusters
    for i in range(rem):
        arr[i] += 1
    return arr

def build_clients_by_label_clusters(train_set, clients_per_cluster, cluster_label_map, seed=42):
    """
    클러스터 라벨 집합에 해당하는 train 인덱스를 모아
    각 클러스터 내부에서 clients_per_cluster[c]로 균등 분할(중복 없음).
    반환:
      - client_subsets: 길이 NUM_CLIENTS 의 Subset 리스트
      - cluster_of: 각 클라이언트의 클러스터 ID (np.array)
    """
    rng = np.random.RandomState(seed)
    targets = np.array(train_set.targets)
    client_subsets, cluster_of = [], []

    # 클러스터마다 라벨 인덱스 모아서 섞고 → n_clients로 등분
    for c, n_clients in enumerate(clients_per_cluster):
        labels = cluster_label_map[c]
        idxs = np.where(np.isin(targets, labels))[0]
        rng.shuffle(idxs)
        splits = np.array_split(idxs, n_clients)
        for s in splits:
            client_subsets.append(Subset(train_set, np.array(s, dtype=int)))
            cluster_of.append(c)

    cluster_of = np.array(cluster_of, dtype=int)
    return client_subsets, cluster_of

# ===== Train / Evaluate =====
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
    return get_state(model), len(subset)

@torch.no_grad()
def evaluate(state, dataset_or_subset):
    model = init_model()
    set_state(model, state)
    model.eval()
    dl = DataLoader(dataset_or_subset, batch_size=256, shuffle=False, num_workers=0)
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total, correct, loss_sum = 0, 0, 0.0
    for xb, yb in dl:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        loss_sum += loss_fn(logits, yb).item()
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
    return correct/total, loss_sum/total

@torch.no_grad()
def evaluate_on_labels(state, test_set, labels):
    targets = np.array(test_set.targets)
    mask = np.isin(targets, labels)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return float("nan"), float("nan")
    subset = Subset(test_set, idx)
    return evaluate(state, subset)

# ===== Main (HFL + Clustered) =====
def main():
    # 1) 데이터
    train_set, test_set = load_mnist()

    # 2) 클라이언트/클러스터 구성
    clients_per_cluster = auto_clients_per_cluster(NUM_CLIENTS, N_CLUSTERS)
    client_subsets, cluster_of = build_clients_by_label_clusters(
        train_set, clients_per_cluster, CLUSTER_LABEL_MAP, seed=SEED
    )
    assert len(client_subsets) == NUM_CLIENTS
    assert len(cluster_of) == NUM_CLIENTS

    # 3) Zone 랜덤 할당
    zone_of = assign_zones_random(NUM_CLIENTS, NUM_ZONES, seed=SEED)

    # 4) 클러스터별 모델 초기화
    cluster_states = [get_state(init_model()) for _ in range(N_CLUSTERS)]

    # 5) 로그
    print(f"[INFO] Device={DEVICE} | K={NUM_CLIENTS}, ZONES={NUM_ZONES}, ROUNDS={ROUNDS}, E={E}, clusters={N_CLUSTERS}")
    print(f"[INFO] Cluster label map: {CLUSTER_LABEL_MAP}")
    print(f"[INFO] Clients per cluster: {clients_per_cluster}")
    zone_counts = {z: 0 for z in range(NUM_ZONES)}
    for cid in range(NUM_CLIENTS):
        zone_counts[zone_of[cid]] += 1
    print(f"[INFO] Zone client counts: {zone_counts}")

    # 6) 라운드 루프
    for t in range(1, ROUNDS + 1):
        # 선택(전원 참여)
        if PARTICIPANTS_PER_ROUND >= NUM_CLIENTS:
            selected = np.arange(NUM_CLIENTS)
        else:
            selected = np.random.choice(NUM_CLIENTS, size=PARTICIPANTS_PER_ROUND, replace=False)

        # --- 1차 집계: Zone 내 클러스터별 FedAvg 준비 ---
        # zone_cluster_updates[z][c] = list of (n_k, state)
        zone_cluster_updates = [[[] for _ in range(N_CLUSTERS)] for _ in range(NUM_ZONES)]

        # 로컬 학습 (각 클라는 "자신의 클러스터 모델"에서 시작)
        for k in selected:
            c = cluster_of[k]
            z = zone_of[k]
            subset = client_subsets[k]
            if len(subset) == 0:
                continue
            new_state, n_k = local_train(cluster_states[c], subset, epochs=E)
            zone_cluster_updates[z][c].append((n_k, new_state))

        # --- Zone 대표(클러스터별) 산출 ---
        # zone_cluster_states[z][c] = (N_zc, state_dict)
        zone_cluster_states = [[None for _ in range(N_CLUSTERS)] for _ in range(NUM_ZONES)]
        for z in range(NUM_ZONES):
            for c in range(N_CLUSTERS):
                if len(zone_cluster_updates[z][c]) > 0:
                    st = average_states_weighted(zone_cluster_updates[z][c])
                    N_zc = sum(n for n, _ in zone_cluster_updates[z][c])
                    zone_cluster_states[z][c] = (N_zc, st)

        # --- 2차 집계: 클러스터별로 Zone 대표들을 FedAvg → 최종 클러스터 모델 업데이트 ---
        for c in range(N_CLUSTERS):
            pool = []
            for z in range(NUM_ZONES):
                item = zone_cluster_states[z][c]
                if item is not None:
                    pool.append(item)  # (N_zc, state)
            if len(pool) > 0:
                cluster_states[c] = average_states_weighted(pool)

        # --- 평가: 클러스터별 Global / In-Cluster ---
        global_accs, in_cluster_accs = [], []
        for c in range(N_CLUSTERS):
            acc_g, _ = evaluate(cluster_states[c], test_set)
            global_accs.append(acc_g)
            labels = CLUSTER_LABEL_MAP[c]
            acc_ic, _ = evaluate_on_labels(cluster_states[c], test_set, labels)
            in_cluster_accs.append(acc_ic)

        g_line = " | ".join([f"C{c}:{global_accs[c]*100:5.2f}%" for c in range(N_CLUSTERS)])
        ic_line = " | ".join([f"C{c}:{in_cluster_accs[c]*100:5.2f}%" for c in range(N_CLUSTERS)])
        print(f"Round {t:02d} | Global Acc    | {g_line}")
        print(f"          | In-Cluster Acc | {ic_line}")

    # 7) 최종 결과
    print("\n[DONE] Final accuracies:")
    for c in range(N_CLUSTERS):
        acc_g, loss_g = evaluate(cluster_states[c], test_set)
        acc_ic, loss_ic = evaluate_on_labels(cluster_states[c], test_set, CLUSTER_LABEL_MAP[c])
        print(f"  - Cluster {c} (labels {CLUSTER_LABEL_MAP[c]}): "
              f"Global Acc={acc_g*100:.2f}%  (loss={loss_g:.4f}) | "
              f"In-Cluster Acc={acc_ic*100:.2f}%  (loss={loss_ic:.4f})")

if __name__ == "__main__":
    main()
