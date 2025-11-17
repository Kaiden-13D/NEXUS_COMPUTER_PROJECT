# HPFL 시스템 프로세스 상세 문서화

## 1. Client → UAV → Satellite 정보 흐름

### 1.1 전체 흐름 개요

```
Client → (Local Training) → Model Update → UAV → (Batching) → Satellite → (Clustering) → Cluster Models
                                                                                              ↓
Client ← (Cluster Model) ← UAV ← (Broadcast) ← Satellite ← (Aggregation)
```

### 1.2 상세 프로세스

#### Step 1: Client에서 Local Training 및 전송

**위치**: `src/experiments/abl1.py`, `bl3.py` 등

```python
# Client가 cluster model 또는 global model로 학습
if c.id in client_to_cluster:
    cluster_id = client_to_cluster[c.id]
    if cluster_id in cluster_models_dict:
        cluster_model = SimpleCNN(num_classes).to(Config.DEVICE)
        cluster_model.load_state_dict(cluster_models_dict[cluster_id])
        sd = c.train(cluster_model)  # Cluster model로 학습
    else:
        sd = c.train(model)  # Global model로 학습
else:
    sd = c.train(model)  # Global model로 학습

# Model state_dict를 bytes로 변환
payload = get_state_dict_bytes(sd)

# Link를 통해 전송 (client_id 포함)
if sent := await c.link.transmit(payload, cost_meter=COST):
    await uav_in_qs[int(c.uav_id[-1])].put(
        Packet(c.id, c.uav_id, time.time(), sent)  # Packet에 client_id 포함
    )
```

**핵심 포인트**:
- `Packet` 객체에 `src` 필드로 `client_id`가 포함됨 (`c.id`)
- `dst` 필드로 `uav_id`가 포함됨 (`c.uav_id`)
- `data` 필드에 모델 state_dict의 bytes가 포함됨

#### Step 2: UAV에서 Batching 및 Satellite로 전송

**위치**: `src/core/fl_nodes.py` - `UAV.run()`, `UAV.flush()`

```python
async def run(self):
    """UAV run loop (batch packet processing)"""
    batch = []
    while True:
        try:
            pkt = await asyncio.wait_for(self.in_q.get(), 0.8)
            batch.append(pkt)  # Packet 객체를 batch에 추가
            if len(batch) >= 8:
                await self.flush(batch)
                batch = []
        except asyncio.TimeoutError:
            if batch:
                await self.flush(batch)
                batch = []

async def flush(self, batch: List[Packet]):
    """Send batch packets to satellite"""
    payload = b"".join(p.data for p in batch)  # 모든 패킷의 data를 합침
    if await self.link.transmit(payload):
        # Packet 객체 전체를 out_q에 전달 (client_id 정보 보존)
        for pkt in batch:
            await self.out_q.put(pkt)  # Packet 객체 그대로 전달
```

**핵심 포인트**:
- UAV는 여러 client의 패킷을 batch로 묶어서 전송 (8개씩 또는 timeout 시)
- `Packet` 객체가 그대로 전달되므로 `client_id` 정보가 보존됨
- 실제 네트워크 전송 시에는 data만 합쳐서 전송하지만, 내부적으로는 Packet 객체를 유지

#### Step 3: Satellite에서 수신 및 Buffer 저장

**위치**: `src/core/fl_nodes.py` - `SatAgg.run()`

```python
async def run(self):
    """Satellite run loop (collect model updates)"""
    while True:
        pkt = await self.in_q.get()
        # Packet 객체에서 client_id와 state_dict 추출
        if isinstance(pkt, Packet):
            client_id = pkt.src  # Packet의 src 필드에서 client_id 추출
            state_dict = bytes_to_state_dict(pkt.data)  # data를 state_dict로 변환
            self.buffer.append((client_id, state_dict))  # (client_id, state_dict) 튜플로 저장
        else:
            # Backward compatibility: bytes only
            state_dict = bytes_to_state_dict(pkt)
            self.buffer.append((None, state_dict))
```

**핵심 포인트**:
- `sat.buffer`는 `List[Tuple[client_id, state_dict]]` 형태
- 각 업데이트마다 어떤 client에서 왔는지 추적 가능
- 이 정보는 이후 clustering 및 cluster 할당에 사용됨

#### Step 4: Satellite에서 Clustering 및 Cluster 할당

**위치**: `src/experiments/abl1.py`, `bl3.py` - Clustering 섹션

```python
# Buffer에서 client_id와 state_dict 분리
buffer_state_dicts = [sd for _, sd in sat.buffer]
buffer_client_ids = [cid for cid, _ in sat.buffer]

# Model similarity 기반 clustering
similarity_matrix = compute_similarity_matrix(buffer_state_dicts)
num_clusters = max(2, min(5, len(buffer_state_dicts) // 3))
cluster_labels = cluster_models(buffer_state_dicts, num_clusters=num_clusters)

# Cluster별 aggregation
new_cluster_models = cluster_based_aggregation(
    buffer_state_dicts,
    cluster_labels,
    weights=client_weights
)

# client_to_cluster 매핑 업데이트
for idx, client_id in enumerate(buffer_client_ids):
    if client_id and idx < len(cluster_labels):
        client_to_cluster[client_id] = cluster_labels[idx]  # Client → Cluster 매핑 저장
```

**핵심 포인트**:
- `buffer_client_ids`와 `cluster_labels`의 인덱스가 일치함
- `client_to_cluster` 딕셔너리에 `{client_id: cluster_id}` 매핑이 저장됨
- 이 매핑은 다음 라운드에서 client가 어떤 cluster model을 사용할지 결정하는 데 사용됨

#### Step 5: Cluster Model 업데이트

```python
# Cluster model 업데이트 (이전 round의 cluster model과 현재 round의 cluster model을 weighted average)
for cluster_id, cluster_model_state in new_cluster_models.items():
    if cluster_id in cluster_models_dict:
        # 기존 cluster model과 새 cluster model을 weighted average
        current_cluster_size = sum(1 for label in cluster_labels if label == cluster_id)
        cluster_models_dict[cluster_id] = fedavg(
            [cluster_models_dict[cluster_id], cluster_model_state],
            weights=[1.0, float(current_cluster_size)]  # 기존:1, 현재:cluster_size
        )
    else:
        # 새로운 cluster인 경우
        cluster_models_dict[cluster_id] = cluster_model_state
```

**핵심 포인트**:
- Cluster model은 매 라운드마다 업데이트됨
- 이전 라운드의 cluster model과 현재 라운드의 cluster model을 weighted average
- Weight는 cluster size에 비례

#### Step 6: 다음 라운드에서 Client가 Cluster Model 사용

**위치**: 다음 라운드의 Client Training 단계

```python
# 다음 라운드에서 client가 학습할 때
if c.id in client_to_cluster:  # 이전 라운드에서 cluster 할당이 있었는지 확인
    cluster_id = client_to_cluster[c.id]  # Client의 cluster_id 조회
    if cluster_id in cluster_models_dict:  # 해당 cluster model이 존재하는지 확인
        cluster_model = SimpleCNN(num_classes).to(Config.DEVICE)
        cluster_model.load_state_dict(cluster_models_dict[cluster_id])  # Cluster model 로드
        sd = c.train(cluster_model)  # Cluster model로 학습
```

**핵심 포인트**:
- Client는 `client_to_cluster` 딕셔너리를 통해 자신의 cluster_id를 알 수 있음
- 하지만 실제로는 **Satellite에서 직접 client에게 cluster_id를 전달하지 않음**
- 대신, **다음 라운드에서 client가 학습할 때 자신의 cluster_id를 조회**하여 해당 cluster model을 사용
- 이는 **중앙 집중식 관리 방식**으로, client는 자신의 cluster_id를 명시적으로 알 필요가 없음

### 1.3 정보 흐름 요약

| 단계 | 위치 | 정보 | 보존 여부 |
|------|------|------|-----------|
| Client → UAV | `Packet(c.id, c.uav_id, time.time(), sent)` | client_id, uav_id, model bytes | ✅ |
| UAV → Satellite | `Packet` 객체 그대로 전달 | client_id, model bytes | ✅ |
| Satellite Buffer | `(client_id, state_dict)` 튜플 | client_id, state_dict | ✅ |
| Clustering | `buffer_client_ids`, `cluster_labels` | client_id → cluster_id 매핑 | ✅ |
| Cluster 할당 | `client_to_cluster[client_id] = cluster_id` | client_id → cluster_id 저장 | ✅ |
| 다음 라운드 | `client_to_cluster[c.id]` 조회 | cluster_id 조회 | ✅ |

**중요**: Satellite에서 client로 cluster_id를 직접 전달하는 메커니즘이 **현재 구현에는 없음**. 대신, 다음 라운드에서 client가 학습할 때 중앙에서 관리되는 `client_to_cluster` 딕셔너리를 조회하여 자신의 cluster_id를 알 수 있음.

---

## 2. 동적 Clustering 프로세스

### 2.1 Clustering이 동적으로 변하는 이유

1. **매 라운드마다 재계산**: 각 라운드마다 새로운 model updates를 받아서 clustering을 재계산
2. **Model similarity 변화**: 학습이 진행되면서 model들이 변화하므로 similarity도 변화
3. **Client 참여 변화**: 매 라운드마다 다른 client들이 참여 (DCS 또는 random selection)
4. **Cluster 수 자동 결정**: `num_clusters = max(2, min(5, len(buffer_state_dicts) // 3))`

### 2.2 Clustering 업데이트 메커니즘

#### 매 라운드 Clustering 프로세스

```python
# 1. 현재 라운드의 model updates 수집
buffer_state_dicts = [sd for _, sd in sat.buffer]
buffer_client_ids = [cid for cid, _ in sat.buffer]

# 2. Model similarity 계산
similarity_matrix = compute_similarity_matrix(buffer_state_dicts)

# 3. Clustering 수행 (매 라운드마다 새로 계산)
num_clusters = max(2, min(5, len(buffer_state_dicts) // 3))
cluster_labels = cluster_models(buffer_state_dicts, num_clusters=num_clusters)

# 4. client_to_cluster 매핑 업데이트 (덮어쓰기)
for idx, client_id in enumerate(buffer_client_ids):
    if client_id and idx < len(cluster_labels):
        client_to_cluster[client_id] = cluster_labels[idx]  # 기존 값 덮어쓰기
```

**핵심 포인트**:
- `client_to_cluster`는 **덮어쓰기 방식**으로 업데이트됨
- 이전 라운드의 cluster 할당은 현재 라운드의 할당으로 **완전히 대체**됨
- 하지만 **cluster model은 누적 업데이트**됨 (이전 round의 cluster model과 weighted average)

#### Cluster Model 업데이트 (누적)

```python
# Cluster model은 이전 round의 정보를 유지하면서 업데이트
for cluster_id, cluster_model_state in new_cluster_models.items():
    if cluster_id in cluster_models_dict:
        # 기존 cluster model과 새 cluster model을 weighted average
        cluster_models_dict[cluster_id] = fedavg(
            [cluster_models_dict[cluster_id], cluster_model_state],
            weights=[1.0, float(current_cluster_size)]
        )
    else:
        # 새로운 cluster인 경우
        cluster_models_dict[cluster_id] = cluster_model_state
```

**핵심 포인트**:
- Cluster model은 **누적 업데이트**됨 (이전 round의 정보 유지)
- Cluster 할당은 **매 라운드 재계산**되지만, cluster model은 **이전 정보를 유지**

### 2.3 Cluster 변화 추적

**현재 구현**:
- `client_to_cluster` 딕셔너리가 매 라운드마다 업데이트됨
- 하지만 **라운드별 cluster 변화를 기록하는 메커니즘이 없음**

**개선 필요**:
- 라운드별 `client_to_cluster` snapshot 저장
- Cluster 변화 시각화를 위한 데이터 수집

---

## 3. Cost 산정 식

### 3.1 Communication Cost (Memory)

#### Round별 Communication Cost

**위치**: `src/core/network.py` - `CostMeter.end_round()`

```python
def end_round(self):
    """End round and finalize costs"""
    round_bytes = sum(d["bytes"] for d in self._round.values())  # 모든 link의 bytes 합산
    self.total_cum_bytes += round_bytes
    
    round_max_delay = sum(d["max_delay"] for d in self._round.values())
    self.total_cum_time += round_max_delay
    
    self.reset_all()
    return round_bytes, round_max_delay
```

**계산식**:
```
Round Communication Cost (bytes) = Σ(link_bytes) for all links
Round Communication Cost (KB) = Round Communication Cost (bytes) / 1024
```

**각 Link별 Cost**:
```python
# Link.transmit()에서 측정
if ok:  # 전송 성공 시
    self._round[link_name]["bytes"] += nbytes  # 전송된 bytes 누적
```

**구성 요소**:
- Client → UAV: 각 client가 전송한 model state_dict bytes
- UAV → Satellite: UAV가 batch로 전송한 총 bytes

#### Cumulative Communication Cost

```
Cumulative Communication Cost (bytes) = Σ(Round Communication Cost) for all rounds
Cumulative Communication Cost (MB) = Cumulative Communication Cost (bytes) / (1024 * 1024)
```

### 3.2 Time Cost (Simulated Time)

#### Round별 Time Cost

**계산식**:
```
Round Time Cost (seconds) = Σ(max_delay) for all links
```

**각 Link별 Delay**:
```python
# Link.transmit()에서 계산
delay = (latency_ms + random.randint(0, jitter_ms)) / 1000 + len(packet) / max(1, bandwidth_bps)
```

**구성 요소**:
- `latency_ms`: 고정 지연 시간 (ms)
- `jitter_ms`: 랜덤 지터 (ms)
- `len(packet) / bandwidth_bps`: 전송 시간 (packet size / bandwidth)

**Link별 max_delay**:
```python
self._round[link_name]["max_delay"] = max(
    self._round[link_name]["max_delay"], delay_s
)  # 각 link의 최대 delay만 저장
```

#### Cumulative Time Cost

```
Cumulative Time Cost (seconds) = Σ(Round Time Cost) for all rounds
```

### 3.3 통합 Cost 식 (제안)

**현재 구현**: Memory와 Time이 별도로 측정됨

**제안하는 통합 Cost 식**:
```
Total Cost = α × (Communication Cost) + β × (Time Cost)

where:
  Communication Cost = Cumulative Communication Cost (MB)
  Time Cost = Cumulative Time Cost (seconds)
  α, β = 가중치 (예: α = 1.0, β = 0.1)
```

**정규화된 통합 Cost**:
```
Normalized Total Cost = α × (Comm Cost / Max Comm Cost) + β × (Time Cost / Max Time Cost)
```

---

## 4. Personalized Accuracy 도출 방식

### 4.1 현재 구현

**위치**: `src/experiments/abl1.py`, `bl3.py` - Personalized evaluation 섹션

#### BL3 구현 (정확한 구현)

```python
# 각 client의 local test set에서 cluster model로 평가
if cluster_models_dict and client_to_cluster:
    for client_idx, client in enumerate(clients):
        if client.id in client_to_cluster:
            cluster_id = client_to_cluster[client.id]
            if cluster_id in cluster_models_dict:
                cluster_model = SimpleCNN(num_classes).to(Config.DEVICE)
                cluster_model.load_state_dict(cluster_models_dict[cluster_id])
                cluster_model.eval()
                
                test_loader = client_test_loaders[client_idx]  # Client의 local test set
                
                # Cluster model로 client의 local test set 평가
                correct_local = 0
                total_local = 0
                with torch.no_grad():
                    for x, y in test_loader:
                        x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
                        outputs = cluster_model(x)
                        pred = outputs.argmax(1)
                        correct_local += (pred == y).sum().item()
                        total_local += y.size(0)
                
                if total_local > 0:
                    pa = correct_local / total_local * 100.0
                    personalized_accs.append(pa)

# Personalized Accuracy 평균
pa_avg = sum(personalized_accs) / len(personalized_accs) if personalized_accs else 0.0
```

**계산식**:
```
Personalized Accuracy (client_i) = (Correct predictions on client_i's local test set) / (Total samples in client_i's local test set) × 100

Personalized Accuracy (average) = (1/N) × Σ(Personalized Accuracy (client_i)) for all clients with cluster assignment
```

**대상 데이터**:
- 각 client의 **local test set** (client별로 분리된 test 데이터)
- Cluster model로 평가 (client가 속한 cluster의 model)

**평균 계산**:
- **모든 client에 대해** personalized accuracy를 계산하고 평균
- 단, `client_to_cluster`에 할당된 client만 포함

#### ABL-1 구현 (현재 라운드 참여 client만)

```python
# 현재 라운드에 참여한 client만 평가
current_round_client_ids = set(buffer_client_ids)
if cluster_models_dict and client_to_cluster:
    for client_idx, client in enumerate(clients):
        if client.id in client_to_cluster and client.id in current_round_client_ids:
            # 현재 라운드에 참여한 client만 평가
            ...
```

**차이점**:
- BL3: 모든 client에 대해 평가 (cluster 할당이 있는 경우)
- ABL-1: 현재 라운드에 참여한 client만 평가

### 4.2 Cluster별 Accuracy (제안)

**현재 구현에는 없음**. 다음 방식으로 계산 가능:

```python
# Cluster별 accuracy 계산
cluster_accs = {}
for cluster_id in cluster_models_dict.keys():
    cluster_clients = [cid for cid, cid_cluster in client_to_cluster.items() 
                      if cid_cluster == cluster_id]
    cluster_acc_list = []
    for client_id in cluster_clients:
        if client_id in personalized_acc_dict:  # 각 client의 personalized acc
            cluster_acc_list.append(personalized_acc_dict[client_id])
    if cluster_acc_list:
        cluster_accs[cluster_id] = sum(cluster_acc_list) / len(cluster_acc_list)
```

**계산식**:
```
Cluster Accuracy (cluster_k) = (1/|C_k|) × Σ(Personalized Accuracy (client_i)) 
                                for all client_i in cluster_k

where C_k = set of clients in cluster k
```

---

## 5. Model Parameter 정보

### 5.1 Model Architecture

**위치**: `src/core/model.py`

```python
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=62):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
```

**Parameter 수**:
- Conv2d(1→32): 1 × 32 × 3 × 3 = 288
- Conv2d(32→64): 32 × 64 × 3 × 3 = 18,432
- Linear(3136→128): 3136 × 128 = 401,408
- Linear(128→62): 128 × 62 = 7,936
- **Total**: 약 428,000 parameters

**State_dict 크기**:
- 각 parameter는 float32 (4 bytes)
- **Total size**: 약 428,000 × 4 = 1.7 MB per model

### 5.2 Communication Cost에 미치는 영향

**Round별 Cost**:
```
Round Communication Cost = (Number of selected clients) × (Model size) × 2
                          = N × 1.7 MB × 2
                          = 3.4N MB

where:
  N = number of selected clients
  × 2 = upload (client → satellite) + download (satellite → client, if implemented)
```

**현재 구현**:
- Upload만 측정됨 (client → satellite)
- Download는 측정되지 않음 (satellite → client 모델 전달이 시뮬레이션에 없음)

---

## 6. 요약 및 개선 사항

### 6.1 현재 구현의 명확한 부분

✅ **Client → UAV → Satellite 정보 흐름**: Packet 객체를 통해 client_id가 보존됨
✅ **Clustering 프로세스**: Model similarity 기반 clustering이 명확히 구현됨
✅ **Cost 측정**: Communication cost와 time cost가 정확히 측정됨
✅ **Personalized Accuracy**: Client별 local test set에서 cluster model로 평가

### 6.2 개선이 필요한 부분

❌ **Satellite → Client 모델 전달**: 현재 시뮬레이션에는 없음 (다음 라운드에서 client가 cluster_id를 조회)
❌ **Cluster 변화 추적**: 라운드별 cluster 변화를 기록하는 메커니즘이 없음
❌ **Cluster별 Accuracy**: 계산되지 않음
❌ **Download Cost**: Satellite → Client 모델 전송 cost가 측정되지 않음
❌ **통합 Cost 식**: Memory + Time을 합친 통합 cost 식이 없음

### 6.3 제안하는 개선 사항

1. **Cluster 변화 추적**: 라운드별 `client_to_cluster` snapshot 저장
2. **Cluster별 Accuracy 계산**: Cluster별 accuracy 계산 및 시각화
3. **통합 Cost 식**: Memory + Time을 합친 통합 cost 식 제안
4. **시각화 개선**: Cluster 변화, Cluster별 accuracy 시각화

