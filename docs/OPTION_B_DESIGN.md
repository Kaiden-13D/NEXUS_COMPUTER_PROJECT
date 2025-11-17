# UAV Local Aggregation (Option B) Design

## 1. 목표

- **3계층(Hierarchical) 구조 유지**: Client → UAV → Satellite
- **UAV 역할 확장**: 클라이언트 업데이트를 수집해 `fedavg`로 cluster별 부분 모델 생성 후 위성으로 업로드
- **Satellite 역할 집중**: UAV에서 올라온 cluster별 모델을 통합하고, cluster assignment를 재계산하여 UAV에 전달
- **클러스터 수 고정 (예: 3개)**: 초기 라운드에서 위성이 자동 결정한 후, 이후 라운드에서는 클러스터 수는 고정하고 label만 갱신
- **평가 지표 통일**: 모든 실험에서 100개 클라이언트의 테스트 정확도를 평균(Per-Client Test Accuracy)으로 비교
- **네트워크 비용 측정 유지**: Communication Cost, Simulated Time 측정 방식은 기존과 동일하게 유지

## 2. 컴포넌트 역할

| 계층 | 주요 역할 | 추가 요구사항 |
|------|-----------|----------------|
| Client | 로컬 학습, UAV로 모델 업데이트 업로드 | 자신이 속한 cluster ID를 저장/조회, UAV에서 내려준 cluster model로 학습 |
| UAV | 클라이언트 선택(DCS), cluster별 부분집계(`fedavg`), Satellite에 cluster별 모델 업로드 | 클라이언트↔클러스터 매핑 관리, 위성에서 받은 cluster assignment 갱신 |
| Satellite | 모든 UAV로부터 cluster별 모델 수신 후 전역 cluster 모델 갱신, cluster assignment 재계산 | UAV별/클러스터별 메타데이터 수신 및 분배, 클러스터 수 고정 관리 |

## 3. 라운드 흐름 (클러스터 수 = K)

1. **라운드 0 초기화**
   1. 각 클라이언트는 전역 모델로 학습 후 UAV에 업로드
   2. UAV는 클라이언트 업데이트를 `fedavg`해 `M_uav` 단일 모델을 Satellite로 전송
   3. Satellite는 `M_uav`들의 feature similarity로 클러스터 수 `K`를 자동 결정 후 고정하고, cluster assignment(클라이언트/클러스터 매핑)를 계산해 모든 UAV에 전달

2. **라운드 ≥ 1**
   1. **클라이언트 단계**: UAV에서 받은 cluster model로 로컬 학습, 업데이트 전송 시 자신의 cluster ID를 포함
   2. **UAV 단계**:
      - DCS(선택 적용)로 참여 클라이언트 결정
      - cluster ID별 버킷에 업데이트를 저장
      - cluster별 `fedavg` → `M_uav_cluster0 ... M_uav_clusterK-1`
      - 각 cluster 모델과 메타데이터(데이터 사이즈, 참여 클라이언트 리스트 등)를 Satellite에 업로드
   3. **Satellite 단계**:
      - 모든 UAV에서 올라온 cluster별 모델을 cluster ID별로 모아 전역 cluster 모델 생성
      - 필요 시 상위 수준에서 클러스터링 재검토 (클러스터 수는 고정)
      - 최신 cluster assignment(클라이언트 ID ↔ cluster ID)를 각 UAV에 전달
   4. **배포**: Satellite → UAV → 클라이언트 순으로 cluster model/assignment를 전달

3. **Cluster Assignment 관리**
   - Satellite는 클라이언트 ID 단위의 cluster assignment를 계산하여 UAV별로 전달
   - UAV는 자체 매핑 테이블(`client_to_cluster`)을 업데이트하고, 클라이언트에게 cluster 정보를 재배포
   - 클라이언트는 cluster ID를 로컬에 저장하여 다음 라운드에서 해당 cluster 모델로 학습

## 4. 데이터 구조 설계

### 4.1 UAV 측

```python
class UAV:
    client_to_cluster: Dict[str, int]
    cluster_buffers: Dict[int, List[Tuple[str, StateDict, float]]]

    def aggregate_cluster(self, cluster_id: int) -> Dict:
        state_dicts = [sd for _, sd, _ in self.cluster_buffers[cluster_id]]
        weights = [w for _, _, w in self.cluster_buffers[cluster_id]]
        return fedavg(state_dicts, weights)
```

- `cluster_buffers`는 라운드 단위로 초기화
- Satellite로 올릴 payload 예시:

```json
{
  "uav_id": "uav_0",
  "round": 5,
  "clusters": [
    {"cluster_id": 0, "state_dict": "...", "data_weight": 1234, "clients": ["client_1", "..."]},
    {"cluster_id": 1, "..."},
    {"cluster_id": 2, "..."}
  ]
}
```

### 4.2 Satellite 측

```python
class SatelliteAggregator:
    cluster_models: Dict[int, StateDict]
    client_to_cluster: Dict[str, int]

    def update_clusters(self, uav_payloads):
        for cluster_id in range(K):
            sd_list = [payload.cluster_models[cluster_id] for payload in uav_payloads]
            weight_list = [payload.cluster_weights[cluster_id] for payload in uav_payloads]
            self.cluster_models[cluster_id] = fedavg(sd_list, weight_list)
```

- 클러스터 재할당 시 기존 클러스터 수(K)는 고정. 필요 시 유사도 기반으로 label만 조정.

## 5. 평가 지표 및 로깅

1. **Per-Client Test Accuracy**
   - 모든 실험에서 100개 클라이언트 각각의 로컬 테스트셋 정확도를 측정하고 평균
   - 로그 포맷 예:
     - `[Perf] Avg Test Acc: 92.34% (100 clients)`
   - 필요 시 분산/표준편차를 추가 기록해 클러스터 효과 분석

2. **네트워크 비용**
   - 기존 `CostMeter` 유지:
     - Round Communication Cost (KB)
     - Round Simulated Time (sec)
     - Cumulative Cost/Time
   - UAV에서 cluster별 모델을 올리므로 payload 크기가 늘어날 수 있음 → 로깅 시 cluster별 payload 크기 합을 측정

## 6. 구현 계획

1. **데이터 구조 업데이트**
   - `src/core/fl_nodes.py`의 `UAV` 클래스에 cluster buffer 및 aggregation 로직 추가
   - `SatAgg`는 클러스터별 payload를 파싱하도록 수정

2. **실험 스크립트 리팩토링**
   - `bl1`, `bl2`: UAV가 cluster 수 1개로 동작 (기존 FedAvg와 동일) → 최소 변경
   - `bl3`, `abl1`: Satellite에서 최신 cluster assignment 관리, UAV는 cluster별 aggregation 수행

3. **클러스터 수 관리**
   - 라운드 0에서 Satellite가 자동 결정한 `K`를 Config나 상태 파일에 저장
   - 이후 라운드에서는 `K` 고정, label만 업데이트

4. **클러스터 정보 배포**
   - Satellite → UAV: `{client_id: cluster_id}` mapping 전달 메커니즘 마련 (예: 라운드별 메시지)
   - UAV → Client: 학습 시작 전 cluster 모델과 함께 cluster ID 전달

5. **평가/로그 업데이트**
   - Per-Client Test Accuracy 계산으로 통일
   - Cluster별 모델 통계, UAV payload 사이즈 등 추가 로깅

6. **마이그레이션 메모**
   - 기존 결과/스크립트는 `jaebaek` 브랜치에 보존
   - 새로운 실험은 `feature/uav-local-cluster` 브랜치에서 수행

