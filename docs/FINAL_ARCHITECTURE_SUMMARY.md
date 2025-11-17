# Final Architecture and Design Summary

## 개요

이 문서는 Hierarchical Cluster-based Personalized Federated Learning (HPFL) 시스템의 최종 아키텍처와 설계 결정사항을 종합적으로 정리한 것입니다. Option B 아키텍처로의 리팩토링 과정에서 논의된 모든 주요 사항들을 포함합니다.

---

## 1. 아키텍처 설계 결정

### 1.1 Option B 아키텍처 채택

**결정 배경:**
- 기존 아키텍처에서는 Satellite가 모든 클라이언트의 개별 모델을 받아 처리하는 구조였음
- 이는 통신 비용이 높고 확장성이 떨어지는 문제가 있었음
- UAV가 단순히 중계 역할만 하는 것이 아니라 지역 집계를 수행하도록 변경

**핵심 설계 원칙:**
- **3계층 구조 유지**: Client → UAV → Satellite
- **UAV 역할 확장**: 클라이언트 업데이트를 클러스터별로 수집하여 FedAvg 수행
- **Satellite 역할 집중**: 전역 클러스터 모델 관리 및 클러스터 할당 결정
- **계층적 집계**: UAV에서 지역 집계 → Satellite에서 전역 집계

### 1.2 클러스터 수 관리 전략

**초기 결정 (라운드 0):**
- Satellite가 모든 UAV의 집계된 모델을 받아서 클러스터 수를 자동 결정
- 클러스터 수는 `NUM_CLUSTERS` 설정값을 사용하거나 `AUTO_DETERMINE_CLUSTERS=True`일 경우 자동 결정
- 결정된 클러스터 수는 이후 모든 라운드에서 고정

**이후 라운드:**
- 클러스터 수는 변경하지 않음 (고정)
- 클러스터 할당(assignment)만 업데이트
- 이는 클러스터가 너무 자주 변하면 학습이 불안정해지는 문제를 방지하기 위함

### 1.3 클라이언트-클러스터 매핑 관리

**문제 상황:**
- Satellite는 개별 클라이언트 정보를 직접 알 수 없음
- UAV가 클러스터별로 집계한 모델만 받음
- 하지만 클러스터 할당을 업데이트하려면 클라이언트 정보가 필요함

**해결 방법:**
- **Round 0**: Satellite가 개별 클라이언트 모델을 받아서 초기 클러스터 할당 결정
- **이후 라운드**: 
  - UAV가 클러스터별 집계 시 참여한 클라이언트 ID 리스트를 메타데이터로 포함
  - Satellite는 이 정보를 바탕으로 전역 클러스터 할당을 재계산
  - Satellite → UAV → Client 순으로 클러스터 할당 정보 전달
  - 각 계층에서 `client_to_cluster` 매핑 테이블 유지

---

## 2. 시스템 컴포넌트 상세

### 2.1 Client (FLClient)

**주요 기능:**
- 로컬 데이터셋으로 모델 학습
- 자신이 속한 클러스터 ID 저장 (`cluster_id: Optional[int]`)
- UAV로 모델 업데이트 전송 시 클러스터 ID 포함

**학습 프로세스:**
- UAV에서 받은 클러스터별 모델로 로컬 학습 수행
- 학습 후 업데이트된 모델 파라미터를 UAV로 전송

### 2.2 UAV (Regional Aggregator)

**역할 변화:**
- 기존: 단순 중계 (Client → Satellite)
- 현재: 지역 집계 수행 + 중계

**주요 데이터 구조:**
```python
class UAV:
    client_to_cluster: Dict[str, int]  # 클라이언트-클러스터 매핑
    cluster_buffers: Dict[int, List[Tuple[str, Dict, float]]]  # 클러스터별 버퍼
    # 각 버퍼: (client_id, state_dict, data_weight)
```

**주요 메서드:**
- `update_client_cluster_mapping()`: Satellite에서 받은 클러스터 할당 업데이트
- `add_client_update()`: 클라이언트 업데이트를 해당 클러스터 버퍼에 추가
- `aggregate_cluster_models()`: 클러스터별로 FedAvg 수행
- `send_cluster_models_to_satellite()`: 집계된 클러스터별 모델을 Satellite로 전송

**데이터 전송 형식:**
- JSON + Base64 인코딩 사용
- `state_dict`는 바이트로 직렬화 후 Base64로 인코딩하여 JSON에 포함
- 메타데이터: 클러스터 ID, 데이터 가중치, 참여 클라이언트 리스트

### 2.3 Satellite (SatAgg)

**주요 역할:**
- 모든 UAV로부터 클러스터별 모델 수신
- 전역 클러스터 모델 업데이트 (FedAvg)
- 클러스터 할당 재계산 및 배포

**주요 데이터 구조:**
```python
class SatAgg:
    cluster_models: Dict[int, Dict]  # 전역 클러스터 모델
    client_to_cluster: Dict[str, int]  # 전역 클라이언트-클러스터 매핑
    num_clusters: Optional[int]  # 고정된 클러스터 수
```

**클러스터 할당 업데이트:**
- Round 0: 개별 클라이언트 모델로 초기 클러스터 할당 결정
- 이후 라운드: UAV에서 받은 메타데이터(참여 클라이언트 리스트)를 바탕으로 할당 업데이트
- 업데이트된 할당을 모든 UAV에 전달

---

## 3. 실험 설계

### 3.1 실험 구성

| 실험 | 설명 | DCS | Clustering | 클러스터 수 |
|------|------|-----|------------|-------------|
| **BL1** | Hierarchical FedAvg (Baseline) | ❌ | ❌ | 1 (단일 전역 모델) |
| **BL2** | DCS Only | ✅ | ❌ | 1 (단일 전역 모델) |
| **BL3** | Clustering Only | ❌ | ✅ | K (자동 결정, 기본 3) |
| **ABL1** | DCS + Clustering (Proposed) | ✅ | ✅ | K (자동 결정, 기본 3) |

### 3.2 BL1: Hierarchical FedAvg

**특징:**
- 가장 기본적인 계층적 연합학습
- 랜덤 클라이언트 선택 (SAMPLE_FRAC = 0.3)
- 모든 클라이언트를 클러스터 0에 할당
- UAV에서 단일 모델 집계 → Satellite에서 전역 집계

**목적:**
- 다른 실험들의 베이스라인 성능 제공
- 계층적 구조의 기본 비용 측정

### 3.3 BL2: DCS Only

**특징:**
- Dynamic Client Selection 적용
- 클러스터링 없이 단일 전역 모델 사용
- DCS 점수 기반으로 상위 30% 클라이언트 선택

**DCS 점수 계산:**
```
s_i = α·q_i + β·c_i + γ·d_i + δ·g_i
```
- `q_i`: 통신 품질
- `c_i`: 계산 능력
- `d_i`: 데이터 중요도
- `g_i`: 기여도

**목적:**
- 시스템 이질성 관리 효과 측정
- 통신 비용 절감 효과 확인

### 3.4 BL3: Clustering Only

**특징:**
- 클러스터링만 적용 (DCS 없음)
- 랜덤 클라이언트 선택
- Round 0에서 클러스터 수 자동 결정 후 고정

**라운드별 흐름:**
1. **Round 0**:
   - 각 클라이언트가 전역 모델로 학습
   - 개별 모델을 Satellite로 전송
   - Satellite가 클러스터 수 결정 및 초기 할당
   
2. **Round ≥ 1**:
   - 클라이언트는 자신의 클러스터 모델로 학습
   - UAV가 클러스터별로 집계
   - Satellite가 전역 클러스터 모델 업데이트 및 할당 재계산

**목적:**
- 데이터 이질성 관리 효과 측정
- 클러스터링만으로의 성능 향상 확인

### 3.5 ABL1: DCS + Clustering (Proposed Method)

**특징:**
- DCS와 클러스터링 모두 적용
- BL2의 클라이언트 선택 + BL3의 클러스터링 결합

**목적:**
- 시스템 이질성과 데이터 이질성을 동시에 관리
- 최적의 성능과 비용 효율성 달성

---

## 4. 평가 메트릭 통일

### 4.1 Per-Client Test Accuracy

**변경 배경:**
- 기존에는 "Global Accuracy"와 "Personalized Accuracy"를 별도로 측정
- 이는 비교가 복잡하고 일관성이 떨어짐
- 모든 실험에서 동일한 기준으로 비교하기 위해 통일

**측정 방법:**
- 100개 클라이언트 각각의 로컬 테스트셋에서 정확도 측정
- 모든 클라이언트의 정확도를 평균
- 로그 포맷: `[Perf] Avg Test Acc: 92.34% | Avg Loss: 0.25`

**장점:**
- 모든 실험에서 동일한 평가 기준
- 클러스터별 특화 모델의 효과를 직접 비교 가능
- 해석이 간단하고 명확함

### 4.2 네트워크 비용 측정

**유지된 메트릭:**
- **Round Communication Cost**: 각 라운드에서 전송된 데이터 크기 (KB)
- **Round Simulated Time**: 각 라운드의 시뮬레이션 시간 (초)
- **Cumulative Cost/Time**: 누적 통신 비용 및 시간

**측정 방식:**
- `CostMeter` 클래스를 통해 모든 패킷 전송 추적
- 클라이언트 → UAV, UAV → Satellite 모든 링크 측정
- 클러스터별 모델 전송으로 인한 추가 비용도 포함

---

## 5. 시각화 개선

### 5.1 2x2 그리드 레이아웃

**변경 사항:**
- 기존: 여러 개의 차트를 세로로 나열
- 현재: 2x2 그리드로 4개 주요 메트릭만 표시

**표시되는 메트릭:**
1. **(a) Average Test Accuracy**: 라운드별 평균 테스트 정확도
2. **(b) Average Loss**: 라운드별 평균 손실
3. **(c) Cumulative Communication Cost**: 누적 통신 비용
4. **(d) Cumulative Time**: 누적 시간

### 5.2 스타일 개선

**라인 스타일 및 마커:**
- **BL1**: solid black line with `+` marker
- **BL2**: dashed red line with `*` marker
- **BL3**: dotted blue line with `o` marker
- **ABL1**: dash-dot green line with `x` marker

**시각적 개선:**
- 그리드 라인: dashed 스타일로 가독성 향상
- 범례: framealpha 적용으로 배경과 구분
- 마커: `markevery` 옵션으로 과도한 마커 방지

---

## 6. 주요 논의 사항 및 결정

### 6.1 UAV 집계 모델

**질문:** UAV에서 어떤 모델로 집계를 수행하는가?

**답변:**
- BL1, BL2: 클러스터 0 (단일 전역 모델)로만 집계
- BL3, ABL1: 클러스터별로 분리하여 집계
- 각 클러스터마다 별도의 FedAvg 수행

### 6.2 클러스터 수 동적 변경

**질문:** 클러스터 수를 라운드마다 동적으로 변경할 수 있는가?

**답변:**
- 초기에는 동적 변경을 고려했으나, 학습 안정성을 위해 고정
- Round 0에서 결정한 클러스터 수를 이후 모든 라운드에서 유지
- 클러스터 할당(assignment)만 업데이트

### 6.3 DCS 적용 범위

**질문:** DCS는 클러스터별로 적용하는가, UAV별로 적용하는가?

**답변:**
- **UAV별로 적용**: 각 UAV가 자신이 담당하는 클라이언트들 중에서 상위 30% 선택
- 클러스터별로 따로 선택하지 않음
- 선택된 클라이언트들이 속한 클러스터에 따라 버퍼에 분류

### 6.4 클라이언트 선택 비율

**질문:** 10%만 선택하면 너무 적지 않은가? (100명 중 10명 → 6개 UAV → 각 UAV당 약 1-2명)

**답변:**
- `SAMPLE_FRAC`를 0.1에서 0.3으로 증가
- 이제 각 라운드에서 약 30명의 클라이언트가 선택됨
- 각 UAV당 평균 5명 정도의 클라이언트가 참여

### 6.5 평가 시점

**질문:** Accuracy는 어디서 계산되는가? FedAvg는 파라미터 집계인데?

**답변:**
- FedAvg는 모델 파라미터(state_dict)를 집계하는 것
- Accuracy는 집계된 모델을 각 클라이언트의 테스트셋에서 평가
- 매 라운드마다 평가 수행
- 클라이언트 레벨에서 평가 후 평균

---

## 7. 구현 세부사항

### 7.1 데이터 직렬화

**문제:**
- PyTorch `state_dict`는 바이트 데이터
- JSON으로 직접 직렬화 불가

**해결:**
- `state_dict`를 바이트로 직렬화 (`torch.save` → `io.BytesIO`)
- Base64로 인코딩하여 문자열로 변환
- JSON에 문자열로 포함하여 전송
- 수신 측에서 Base64 디코딩 후 `torch.load`로 복원

### 7.2 클러스터 할당 전파

**전파 경로:**
1. Satellite가 전역 `client_to_cluster` 매핑 계산
2. Satellite → UAV: 각 UAV가 담당하는 클라이언트들의 할당 정보 전달
3. UAV가 로컬 `client_to_cluster` 업데이트
4. UAV → Client: 각 클라이언트에게 자신의 클러스터 ID 전달
5. Client가 `cluster_id` 필드 업데이트

**구현:**
- `Packet` 타입에 `CLUSTER_ASSIGNMENT` 추가
- 각 계층에서 매핑 테이블 유지 및 업데이트

### 7.3 라운드 버퍼 관리

**UAV 클러스터 버퍼:**
- 각 라운드 시작 시 `reset_round_buffer()`로 초기화
- 클라이언트 업데이트 수신 시 해당 클러스터 버퍼에 추가
- 라운드 종료 시 모든 클러스터에 대해 집계 수행

**Satellite 버퍼:**
- UAV로부터 클러스터별 모델 수신
- 모든 UAV의 응답을 기다린 후 전역 집계 수행

---

## 8. 설정 파라미터

### 8.1 주요 설정값

```python
# 클라이언트 선택
SAMPLE_FRAC = 0.3  # 30% 클라이언트 선택

# 클러스터링
NUM_CLUSTERS = 3  # 기본 클러스터 수
AUTO_DETERMINE_CLUSTERS = True  # 자동 결정 여부

# 학습 파라미터
BATCH_SIZE = 32
LEARNING_RATE = 0.01
LOCAL_EPOCHS = 1
TOTAL_ROUNDS = 50

# DCS 가중치
ALPHA = 0.3  # 통신 품질
BETA = 0.3   # 계산 능력
GAMMA = 0.2  # 데이터 중요도
DELTA = 0.2  # 기여도
```

### 8.2 네트워크 토폴로지

- **클라이언트 수**: 100
- **UAV 수**: 6
- **Satellite**: 1
- **클라이언트당 할당된 UAV**: 1 (고정)

---

## 9. 로그 출력 형식

### 9.1 라운드별 로그 구조

```
=== Round X ===
[Selection] Selected 30 clients (30.0%)
[Training] Clients training...
[UAV] Received updates from 30 clients
[Satellite] Received cluster models from 6 UAVs
[Clustering] Updated cluster assignments (3 clusters)
[Perf] Avg Test Acc: 92.34% | Avg Loss: 0.25
[Effi] Round Cost: 1234.56 KB, +5.67s simulated time
[Cumul] Total Data: 123.45 MB | Total Time: 123.45s
```

### 9.2 주요 로그 메시지

- `[Selection]`: 클라이언트 선택 결과
- `[Training]`: 학습 진행 상황
- `[UAV]`: UAV 집계 결과
- `[Satellite]`: Satellite 집계 결과
- `[Clustering]`: 클러스터 할당 업데이트
- `[Perf]`: 성능 메트릭
- `[Effi]`: 라운드별 효율성
- `[Cumul]`: 누적 통계

---

## 10. 향후 개선 방향

### 10.1 잠재적 개선사항

1. **동적 클러스터 수 조정**
   - 학습 안정성을 유지하면서 클러스터 수를 조정하는 방법 연구
   - 예: 일정 라운드마다 재평가

2. **클러스터별 DCS**
   - 각 클러스터마다 별도로 클라이언트 선택
   - 클러스터 특성에 맞는 클라이언트 선택

3. **비동기 집계**
   - 모든 UAV의 응답을 기다리지 않고 부분 집계 수행
   - 통신 지연 감소

4. **적응형 학습률**
   - 클러스터별로 다른 학습률 적용
   - 클러스터 특성에 맞는 최적화

### 10.2 실험 확장

1. **다양한 데이터셋**
   - CIFAR-10, CIFAR-100 등 다른 데이터셋으로 확장
   - 실제 IoT 데이터 적용

2. **네트워크 조건 변화**
   - 동적 네트워크 토폴로지
   - 링크 장애 시나리오

3. **확장성 테스트**
   - 더 많은 클라이언트 (1000+)
   - 더 많은 UAV (10+)

---

## 11. 참고 문서

- `docs/OPTION_B_DESIGN.md`: Option B 아키텍처 상세 설계
- `docs/PROCESS_DOCUMENTATION.md`: 기존 프로세스 문서화
- `docs/LOG_OUTPUT_EXAMPLES.md`: 로그 출력 예시
- `src/core/fl_nodes.py`: 핵심 노드 구현
- `src/experiments/*.py`: 각 실험 스크립트

---

## 12. 요약

이번 리팩토링의 핵심은 **UAV의 역할을 단순 중계에서 지역 집계로 확장**한 것입니다. 이를 통해:

1. **통신 효율성 향상**: 클러스터별 집계로 전송 데이터 최적화
2. **확장성 개선**: Satellite의 부하 감소
3. **성능 향상**: 클러스터별 특화 모델로 정확도 향상
4. **비용 절감**: DCS와 클러스터링의 시너지 효과

모든 실험은 **Per-Client Test Accuracy**로 통일하여 일관된 비교가 가능하며, 시각화는 2x2 그리드로 깔끔하게 정리되어 결과 분석이 용이합니다.

