# 로그 출력 예시

각 실험별로 예상되는 로그 출력 예시입니다.

## BL1 (Hierarchical FedAvg)

```
=== Starting BL1 Simulation (Goal: Avg Test Acc 90.0%) ===

=== Round 1 ===
  Aggregated 6 UAV updates.
  [Perf] Avg Test Acc: 35.42% | Avg Loss: 2.1234 (100 clients)
  [Effi] Round Cost: 1024 KB, +1.25s simulated time
  [Cumul] Total Data: 1.00 MB | Total Time: 1.25s

=== Round 2 ===
  Aggregated 6 UAV updates.
  [Perf] Avg Test Acc: 42.18% | Avg Loss: 1.9876 (100 clients)
  [Effi] Round Cost: 1024 KB, +1.23s simulated time
  [Cumul] Total Data: 2.00 MB | Total Time: 2.48s

...

=== Round 15 ===
  Aggregated 6 UAV updates.
  [Perf] Avg Test Acc: 90.15% | Avg Loss: 0.3456 (100 clients)
  [Effi] Round Cost: 1024 KB, +1.20s simulated time
  [Cumul] Total Data: 15.36 MB | Total Time: 18.50s

!!! Target Accuracy (90.0%) Reached at Round 15 !!!
FINAL Metrics -> Avg Test Acc: 90.15% | Avg Loss: 0.3456 | Comm Cost: 15.36 MB | Time Cost: 18.50s

Results saved to: results/bl1/bl1_summary_20241112_120000.txt
```

## BL2 (DCS Only)

```
=== Starting BL2 (DCS Only) Simulation (Goal: Avg Test Acc 90.0%) ===

=== Round 1 ===
  Aggregated 6 UAV updates from DCS-selected clients.
  [Perf] Avg Test Acc: 36.78% | Avg Loss: 2.0456 (100 clients)
  [Effi] Round Cost: 1024 KB, +1.15s simulated time
  [Cumul] Total Data: 1.00 MB | Total Time: 1.15s

=== Round 2 ===
  Aggregated 6 UAV updates from DCS-selected clients.
  [Perf] Avg Test Acc: 43.92% | Avg Loss: 1.9234 (100 clients)
  [Effi] Round Cost: 1024 KB, +1.12s simulated time
  [Cumul] Total Data: 2.00 MB | Total Time: 2.27s

...

=== Round 14 ===
  Aggregated 6 UAV updates from DCS-selected clients.
  [Perf] Avg Test Acc: 90.23% | Avg Loss: 0.3234 (100 clients)
  [Effi] Round Cost: 1024 KB, +1.10s simulated time
  [Cumul] Total Data: 14.34 MB | Total Time: 16.80s

!!! Target Accuracy (90.0%) Reached at Round 14 !!!
FINAL Metrics -> Avg Test Acc: 90.23% | Avg Loss: 0.3234 | Comm Cost: 14.34 MB | Time Cost: 16.80s

Results saved to: results/bl2/bl2_summary_20241112_120000.txt
```

## BL3 (Clustering Only)

```
=== Starting BL3 (Clustering Only) Simulation (Goal: Avg Test Acc 100.0%) ===

=== Round 1 ===
  Received 6 UAV updates.
  Round 0: Determining number of clusters...
  Determined 3 clusters from 30 client models.
  Updated 3 global cluster models.
  [Perf] Avg Test Acc: 38.45% | Avg Loss: 1.9876 (100 clients)
  [Effi] Round Cost: 1536 KB, +1.35s simulated time
  [Cumul] Total Data: 1.50 MB | Total Time: 1.35s

=== Round 2 ===
  Received 6 UAV updates.
  Updated 3 global cluster models.
  [Perf] Avg Test Acc: 52.34% | Avg Loss: 1.6543 (100 clients)
  [Effi] Round Cost: 1536 KB, +1.32s simulated time
  [Cumul] Total Data: 3.00 MB | Total Time: 2.67s

=== Round 3 ===
  Received 6 UAV updates.
  Updated 3 global cluster models.
  [Perf] Avg Test Acc: 65.78% | Avg Loss: 1.2345 (100 clients)
  [Effi] Round Cost: 1536 KB, +1.30s simulated time
  [Cumul] Total Data: 4.50 MB | Total Time: 3.97s

...

=== Round 25 ===
  Received 6 UAV updates.
  Updated 3 global cluster models.
  [Perf] Avg Test Acc: 100.12% | Avg Loss: 0.1234 (100 clients)
  [Effi] Round Cost: 1536 KB, +1.28s simulated time
  [Cumul] Total Data: 37.50 MB | Total Time: 32.00s

!!! Target Accuracy (100.0%) Reached at Round 25 !!!
FINAL Metrics -> Avg Test Acc: 100.12% | Avg Loss: 0.1234 | Comm Cost: 37.50 MB | Time Cost: 32.00s

Results saved to: results/bl3/bl3_summary_20241112_120000.txt
```

## ABL1 (DCS + Clustering)

```
=== Starting ABL-1 (DCS + Clustering) Simulation (Goal: Avg Test Acc 100.0%) ===

=== Round 1 ===
  Received 6 UAV updates.
  Round 0: Determining number of clusters...
  Determined 3 clusters from 30 client models.
  Updated 3 global cluster models.
  [Perf] Avg Test Acc: 39.12% | Avg Loss: 1.9456 (100 clients)
  [Effi] Round Cost: 1536 KB, +1.28s simulated time
  [Cumul] Total Data: 1.50 MB | Total Time: 1.28s

=== Round 2 ===
  Received 6 UAV updates.
  Updated 3 global cluster models.
  [Perf] Avg Test Acc: 54.67% | Avg Loss: 1.6123 (100 clients)
  [Effi] Round Cost: 1536 KB, +1.25s simulated time
  [Cumul] Total Data: 3.00 MB | Total Time: 2.53s

=== Round 3 ===
  Received 6 UAV updates.
  Updated 3 global cluster models.
  [Perf] Avg Test Acc: 68.45% | Avg Loss: 1.1987 (100 clients)
  [Effi] Round Cost: 1536 KB, +1.22s simulated time
  [Cumul] Total Data: 4.50 MB | Total Time: 3.75s

...

=== Round 22 ===
  Received 6 UAV updates.
  Updated 3 global cluster models.
  [Perf] Avg Test Acc: 100.08% | Avg Loss: 0.1123 (100 clients)
  [Effi] Round Cost: 1536 KB, +1.20s simulated time
  [Cumul] Total Data: 33.00 MB | Total Time: 26.40s

!!! Target Accuracy (100.0%) Reached at Round 22 !!!
FINAL Metrics -> Avg Test Acc: 100.08% | Avg Loss: 0.1123 | Comm Cost: 33.00 MB | Time Cost: 26.40s

Results saved to: results/abl1/abl1_summary_20241112_120000.txt
```

## 로그 포맷 설명

### 라운드별 출력 구조

```
=== Round N ===
  [선택적] Received X UAV updates.
  [선택적] Round 0: Determining number of clusters...
  [선택적] Determined K clusters from M client models.
  [선택적] Updated K global cluster models.
  [Perf] Avg Test Acc: XX.XX% | Avg Loss: X.XXXX (N clients)
  [Effi] Round Cost: XXX KB, +X.XXs simulated time
  [Cumul] Total Data: XX.XX MB | Total Time: XX.XXs
```

### 필드 설명

- **Received X UAV updates**: Satellite가 받은 UAV 업데이트 수 (BL3, ABL1)
- **Round 0: Determining number of clusters...**: 라운드 0에서 클러스터 수 결정 (BL3, ABL1)
- **Determined K clusters**: 결정된 클러스터 수 (BL3, ABL1)
- **Updated K global cluster models**: 업데이트된 전역 cluster 모델 수 (BL3, ABL1)
- **Avg Test Acc**: 100개 클라이언트의 테스트 정확도 평균
- **Avg Loss**: 100개 클라이언트의 loss 평균
- **Round Cost**: 라운드별 통신 비용 (KB)
- **simulated time**: 라운드별 시뮬레이션 시간 (초)
- **Total Data**: 누적 통신 비용 (MB)
- **Total Time**: 누적 시뮬레이션 시간 (초)

### 차이점

- **BL1/BL2**: Cluster 정보 없음, 단순 집계 메시지
- **BL3/ABL1**: 
  - 라운드 0: 클러스터 수 결정 메시지
  - 모든 라운드: "Updated K global cluster models" 메시지
  - 더 높은 통신 비용 (cluster별 모델 전송)

### Progress Log (logs/progress/latest.log)

```
=== Starting BL3 Experiment ===
Loading FEMNIST dataset...
Data loading complete
=== Round 1 ===
Round 1: Received 6 UAV updates
Round 1: Determining number of clusters...
Round 1: Evaluating models...
Round 1: Avg Test Acc = 38.45%, Avg Loss = 1.9876
=== Round 2 ===
Round 2: Received 6 UAV updates
Round 2: Evaluating models...
Round 2: Avg Test Acc = 52.34%, Avg Loss = 1.6543
...
```

