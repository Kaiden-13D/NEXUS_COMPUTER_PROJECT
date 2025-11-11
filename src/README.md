# HPFL (Hierarchical Cluster-based Personalized Federated Learning)

위성-항공-지상 통합망(SAGIN)을 위한 계층적 클러스터 기반 개인화 연합학습 구현

## 디렉토리 구조

```
src/
├── core/              # 핵심 모듈
│   ├── data.py        # FEMNIST 데이터셋 로딩 및 파티셔닝
│   ├── model.py       # 신경망 모델 (SimpleCNN)
│   ├── network.py     # 네트워크 시뮬레이션 (Link, CostMeter)
│   ├── fl_nodes.py    # FL 노드 (FLClient, UAV, SatAgg)
│   └── aggregation.py # 집계 함수 (fedavg)
│
├── utils/             # 유틸리티
│   └── model_utils.py # 모델 관련 유틸리티
│
├── config/           # 설정
│   └── default_config.py  # 기본 설정
│
└── experiments/      # 실험 스크립트
    ├── bl1.py        # BL1: 기본 계층적 FedAvg
    ├── bl2.py        # BL2: DCS Only
    ├── bl3.py        # BL3: Clustering Only
    └── abl1.py       # ABL-1: HPFL (제안 모델, 예정)
```

## 실험 모델

### BL1: Hierarchical FedAvg
- **설명**: 기본 계층적 연합학습
- **특징**: 랜덤 클라이언트 선택, 단순 평균 집계
- **목적**: 베이스라인 성능 및 비용 측정

### BL2: DCS Only
- **설명**: 동적 클라이언트 선택만 적용
- **특징**: DCS를 통한 클라이언트 선택, 클러스터링 미적용
- **목적**: 시스템 이질성 관리 효과 측정 (비용 감소)

### BL3: Clustering Only
- **설명**: 모델 유사도 기반 클러스터링만 적용
- **특징**: 랜덤 클라이언트 선택, 모델 유사도 기반 클러스터링, 클러스터별 집계, DCS 미적용
- **목적**: 데이터 이질성 완화 효과 측정 (정확도 향상)

### ABL-1: HPFL (제안 모델, 예정)
- **설명**: DCS + 클러스터링 모두 적용
- **특징**: 동적 클라이언트 선택 + 모델 유사도 기반 클러스터링
- **목적**: 두 전략의 시너지 효과 검증 (비용 감소 + 정확도 향상)

## 실행 방법

```bash
# BL1 실험
python -m src.experiments.bl1

# BL2 실험
python -m src.experiments.bl2

# BL3 실험
python -m src.experiments.bl3
```

## 평가 지표

### 성능 지표
- **Global Accuracy (GA)**: 전역 테스트셋에서의 정확도
- **Personalized Accuracy (PA)**: 각 클라이언트의 로컬 테스트셋에서의 평균 정확도 (예정)

### 효율성 지표
- **Total Communication Cost**: 목표 정확도 달성까지의 총 통신 비용 (MB/GB)
- **Total Time Cost**: 목표 정확도 달성까지의 총 시간 비용 (초)

## 설정

`config/default_config.py`에서 실험 설정을 변경할 수 있습니다:

- `NUM_UAV`: UAV 수
- `NUM_CLIENTS`: 클라이언트 수
- `ROUNDS`: 최대 라운드 수
- `SAMPLE_FRAC`: 라운드당 참여 클라이언트 비율
- `TARGET_ACC`: 목표 정확도 (%)
- `CLIENT_UPLINK_BW`, `UAV_SAT_BW`: 네트워크 대역폭

