# HPFL 프로젝트 파일 설명서

이 문서는 HPFL (Hierarchical Cluster-based Personalized Federated Learning) 프로젝트의 각 파일과 디렉토리에 대한 상세한 설명을 제공합니다.

## 디렉토리 구조

```
hpfl/
├── src/                    # 소스 코드
│   ├── core/              # 핵심 모듈
│   ├── utils/              # 유틸리티
│   ├── config/             # 설정
│   └── experiments/        # 실험 스크립트
├── scripts/                # 실행 스크립트
├── logs/                   # 로그 파일
├── results/                # 실험 결과
├── back_up/                # 백업 파일
└── materials/              # 논문 자료
```

---

## Core 모듈 (`src/core/`)

### `data.py`
**역할**: FEMNIST 데이터셋 로딩 및 Writer 기반 파티셔닝

**주요 함수/클래스**:
- `FEMNISTDataset`: Hugging Face의 FEMNIST 데이터셋을 PyTorch Dataset으로 래핑
  - 이미지를 grayscale로 변환
  - 정규화 및 리사이즈 처리
  
- `setup_femnist_by_writer()`: Writer ID 기반으로 데이터를 클라이언트에게 분배
  - 각 클라이언트는 여러 Writer의 데이터를 받음 (Non-IID 환경 구성)
  - Train/Test 분할 수행
  - 진행 상황은 tqdm으로 표시

**특징**:
- Non-IID 데이터 분포 생성 (Writer 기반)
- 진행 상황 로깅 지원

---

### `model.py`
**역할**: 신경망 모델 정의

**주요 클래스**:
- `SimpleCNN`: FEMNIST용 간단한 CNN 모델
  - 구조: Conv2d(1→32) → MaxPool → Conv2d(32→64) → MaxPool → FC(128) → FC(62)
  - 62개 클래스 분류 (알파벳 대소문자 + 숫자)

**특징**:
- 경량 모델로 빠른 학습 및 전송 가능

---

### `network.py`
**역할**: 네트워크 시뮬레이션 및 비용 측정

**주요 클래스**:
- `Link`: 네트워크 링크 시뮬레이션
  - 지연 시간(latency), 지터(jitter), 대역폭(bandwidth), 패킷 손실률(loss) 모델링
  - `transmit()`: 패킷 전송 시뮬레이션 (성공/실패, 지연 시간 계산)
  
- `CostMeter`: 통신 비용 및 시간 비용 측정
  - 라운드별 통계 추적
  - 누적 비용 계산 (총 통신량, 총 시간)
  - `end_round()`: 라운드 종료 시 비용 정산

**특징**:
- 위성-항공-지상 통합망의 통신 특성 반영
- 비용 측정을 통한 효율성 평가

---

### `fl_nodes.py`
**역할**: 연합학습 노드 정의 (클라이언트, UAV, 위성)

**주요 클래스**:
- `Packet`: 네트워크 패킷 구조
  - src, dst, timestamp, data, type 정보 포함

- `FLClient`: 연합학습 클라이언트
  - DCS 상태 정보: 통신 품질, 연산 능력, 데이터 유의성, 기여도
  - `calculate_dcs_score()`: DCS 점수 계산 (α·qi + β·ci + γ·di + δ·gi)
  - `train()`: 로컬 학습 수행 (전역 모델 또는 클러스터 모델 사용)

- `UAV`: 구역별 집계자
  - `assigned_clients`: 할당된 클라이언트 목록
  - `select_clients_dcs()`: DCS를 통한 클라이언트 선택
  - `run()`: 패킷 배치 처리 (8개씩 묶어서 전송)
  - `flush()`: 배치 패킷을 위성으로 전송

- `SatAgg`: 위성 집계자
  - `buffer`: 수집된 모델 업데이트 저장
  - `run()`: 모델 업데이트 수집 루프

**특징**:
- 계층적 구조 (Client → UAV → Satellite)
- DCS 지원

---

### `aggregation.py`
**역할**: 연합학습 집계 함수

**주요 함수**:
- `fedavg()`: Federated Averaging (FedAvg)
  - 여러 모델의 state_dict를 평균화
  - 가장 기본적인 집계 방법

**특징**:
- 단순하고 효율적인 평균 집계

---

### `clustering.py`
**역할**: 모델 유사도 기반 클러스터링

**주요 함수**:
- `extract_model_features()`: 모델 state_dict에서 특징 벡터 추출
  - Weight 파라미터만 사용 (경량화)
  - 평탄화하여 연결

- `compute_cosine_similarity()`: 코사인 유사도 계산
  - 두 특징 벡터 간의 유사도 (0~1 범위)

- `compute_similarity_matrix()`: 모델 간 유사도 행렬 계산
  - PCA를 통한 차원 축소 (경량화)
  - 모든 모델 쌍에 대한 유사도 계산
  - 진행 상황은 tqdm으로 표시

- `cluster_models()`: 모델 유사도 기반 클러스터링
  - Agglomerative Clustering 사용
  - 클러스터 수 자동 결정 또는 수동 지정
  - 거리 행렬 기반 클러스터링

- `cluster_based_aggregation()`: 클러스터별 모델 집계
  - 각 클러스터 내에서 FedAvg 수행
  - 클러스터별 모델 반환

**특징**:
- 경량화된 특징 추출 및 유사도 계산
- Non-IID 데이터 이질성 완화 목적

---

## Utils 모듈 (`src/utils/`)

### `model_utils.py`
**역할**: 모델 관련 유틸리티 함수

**주요 함수**:
- `get_state_dict_bytes()`: 모델 state_dict를 bytes로 직렬화
  - 네트워크 전송을 위한 변환
  
- `bytes_to_state_dict()`: bytes를 모델 state_dict로 역직렬화
  - 수신한 데이터를 모델로 복원

**특징**:
- 네트워크 전송을 위한 직렬화/역직렬화

---

### `progress_logger.py`
**역할**: 진행 상황 로깅 및 표시

**주요 클래스/함수**:
- `ProgressLogger`: 진행 상황 로깅 클래스
  - `log()`: 로그 메시지 기록 (파일 + 콘솔)
  - `tqdm()`: tqdm 래퍼 (진행 상황 표시 + 로그 기록)
  - `log_complete()`: 작업 완료 로그
  
- `init_progress_logger()`: 진행 상황 로거 초기화
  - 실험별로 별도 로그 파일 생성 (`logs/progress/{experiment}_{timestamp}.log`)
  
- `get_progress_logger()`: 현재 진행 상황 로거 가져오기

**특징**:
- tqdm 없이도 동작 (fallback 제공)
- 실시간 진행 상황 표시
- 파일과 콘솔에 동시 기록

---

## Config 모듈 (`src/config/`)

### `default_config.py`
**역할**: 실험 설정 관리

**주요 설정**:
- `SEED`: 시드 값 (재현성)
- `DEVICE`: 디바이스 (cuda/cpu)
- `NUM_UAV`: UAV 수
- `NUM_CLIENTS`: 전체 클라이언트 수
- `ROUNDS`: 최대 라운드 수
- `LOCAL_EPOCHS`: 로컬 학습 에포크 수
- `BATCH_SIZE`: 배치 크기
- `LR`: 학습률
- `MOMENTUM`: 모멘텀
- `SAMPLE_FRAC`: 라운드당 참여 클라이언트 비율
- `TARGET_ACC`: 기본 목표 정확도 (%)
- `CLUSTERING_TARGET_ACC`: 클러스터링 실험용 목표 정확도 (%)
- `CLIENT_UPLINK_BW`: 클라이언트 업링크 대역폭
- `UAV_SAT_BW`: UAV-위성 대역폭
- `ALPHA, BETA, GAMMA, DELTA`: DCS 가중치

**특징**:
- 모든 실험 설정을 한 곳에서 관리
- 실험별 목표 정확도 분리

---

## Experiments 모듈 (`src/experiments/`)

### `bl1.py`
**역할**: BL1 실험 (기본 계층적 FedAvg)

**특징**:
- 랜덤 클라이언트 선택
- 단순 평균 집계 (FedAvg)
- DCS 및 클러스터링 미적용
- 목표: 베이스라인 성능 및 비용 측정

**주요 흐름**:
1. 데이터 로딩
2. 네트워크 토폴로지 구축
3. 각 라운드:
   - 랜덤 클라이언트 선택
   - 로컬 학습
   - 모델 전송
   - 위성에서 집계
   - 평가 (Global Accuracy)
   - 비용 측정

---

### `bl2.py`
**역할**: BL2 실험 (DCS Only)

**특징**:
- DCS를 통한 클라이언트 선택
- 클러스터링 미적용
- 목표: 시스템 이질성 관리 효과 측정 (비용 감소)

**주요 흐름**:
1. 데이터 로딩
2. 네트워크 토폴로지 구축
3. 클라이언트를 UAV에 할당
4. 각 라운드:
   - UAV별 DCS 수행
   - 선택된 클라이언트 학습
   - 모델 전송
   - 위성에서 집계
   - 평가 및 비용 측정

**차이점** (BL1 대비):
- 랜덤 선택 대신 DCS 점수 기반 선택
- UAV별로 독립적으로 클라이언트 선택

---

### `bl3.py`
**역할**: BL3 실험 (Clustering Only)

**특징**:
- 랜덤 클라이언트 선택
- 모델 유사도 기반 클러스터링 적용
- 클러스터별 모델 집계
- DCS 미적용
- 목표: 데이터 이질성 완화 효과 측정 (정확도 향상)
- 목표 정확도: 100%

**주요 흐름**:
1. 데이터 로딩
2. 네트워크 토폴로지 구축
3. 각 라운드:
   - 랜덤 클라이언트 선택
   - 로컬 학습 (클러스터 모델 또는 전역 모델 사용)
   - 모델 전송
   - 위성에서 클러스터링 수행
   - 클러스터별 모델 집계
   - 전역 모델 업데이트 (모든 클러스터 모델 평균)
   - 평가 및 비용 측정

**차이점** (BL1 대비):
- 클러스터링 단계 추가
- 클러스터별 모델 관리
- 클라이언트-클러스터 매핑 유지

---

### `abl1.py` (예정)
**역할**: ABL-1 실험 (HPFL - 제안 모델)

**예상 특징**:
- DCS + 클러스터링 모두 적용
- 목표: 두 전략의 시너지 효과 검증 (비용 감소 + 정확도 향상)
- 목표 정확도: 100%

---

## Scripts (`scripts/`)

### `run_experiment.sh`
**역할**: 개별 실험 실행 스크립트

**기능**:
- 실험 타입 선택 (bl1, bl2, bl3, abl1)
- 백그라운드 실행
- 로그 파일 자동 생성 (`logs/{experiment}/{experiment}_{timestamp}.log`)
- PID 파일 저장 (`logs/{experiment}.pid`)
- 의존성 확인
- 프로세스 상태 확인

**사용법**:
```bash
./scripts/run_experiment.sh bl1
./scripts/run_experiment.sh bl2
./scripts/run_experiment.sh bl3
```

---

### `clear.sh`
**역할**: 실험 중지 및 프로세스 정리

**기능**:
- PID 파일 기반 프로세스 종료
- Python 프로세스 검색 및 종료
- TERM 신호 후 KILL 신호로 강제 종료
- PID 파일 정리

**사용법**:
```bash
./scripts/clear.sh bl1      # 특정 실험 중지
./scripts/clear.sh all      # 모든 실험 중지
```

---

### `run_all.sh`
**역할**: 모든 실험 순차 실행

**기능**:
- BL1 → BL2 → BL3 → ABL-1 순차 실행
- 각 실험 완료 후 다음 실험 시작

**사용법**:
```bash
./scripts/run_all.sh
```

---

### `install_dependencies.sh`
**역할**: 의존성 설치 스크립트

**기능**:
- requirements.txt 기반 패키지 설치
- 설치 확인 및 버전 출력
- 누락된 패키지 알림

**사용법**:
```bash
./scripts/install_dependencies.sh
```

---

## 루트 디렉토리 파일

### `requirements.txt`
**역할**: Python 패키지 의존성 목록

**포함 패키지**:
- `torch>=1.9.0`: PyTorch
- `torchvision>=0.10.0`: TorchVision
- `datasets>=2.0.0`: Hugging Face Datasets
- `scikit-learn>=1.0.0`: scikit-learn (클러스터링용)
- `numpy>=1.21.0`: NumPy
- `tqdm>=4.60.0`: 진행 상황 표시

---

### `.gitignore`
**역할**: Git에서 제외할 파일/디렉토리

**제외 항목**:
- Python 캐시 (`__pycache__/`, `*.pyc`)
- 가상 환경 (`venv/`, `env/`)
- IDE 설정 (`.vscode/`, `.idea/`)
- 실험 결과 및 로그 (`logs/`, `results/`)
- 데이터셋 캐시 (`data/`, `datasets/`)
- 모델 체크포인트 (`*.pt`, `*.pth`)
- 백업 파일 (`back_up/`, `archive/`)
- 진행 상황 로그 (`logs/progress/`)

---

### `src/README.md`
**역할**: 프로젝트 개요 및 사용법

**내용**:
- 프로젝트 소개
- 디렉토리 구조
- 실험 모델 설명
- 실행 방법
- 평가 지표
- 설정 변경 방법

---

## 로그 및 결과 디렉토리

### `logs/`
**역할**: 실험 로그 저장

**구조**:
```
logs/
├── bl1/                    # BL1 실험 로그
│   └── bl1_YYYYMMDD_HHMMSS.log
├── bl2/                    # BL2 실험 로그
├── bl3/                    # BL3 실험 로그
├── abl1/                   # ABL-1 실험 로그
└── progress/               # 진행 상황 로그
    └── {experiment}_YYYYMMDD_HHMMSS.log
```

**로그 파일 내용**:
- 실험 진행 상황
- 각 라운드별 성능 지표
- 비용 측정 결과
- 에러 메시지

---

### `results/`
**역할**: 실험 결과 저장 (예정)

**예상 내용**:
- 성능 지표 (CSV, JSON)
- 그래프 및 시각화
- 통계 분석 결과

---

## 기타 디렉토리

### `back_up/`
**역할**: 백업 파일 저장

**내용**:
- 초기 구현 버전의 코드
- 테스트용 스크립트

---

### `materials/`
**역할**: 논문 및 자료 저장

**내용**:
- 논문 PDF
- 발표 자료
- 파라미터 문서

---

## 주요 데이터 흐름

### 1. 데이터 로딩
```
Hugging Face Dataset → Writer ID 그룹핑 → 클라이언트별 데이터셋 생성
```

### 2. 학습 라운드
```
클라이언트 선택 → 로컬 학습 → 모델 전송 (Client → UAV → Satellite)
```

### 3. 집계
```
BL1/BL2: 단순 평균 (FedAvg)
BL3/ABL-1: 클러스터링 → 클러스터별 집계 → 전역 모델 업데이트
```

### 4. 평가
```
전역 모델 → Global Test Set → Global Accuracy 계산
```

---

## 설정 변경 가이드

### 실험 파라미터 변경
`src/config/default_config.py`에서 수정:
- 클라이언트 수: `NUM_CLIENTS`
- 참여 비율: `SAMPLE_FRAC`
- 목표 정확도: `TARGET_ACC`, `CLUSTERING_TARGET_ACC`
- 네트워크 대역폭: `CLIENT_UPLINK_BW`, `UAV_SAT_BW`

### DCS 가중치 조정
`src/config/default_config.py`에서 수정:
- `ALPHA`: 통신 품질 가중치
- `BETA`: 연산 능력 가중치
- `GAMMA`: 데이터 유의성 가중치
- `DELTA`: 기여도 가중치

---

## 문제 해결

### 의존성 설치
```bash
./scripts/install_dependencies.sh
# 또는
pip install -r requirements.txt
```

### 실험 중지
```bash
./scripts/clear.sh {experiment_name}
```

### 로그 확인
```bash
tail -f logs/{experiment}/{experiment}_*.log
tail -f logs/progress/{experiment}_*.log
```

---

## 추가 정보

- 프로젝트 README: `src/README.md`
- 실행 스크립트 가이드: `scripts/README.md`
- 논문 자료: `materials/`

