# 실험 실행 스크립트

## 의존성 설치

먼저 필요한 패키지를 설치하세요:

```bash
# 의존성 설치
./scripts/install_dependencies.sh

# 또는 수동 설치
pip install -r requirements.txt
```

## 사용 방법

### 개별 실험 실행

```bash
# BL1 실험 실행
./scripts/run_experiment.sh bl1

# BL2 실험 실행
./scripts/run_experiment.sh bl2

# BL3 실험 실행
./scripts/run_experiment.sh bl3

# ABL-1 실험 실행
./scripts/run_experiment.sh abl1
```

### 실험 중지

```bash
# 특정 실험 중지
./scripts/clear.sh bl1
./scripts/clear.sh bl2
./scripts/clear.sh bl3
./scripts/clear.sh abl1

# 모든 실험 중지
./scripts/clear.sh all
```

### 모든 실험 순차 실행

```bash
# 모든 실험을 순차적으로 실행 (각 실험이 완료될 때까지 대기)
./scripts/run_all.sh
```

## 로그 확인

실험 로그는 실험별 디렉토리에 저장됩니다:

```bash
# 실시간 로그 확인
tail -f logs/bl1/bl1_YYYYMMDD_HHMMSS.log
tail -f logs/bl2/bl2_YYYYMMDD_HHMMSS.log
tail -f logs/bl3/bl3_YYYYMMDD_HHMMSS.log
tail -f logs/abl1/abl1_YYYYMMDD_HHMMSS.log

# 최근 로그 확인
ls -lt logs/bl1/ | head -5
ls -lt logs/bl2/ | head -5
ls -lt logs/bl3/ | head -5
ls -lt logs/abl1/ | head -5
```

## PID 파일

각 실험의 PID는 `logs/{experiment_type}.pid`에 저장됩니다.

## 주의사항

- 실험은 백그라운드로 실행됩니다
- 실험을 중지하려면 `clear.sh` 스크립트를 사용하세요
- 여러 실험을 동시에 실행할 수 있지만, 리소스 사용량에 주의하세요

