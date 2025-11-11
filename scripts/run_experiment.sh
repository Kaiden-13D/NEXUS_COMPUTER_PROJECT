#!/bin/bash

# HPFL 실험 실행 스크립트
# Usage: ./scripts/run_experiment.sh [bl1|bl2|bl3|abl1]

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 스크립트 디렉토리로 이동
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# 실험 타입 확인
EXPERIMENT_TYPE=${1:-bl1}

if [[ ! "$EXPERIMENT_TYPE" =~ ^(bl1|bl2|bl3|abl1)$ ]]; then
    echo -e "${RED}Error: Invalid experiment type${NC}"
    echo "Usage: $0 [bl1|bl2|bl3|abl1]"
    exit 1
fi

# 로그 디렉토리 생성 (실험별 디렉토리)
mkdir -p logs/${EXPERIMENT_TYPE}
mkdir -p results

# 타임스탬프
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/${EXPERIMENT_TYPE}/${EXPERIMENT_TYPE}_${TIMESTAMP}.log"
PID_FILE="logs/${EXPERIMENT_TYPE}.pid"

echo -e "${GREEN}=== Starting ${EXPERIMENT_TYPE} Experiment ===${NC}"
echo "Log file: $LOG_FILE"
echo "PID file: $PID_FILE"

# Python 경로 확인
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 not found${NC}"
    exit 1
fi

# 의존성 확인 (선택적)
if [ -f "requirements.txt" ]; then
    echo -e "${YELLOW}Checking dependencies...${NC}"
    python3 -c "import torch; import datasets; import sklearn" 2>/dev/null || {
        echo -e "${YELLOW}Warning: Some dependencies may be missing. Install with: pip install -r requirements.txt${NC}"
    }
fi

# 실험 실행 (백그라운드)
# PYTHONUNBUFFERED=1: Python 출력 버퍼링 해제 (즉시 로그 파일에 기록)
echo -e "${GREEN}Running experiment in background...${NC}"
PYTHONUNBUFFERED=1 nohup python3 -u -m src.experiments.${EXPERIMENT_TYPE} > "$LOG_FILE" 2>&1 &
EXPERIMENT_PID=$!

# PID 저장
echo $EXPERIMENT_PID > "$PID_FILE"

echo -e "${GREEN}Experiment started with PID: $EXPERIMENT_PID${NC}"
echo "To view logs: tail -f $LOG_FILE"
echo "To stop experiment: ./scripts/clear.sh ${EXPERIMENT_TYPE}"

# 프로세스 확인
sleep 2
if ps -p $EXPERIMENT_PID > /dev/null; then
    echo -e "${GREEN}Experiment is running successfully${NC}"
else
    echo -e "${RED}Error: Experiment failed to start. Check log file: $LOG_FILE${NC}"
    rm -f "$PID_FILE"
    exit 1
fi

