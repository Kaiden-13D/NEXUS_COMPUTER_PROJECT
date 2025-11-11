#!/bin/bash

# 모든 실험을 순차적으로 실행하는 스크립트
# Usage: ./scripts/run_all.sh

set -e

# 색상 정의
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 스크립트 디렉토리로 이동
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

echo -e "${GREEN}=== Running All Experiments Sequentially ===${NC}"

EXPERIMENTS=("bl1" "bl2" "bl3" "abl1")

for exp in "${EXPERIMENTS[@]}"; do
    echo -e "\n${YELLOW}=== Starting ${exp} ===${NC}"
    ./scripts/run_experiment.sh "$exp"
    
    # 실험 완료 대기 (간단한 체크)
    PID_FILE="logs/${exp}.pid"
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        echo "Waiting for ${exp} to complete (PID: $PID)..."
        
        # 프로세스가 종료될 때까지 대기
        while ps -p "$PID" > /dev/null 2>&1; do
            sleep 5
        done
        
        echo -e "${GREEN}${exp} completed${NC}"
    fi
    
    # 다음 실험 전 잠시 대기
    sleep 2
done

echo -e "\n${GREEN}=== All Experiments Completed ===${NC}"

