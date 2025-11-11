#!/bin/bash

# HPFL 실험 중지 및 프로세스 정리 스크립트
# Usage: ./scripts/clear.sh [bl1|bl2|bl3|abl1|all]

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
EXPERIMENT_TYPE=${1:-all}

# PID 파일 디렉토리
PID_DIR="logs"

if [ ! -d "$PID_DIR" ]; then
    echo -e "${YELLOW}No logs directory found. Nothing to clear.${NC}"
    exit 0
fi

# 프로세스 종료 함수
kill_experiment() {
    local exp_type=$1
    local pid_file="$PID_DIR/${exp_type}.pid"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p "$pid" > /dev/null 2>&1; then
            echo -e "${YELLOW}Killing ${exp_type} experiment (PID: $pid)...${NC}"
            kill -TERM "$pid" 2>/dev/null || true
            sleep 2
            
            # 강제 종료가 필요한 경우
            if ps -p "$pid" > /dev/null 2>&1; then
                echo -e "${RED}Force killing ${exp_type} experiment (PID: $pid)...${NC}"
                kill -KILL "$pid" 2>/dev/null || true
                sleep 1
            fi
            
            if ! ps -p "$pid" > /dev/null 2>&1; then
                echo -e "${GREEN}${exp_type} experiment stopped${NC}"
            else
                echo -e "${RED}Failed to stop ${exp_type} experiment${NC}"
            fi
        else
            echo -e "${YELLOW}${exp_type} experiment (PID: $pid) is not running${NC}"
        fi
        rm -f "$pid_file"
    else
        echo -e "${YELLOW}No PID file found for ${exp_type}${NC}"
    fi
}

# Python 프로세스 검색 및 종료
kill_python_experiments() {
    echo -e "${YELLOW}Searching for running experiment processes...${NC}"
    
    # src.experiments 모듈을 실행하는 Python 프로세스 찾기
    local pids=$(ps aux | grep -E "python.*src\.experiments\.(bl1|bl2|bl3|abl1)" | grep -v grep | awk '{print $2}')
    
    if [ -z "$pids" ]; then
        echo -e "${GREEN}No running experiment processes found${NC}"
        return
    fi
    
    for pid in $pids; do
        echo -e "${YELLOW}Killing Python experiment process (PID: $pid)...${NC}"
        kill -TERM "$pid" 2>/dev/null || true
        sleep 2
        
        if ps -p "$pid" > /dev/null 2>&1; then
            echo -e "${RED}Force killing process (PID: $pid)...${NC}"
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done
    
    sleep 1
    
    # 남은 프로세스 확인
    local remaining=$(ps aux | grep -E "python.*src\.experiments\.(bl1|bl2|bl3|abl1)" | grep -v grep | awk '{print $2}')
    if [ -z "$remaining" ]; then
        echo -e "${GREEN}All experiment processes stopped${NC}"
    else
        echo -e "${RED}Warning: Some processes may still be running: $remaining${NC}"
    fi
}

# 메인 로직
if [ "$EXPERIMENT_TYPE" == "all" ]; then
    echo -e "${GREEN}=== Stopping all experiments ===${NC}"
    for exp in bl1 bl2 bl3 abl1; do
        kill_experiment "$exp"
    done
    kill_python_experiments
    echo -e "${GREEN}All experiments cleared${NC}"
elif [[ "$EXPERIMENT_TYPE" =~ ^(bl1|bl2|bl3|abl1)$ ]]; then
    echo -e "${GREEN}=== Stopping ${EXPERIMENT_TYPE} experiment ===${NC}"
    kill_experiment "$EXPERIMENT_TYPE"
    kill_python_experiments
    echo -e "${GREEN}${EXPERIMENT_TYPE} experiment cleared${NC}"
else
    echo -e "${RED}Error: Invalid experiment type${NC}"
    echo "Usage: $0 [bl1|bl2|bl3|abl1|all]"
    exit 1
fi

# PID 파일 정리
echo -e "${YELLOW}Cleaning up PID files...${NC}"
rm -f "$PID_DIR"/*.pid

echo -e "${GREEN}Done!${NC}"

