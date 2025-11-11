#!/bin/bash

# 의존성 설치 스크립트
# Usage: ./scripts/install_dependencies.sh

set -e

# 색상 정의
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 스크립트 디렉토리로 이동
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

echo -e "${GREEN}=== Installing HPFL Dependencies ===${NC}"

# Python 버전 확인
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 not found${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo -e "${GREEN}Python version: $PYTHON_VERSION${NC}"

# pip 확인
if ! command -v pip3 &> /dev/null && ! command -v pip &> /dev/null; then
    echo -e "${RED}Error: pip not found. Please install pip first.${NC}"
    exit 1
fi

PIP_CMD="pip3"
if ! command -v pip3 &> /dev/null; then
    PIP_CMD="pip"
fi

# requirements.txt 확인
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}Error: requirements.txt not found${NC}"
    exit 1
fi

echo -e "${YELLOW}Installing packages from requirements.txt...${NC}"
$PIP_CMD install -r requirements.txt

echo -e "\n${GREEN}=== Verifying Installation ===${NC}"

# 패키지 확인
python3 -c "
import sys
errors = []

try:
    import torch
    print('✓ torch:', torch.__version__)
except ImportError as e:
    errors.append('torch')
    print('✗ torch: Not installed')

try:
    import torchvision
    print('✓ torchvision:', torchvision.__version__)
except ImportError as e:
    errors.append('torchvision')
    print('✗ torchvision: Not installed')

try:
    import datasets
    print('✓ datasets:', datasets.__version__)
except ImportError as e:
    errors.append('datasets')
    print('✗ datasets: Not installed')

try:
    import sklearn
    print('✓ scikit-learn:', sklearn.__version__)
except ImportError as e:
    errors.append('scikit-learn')
    print('✗ scikit-learn: Not installed')

try:
    import numpy
    print('✓ numpy:', numpy.__version__)
except ImportError as e:
    errors.append('numpy')
    print('✗ numpy: Not installed')

if errors:
    print(f'\n{len(errors)} package(s) failed to install: {', '.join(errors)}')
    sys.exit(1)
else:
    print('\n✓ All packages installed successfully!')
"

if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}=== Installation Complete ===${NC}"
    echo "You can now run experiments with: ./scripts/run_experiment.sh [bl1|bl2|bl3|abl1]"
else
    echo -e "\n${RED}=== Installation Failed ===${NC}"
    echo "Please check the error messages above and try again."
    exit 1
fi

