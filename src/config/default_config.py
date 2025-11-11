"""
기본 설정
"""
import torch


class Config:
    """HPFL 실험 설정"""
    
    # 시드
    SEED = 42
    
    # 디바이스
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 네트워크 토폴로지
    NUM_UAV = 6
    NUM_CLIENTS = 100
    
    # 학습 설정
    ROUNDS = 50
    LOCAL_EPOCHS = 1
    BATCH_SIZE = 32
    LR = 0.01
    MOMENTUM = 0.9
    
    # 클라이언트 선택
    SAMPLE_FRAC = 0.1  # 라운드당 10% 클라이언트 참여
    
    # 목표 정확도
    TARGET_ACC = 87.0  # 목표 Global Accuracy (%) - 기본 실험용
    CLUSTERING_TARGET_ACC = 100.0  # 클러스터링 실험(BL3, ABL-1)용 목표 정확도 (%)
    
    # 네트워크 대역폭 (bps)
    CLIENT_UPLINK_BW = 200_000
    UAV_SAT_BW = 600_000
    
    # DCS 가중치 (균등 분배)
    ALPHA = 0.25  # 통신 품질
    BETA = 0.25   # 연산 능력
    GAMMA = 0.25  # 데이터 유의성
    DELTA = 0.25  # 기여도

