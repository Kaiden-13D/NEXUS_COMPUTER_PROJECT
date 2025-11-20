"""
Default configuration
"""
import torch


class Config:
    """HPFL experiment configuration"""
    
    # Seed
    SEED = 42
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Network topology
    NUM_UAV = 6
    NUM_CLIENTS = 100
    
    # Training settings
    ROUNDS = 40
    LOCAL_EPOCHS = 1
    BATCH_SIZE = 32
    LR = 0.01
    MOMENTUM = 0.9
    MAX_CLIENT_DATA_SIZE = 120000
    
    # Client selection
    SAMPLE_FRAC = 0.5  # 30% of clients participate per round
    
    # Target accuracy
    TARGET_ACC = 90.0  # Target Global Accuracy (%) - for baseline experiments
    CLUSTERING_TARGET_ACC = 100.0  # Target accuracy (%) for clustering experiments (BL3, ABL-1)
    
    # Clustering settings
    NUM_CLUSTERS = 3  # Fixed number of clusters (determined in round 0, then fixed)
    AUTO_DETERMINE_CLUSTERS = False  # If True, auto-determine in round 0, else use NUM_CLUSTERS
    
    # Network bandwidth (bps)
    CLIENT_UPLINK_BW = 200_000
    UAV_SAT_BW = 600_000
    
    # Data Distribution
    DATA_DISTRIBUTION = "tiered"  # "tiered" or "default"
    LABEL_NOISE_RATIO = 0.1  # Ratio of labels to flip for "Bad" clients
    
    # DCS weights (equal distribution)
    ALPHA = 0.1  # Communication quality
    BETA = 0.1   # Computational capability
    GAMMA = 0.5  # Data significance
    DELTA = 0.3  # Contribution
    
    # Data Significance Component Weights
    DS_SIZE_WEIGHT = 0.3
    DS_ENTROPY_WEIGHT = 0.35
    DS_COVERAGE_WEIGHT = 0.35

