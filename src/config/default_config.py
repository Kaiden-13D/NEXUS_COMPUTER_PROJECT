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
    ROUNDS = 50
    LOCAL_EPOCHS = 3  # Increased to 3 for better convergence and lower loss
    BATCH_SIZE = 32
    LR = 0.005  # Reduced from 0.01 to 0.005 for more stable training and lower loss
    MOMENTUM = 0.9
    
    # Client selection
    SAMPLE_FRAC = 0.3  # 30% for random selection (BL1, BL3)
    DCS_SAMPLE_FRAC = 0.25  # 25% for DCS-based selection (BL2, ABL1) - Increased for better convergence
    # Note: Increased to 0.25 to select more clients for lower loss and 90% accuracy target
    
    # Target accuracy
    TARGET_ACC = 90.0  # Target Global Accuracy (%) - for baseline experiments
    CLUSTERING_TARGET_ACC = 100.0  # Target accuracy (%) for clustering experiments (BL3, ABL-1)
    
    # Clustering settings
    NUM_CLUSTERS = 3  # Fixed number of clusters (determined in round 0, then fixed)
    AUTO_DETERMINE_CLUSTERS = True  # If True, auto-determine in round 0, else use NUM_CLUSTERS
    
    # Network bandwidth (bps)
    CLIENT_UPLINK_BW = 200_000
    UAV_SAT_BW = 600_000
    
    # DCS weights (optimized for ABL1: Data Significance focused for lower loss)
    # Strategy: Prioritize high-quality data to reduce loss and improve convergence
    ALPHA = 0.25  # Communication quality (balanced)
    BETA = 0.15   # Computational capability (reduced)
    GAMMA = 0.45  # Data significance (increased - prioritize high-quality data for lower loss)
    DELTA = 0.15  # Contribution (reduced)

