import sys
import os
import random
import torch
from collections import Counter

# We are running as src.verify_changes, so we can import from .config, .core
from .config.default_config import Config
from .core.data import setup_femnist_by_writer
from .core.fl_nodes import FLClient

def verify_changes():
    print("=== Verifying Data Distribution and Client Scoring ===")
    
    # 1. Setup Data Distribution
    print("\n[1] Setting up FEMNIST with 'tiered' distribution...")
    Config.DATA_DISTRIBUTION = "tiered"
    Config.LABEL_NOISE_RATIO = 0.1
    
    # Use a small number of clients for quick verification
    num_clients = 10
    
    try:
        train_datasets, test_datasets, num_classes = setup_femnist_by_writer(num_clients=num_clients)
    except Exception as e:
        print(f"Error setting up data: {e}")
        # Print full traceback for debugging
        import traceback
        traceback.print_exc()
        return

    print(f"\n[2] Verifying Client Tiers and Scoring...")
    
    clients = []
    for i in range(num_clients):
        # Create a mock link and loader
        client = FLClient(
            id=f"client_{i}",
            uav_id="uav_0",
            link=None,
            train_loader=torch.utils.data.DataLoader(train_datasets[i], batch_size=32)
        )
        clients.append(client)
        
    # Compute scores
    print(f"{'Client ID':<10} | {'Size':<6} | {'Entropy':<8} | {'Coverage':<8} | {'Score':<8} | {'Noise Ratio':<11}")
    print("-" * 70)
    
    for c in clients:
        # Access underlying dataset to check noise ratio
        ds = c.train_loader.dataset
        noise_ratio = getattr(ds, "noise_ratio", 0.0)
        
        # Compute significance components manually to verify
        c.compute_data_significance() # This updates c.data_significance
        
        # We can't easily get the individual components from the method return (it returns final score)
        # But we can inspect the dataset
        labels = []
        for _, y in c.train_loader:
             labels.extend(y.tolist())
             if len(labels) > 1000: break
        labels = labels[:1000]
        
        cnt = Counter(labels)
        total = len(labels)
        if total > 0:
            probs = [v/total for v in cnt.values()]
            entropy = -sum(p * torch.log(torch.tensor(p + 1e-12)) for p in probs).item()
            k = max(2, len(cnt))
            norm_entropy = entropy / torch.log(torch.tensor(float(k))).item()
            norm_coverage = len(cnt) / 62.0
        else:
            norm_entropy = 0.0
            norm_coverage = 0.0
            
        norm_size = min(1.0, len(ds) / Config.MAX_CLIENT_DATA_SIZE)
        
        score = c.data_significance
        
        print(f"{c.id:<10} | {len(ds):<6} | {norm_entropy:.4f}   | {norm_coverage:.4f}   | {score:.4f}   | {noise_ratio:.4f}")

    print("\n[3] Verification Criteria:")
    print("- 'Good' clients should have high Size (~2x avg).")
    print("- 'Bad' clients should have low Size (~0.5x avg) and Noise Ratio > 0.")
    print("- 'Medium' clients should be in between.")
    print("- Score should reflect these factors.")

if __name__ == "__main__":
    verify_changes()
