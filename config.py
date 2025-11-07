
# Centralized configuration for the HPFL simulation

CONFIG = {
    # 1. Dataset and Model Configuration
    'dataset': 'FEMNIST',
    'non_iid_scenario': 'non-iid-label', # or 'iid'
    'model_name': 'SimpleCNN',
    'max_samples_per_client': 500, # Max samples per client, or None for all

    # 2. Federated Learning Parameters
    'num_rounds': 50,          # Total number of global training rounds
    'local_epochs': 1,           # Number of local training epochs on each client
    'learning_rate': 0.01,
    'batch_size': 64,

    # 3. System Architecture
    'num_clients': 100,
    'num_uavs': 6,
    'clients_per_uav': 16, # num_clients / num_uavs
    'clients_to_select': 10, # Number of clients selected by each UAV per round (m)

    # 4. HPFL Specific Parameters
    # 4.1. Dynamic Client Selection (DCS) weights
    'dcs_weights': {
        'alpha': 0.25, # Communication quality
        'beta': 0.25,  # Computation capacity
        'gamma': 0.25, # Data significance
        'delta': 0.25, # Contribution (loss improvement)
    },
    
    # 4.2. Similarity-based Clustering
    'initial_clusters_k': 3, # Initial number of clusters
    'cluster_threshold': 0.90, # Similarity threshold for merging/splitting (future use)

    # 5. Simulation Mode
    # 'baseline_mode': None, # Options: 'Hierarchical_FedAvg', 'DCS_Only', 'Clustering_Only', 'HPFL'
    'baseline_mode': 'HPFL', # Set to the full proposed method by default

    # 6. Evaluation
    'eval_every': 5, # Evaluate personalized and global accuracy every N rounds

    # 7. Miscellaneous
    'seed': 42, # For reproducibility
    'results_dir': 'results/', # Directory to save metrics and plots
}

if __name__ == '__main__':
    # Example of how to access the configuration
    print("--- HPFL Simulation Configuration ---")
    for key, value in CONFIG.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for sub_key, sub_value in value.items():
                print(f"  - {sub_key}: {sub_value}")
        else:
            print(f"{key}: {value}")
    print("-------------------------------------")

    # You can also access items directly
    print(f"\nNumber of global rounds: {CONFIG['num_rounds']}")
    print(f"Baseline mode: {CONFIG['baseline_mode']}")
