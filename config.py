"""
Central configuration for experiments.

This file contains all the parameters that can be varied for the HPFL simulation.
Centralizing configuration simplifies experimentation and parameter sweeps.
"""

config = {
    # Data and model configuration
    "dataset": "MNIST",  # "MNIST" or "FEMNIST"
    "scenario": "strong",  # "weak", "medium", or "strong" Non-IID

    # Federated learning parameters
    "rounds": 100,
    "local_epochs": 5,
    "learning_rate": 0.01,

    # Hierarchical structure
    "num_clients": 100,
    "num_uavs": 10,
    "max_clients_per_uav": 10,

    # Clustering parameters
    "initial_clusters_k": 3,
    "cluster_threshold": 0.5,  # Similarity threshold for merging/splitting

    # DCS (Dynamic Client Selection) parameters
    "dcs_weights": {
        "alpha": 0.25,  # Communication quality
        "beta": 0.25,   # Compute capacity
        "gamma": 0.25,  # Data significance
        "delta": 0.25,  # Contribution (loss improvement)
    },

    # Experiment mode
    "baseline_mode": "HPFL",  # "FedAvg", "DCS_only", "Clustering_only", "HPFL"
}
