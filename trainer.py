"""
Training loop for the HPFL simulator.

This module orchestrates the federated training rounds for HPFL and baseline experiments.
It manages the simulation setup, runs the training rounds, and executes experiments.
"""

import torch
from config import config
from data import load_dataset, partition_data
from models import SimpleCNN, PersonalizedHead, get_model_parameters, set_model_parameters
from client import Client
from uav import UAV
from satellite import Satellite
from metrics import compute_personalized_accuracy, log_round_metrics

def initialize_simulation(config):
    """Sets up the simulation environment.

    Args:
        config (dict): The experiment configuration.

    Returns:
        A tuple containing the list of clients, UAVs, the satellite, and the global test loader.
    """
    print("Initializing simulation...")
    # Load data
    train_dataset = load_dataset("./data", config["dataset"], train=True)
    test_dataset = load_dataset("./data", config["dataset"], train=False)
    client_dataloaders = partition_data(train_dataset, config["num_clients"], config["scenario"])
    global_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128)

    # Create models
    shared_backbone = SimpleCNN()
    initial_shared_state = get_model_parameters(shared_backbone)

    # Create clients
    clients = []
    for i in range(config["num_clients"]):
        personal_head = PersonalizedHead()
        client = Client(f"client_{i}", client_dataloaders[i], shared_backbone, personal_head)
        clients.append(client)

    # Create UAVs and assign clients
    uavs = []
    clients_per_uav = config["num_clients"] // config["num_uavs"]
    for i in range(config["num_uavs"]):
        start_idx = i * clients_per_uav
        end_idx = (i + 1) * clients_per_uav
        uav_clients = clients[start_idx:end_idx]
        uav = UAV(f"uav_{i}", uav_clients)
        uav.shared_model_state = initial_shared_state
        uavs.append(uav)

    # Create satellite
    satellite = Satellite(config["initial_clusters_k"])
    
    print("Initialization complete.")
    return clients, uavs, satellite, global_test_loader

def run_round(round_idx, uavs, satellite, config):
    """Runs a single round of federated learning.

    Args:
        round_idx (int): The current round index.
        uavs (list): The list of UAVs.
        satellite (Satellite): The satellite object.
        config (dict): The experiment configuration.
    """
    print(f"--- Round {round_idx+1} ---")
    uav_aggregated_models = []
    total_comm_cost = 0
    total_time_cost = 0

    for uav in uavs:
        # 1. Client Selection (DCS)
        selected_clients = uav.select_clients(m=config["max_clients_per_uav"], weights=config["dcs_weights"])
        
        # 2. Local Training
        client_updates = []
        round_training_times = []
        for client in selected_clients:
            deltas, _, training_time, num_samples = client.local_train(
                uav.shared_model_state, config["local_epochs"], config["learning_rate"]
            )
            client_updates.append((deltas, num_samples))
            round_training_times.append(training_time)
        
        if not client_updates:
            continue

        # 3. UAV Aggregation
        aggregated_deltas = uav.aggregate_updates(client_updates)
        
        # Apply aggregated deltas to the UAV's model
        current_uav_model = uav.shared_model_state.copy()
        for key in aggregated_deltas:
            current_uav_model[key] += aggregated_deltas[key]
        uav_aggregated_models.append(current_uav_model)
        
        # Update time cost (max training time in the round)
        total_time_cost += max(round_training_times) if round_training_times else 0

    # 4. Satellite Clustering and Aggregation
    if config["baseline_mode"] in ["Clustering_only", "HPFL"]:
        clusters = satellite.cluster_models(uav_aggregated_models)
        cluster_models = satellite.aggregate_clusters(clusters)
    else: # FedAvg or DCS_only (simple aggregation)
        clusters = {"global": uav_aggregated_models}
        cluster_models = satellite.aggregate_clusters(clusters)

    # 5. Model Update
    # In a real scenario, UAVs would be assigned to a cluster.
    # Here, we simplify and give all UAVs the first cluster's model (or the global one).
    global_model_state = list(cluster_models.values())[0]
    for uav in uavs:
        uav.shared_model_state = global_model_state

    # Update clients with the new shared state for the next round
    for uav in uavs:
        for client in uav.clients:
            set_model_parameters(client.model.backbone, uav.shared_model_state)


def run_experiment(config):
    """Runs the full HPFL simulation experiment."""
    clients, uavs, satellite, global_test_loader = initialize_simulation(config)
    
    for r in range(config["rounds"]):
        run_round(r, uavs, satellite, config)
        
        # Evaluate metrics
        all_accuracies = []
        for client in clients:
            # Use a subset of test data for quick personalized eval
            # In a real scenario, each client has its own test set.
            acc = compute_personalized_accuracy(client, global_test_loader)
            all_accuracies.append(acc)
        
        avg_personalized_acc = sum(all_accuracies) / len(all_accuracies)
        
        metrics = {"Personalized Accuracy": avg_personalized_acc}
        log_round_metrics(r, metrics)

if __name__ == '__main__':
    # This allows running the simulation directly
    run_experiment(config)
