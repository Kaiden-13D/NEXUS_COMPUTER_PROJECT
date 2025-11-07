
import os
import random
import numpy as np
import torch
from tqdm import tqdm
from collections import OrderedDict

from config import CONFIG
from data import load_femnist_dataset, partition_data
from client import Client
from uav import UAV
from satellite import Satellite
from dcs import compute_scores
from metrics import MetricsLogger, compute_personalized_accuracy, compute_global_accuracy
from models import CNNBackbone # Import the backbone model

def set_seed(seed):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def initialize_simulation(config):
    """Initializes the entire simulation environment."""
    print("1. Initializing simulation...")
    set_seed(config['seed'])

    # Load and partition dataset
    print("   - Loading and partitioning dataset...")
    full_hf_dataset = load_femnist_dataset()
    
    # Split the Hugging Face dataset into training and testing sets
    hf_dataset_split = full_hf_dataset.train_test_split(test_size=0.1, seed=config['seed'])
    train_hf_dataset = hf_dataset_split['train']
    test_hf_dataset = hf_dataset_split['test']

    # Partition the training data for clients
    client_datasets = partition_data(train_hf_dataset, config['num_clients'])
    
    # Create a single global test set (PyTorch Dataset)
    from data import FEMNISTDataset
    from torchvision.transforms import Compose, ToTensor, Normalize
    
    transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
    global_test_dataset = FEMNISTDataset(test_hf_dataset, transform=transform)
    
    # Create clients
    print("   - Creating clients...")
    clients = []
    for i in range(config['num_clients']):
        # Assign random compute and communication quality for simulation purposes
        compute_power = np.random.uniform(0.5, 1.5)
        comm_quality = np.random.uniform(0.5, 1.5)
        client = Client(client_id=i, dataset=client_datasets[i], compute_power=compute_power, comm_quality=comm_quality)
        clients.append(client)

    # Create UAVs and assign clients
    print("   - Creating UAVs and assigning clients...")
    uavs = []
    clients_per_uav = config['clients_per_uav']
    for i in range(config['num_uavs']):
        start_idx = i * clients_per_uav
        end_idx = (i + 1) * clients_per_uav
        uav_clients = clients[start_idx:end_idx]
        uav = UAV(uav_id=i, clients=uav_clients)
        uavs.append(uav)

    # Create Satellite
    print("   - Creating Satellite...")
    satellite = Satellite(num_clusters_k=config['initial_clusters_k'])

    # Initialize a global model on the satellite to start with
    initial_global_model = CNNBackbone().state_dict()
    satellite.cluster_models = {0: initial_global_model} # Start with one cluster

    print("Initialization complete.")
    return clients, uavs, satellite, global_test_dataset

def run_experiment(config):
    """Runs the full HPFL experiment."""
    clients, uavs, satellite, global_test_dataset = initialize_simulation(config)
    logger = MetricsLogger()

    # Get baseline mode from config
    mode = config['baseline_mode']
    print(f"\n2. Starting experiment in mode: {mode}\n")

    for round_idx in range(1, config['num_rounds'] + 1):
        round_losses = []
        round_comm_costs = []
        round_time_costs = []
        uav_aggregated_models = {}

        # --- UAV and Client Level --- #
        for uav in tqdm(uavs, desc=f"Round {round_idx} - UAVs"):
            # Get the appropriate model for this UAV (based on previous round's clustering)
            # For simplicity, we can have a default model or more complex logic here.
            # In this version, we assume a single global model is broadcast to all.
            # A more advanced version would map UAVs to clusters.
            global_model_state = list(satellite.cluster_models.values())[0]

            # Client Selection
            if mode == 'Hierarchical_FedAvg' or mode == 'Clustering_Only':
                # Random selection
                selected_clients = random.sample(uav.clients, config['clients_to_select'])
            else: # 'DCS_Only' or 'HPFL'
                # DCS-based selection
                client_scores = compute_scores(uav.clients, config['dcs_weights'])
                selected_clients = [cs[0] for cs in sorted(client_scores, key=lambda x: x[1], reverse=True)[:config['clients_to_select']]]

            # Local Training
            client_updates = []
            max_train_time = 0
            for client in selected_clients:
                updated_params, train_time, comm_cost = client.local_train(
                    global_model_state, config['local_epochs'], config['learning_rate']
                )
                client_updates.append((updated_params, len(client.dataset)))
                round_losses.append(client.last_loss)
                round_comm_costs.append(comm_cost)
                if train_time > max_train_time:
                    max_train_time = train_time
            
            round_time_costs.append(max_train_time)

            # UAV Aggregation
            if client_updates:
                aggregated_params = uav.aggregate_updates(client_updates)
                
                # Convert list of numpy arrays back to a state_dict
                new_state_dict = uav.model.state_dict()
                for i, key in enumerate(new_state_dict.keys()):
                    new_state_dict[key] = torch.from_numpy(aggregated_params[i])

                uav.model.load_state_dict(new_state_dict)

            uav_aggregated_models[uav.uav_id] = uav.model.state_dict()

        # --- Satellite Level --- #
        if mode == 'Clustering_Only' or mode == 'HPFL':
            # Clustering and Global Aggregation
            cluster_models, _ = satellite.cluster_and_aggregate(uav_aggregated_models)
        else: # 'Hierarchical_FedAvg' or 'DCS_Only'
            # Simple global aggregation without clustering
            all_uav_states = list(uav_aggregated_models.values())
            global_model = satellite._federated_averaging(all_uav_states)
            cluster_models = {0: global_model}
        
        satellite.cluster_models = cluster_models

        # --- Evaluation --- #
        if round_idx % config['eval_every'] == 0:
            # Personalized Accuracy
            pers_accs = [compute_personalized_accuracy(c) for c in clients]
            avg_pers_acc = np.mean(pers_accs)

            # Global Accuracy (evaluate each cluster model and average)
            global_accs = [compute_global_accuracy(cm, clients, global_test_dataset) for cm in cluster_models.values()]
            avg_global_acc = np.mean(global_accs)

            # Log metrics
            logger.log_round(
                round_idx=round_idx,
                personalized_acc=avg_pers_acc,
                global_acc=avg_global_acc,
                avg_loss=np.mean(round_losses),
                comm_cost=np.sum(round_comm_costs),
                time_cost=np.sum(round_time_costs) # Simplified: sum of max times per UAV zone
            )

    print("\n3. Experiment finished.")
    # Save results
    results_dir = config['results_dir']
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    results_file = os.path.join(results_dir, f"{mode}_metrics.csv")
    logger.save_to_file(results_file)
    print(f"Results saved to {results_file}")

if __name__ == '__main__':
    # Run the main experiment with the settings from config.py
    run_experiment(CONFIG)
