import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from matplotlib import pyplot as plt
import copy
import random

# --- 1. Hyperparameters and Settings ---
class Args:
    def __init__(self):
        self.epochs = 30
        self.local_epochs = 2
        self.num_clients = 100
        self.num_zones = 6
        self.num_clusters_per_zone = 3 # New: Number of clusters within each zone
        self.batch_size = 64
        self.lr = 0.01
        self.momentum = 0.5
        self.seed = 42
        self.log_interval = 10

args = Args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# --- 2. Model Definition ---
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# --- 3. Entity Classes (UE, UAV, Satellite) ---

class UE:
    """Represents a User Equipment (Client)"""
    def __init__(self, id, data_loader):
        self.id = id
        self.data_loader = data_loader
        self.model = Net()

    def train(self, cluster_model_state, local_epochs, lr, momentum):
        """Train the client's model locally."""
        self.model.load_state_dict(copy.deepcopy(cluster_model_state))
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        
        for epoch in range(local_epochs):
            for data, target in self.data_loader:
                if len(data) == 0: continue
                optimizer.zero_grad()
                output = self.model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
        
        return self.model.state_dict()

    def evaluate_model_loss(self, model_state):
        """Calculates the loss of a given model on the client's local data."""
        eval_model = Net()
        eval_model.load_state_dict(copy.deepcopy(model_state))
        eval_model.eval()
        total_loss = 0
        total_samples = 0
        with torch.no_grad():
            for data, target in self.data_loader:
                if len(data) == 0: continue
                output = eval_model(data)
                loss = F.nll_loss(output, target, reduction='sum').item()
                total_loss += loss
                total_samples += len(data)
        
        if total_samples == 0:
            return float('inf')
        return total_loss / total_samples

class UAV:
    """Represents a UAV (Zone Server)"""
    def __init__(self, id, num_clusters):
        self.id = id
        self.clients = []
        self.num_clusters = num_clusters
        # Initialize cluster models for this zone
        self.cluster_models = {c: Net() for c in range(num_clusters)}

    def add_client(self, client):
        self.clients.append(client)

    def update_cluster_models_from_global(self, global_model_state):
        """Copy the global model state to all cluster models."""
        for c_model in self.cluster_models.values():
            c_model.load_state_dict(copy.deepcopy(global_model_state))

    def assign_clients_to_clusters(self):
        """
        Assign clients to clusters based on the minimum loss.
        Blueprint Section 5.G: IFCA/CFL-style cluster assignment.
        """
        assignments = {c: [] for c in range(self.num_clusters)}
        if not self.clients:
            return assignments

        print(f"  Zone {self.id}: Assigning {len(self.clients)} clients to {self.num_clusters} clusters...")
        for client in self.clients:
            losses = []
            for cluster_id in range(self.num_clusters):
                cluster_model_state = self.cluster_models[cluster_id].state_dict()
                loss = client.evaluate_model_loss(cluster_model_state)
                losses.append(loss)
            
            best_cluster_id = np.argmin(losses)
            assignments[best_cluster_id].append(client)
        
        # Log cluster distribution
        dist_str = ", ".join([f"C{c}: {len(clients)} clients" for c, clients in assignments.items()])
        print(f"  Zone {self.id}: Cluster distribution: {dist_str}")

        return assignments

    def train_zone(self, local_epochs, lr, momentum):
        """Orchestrate training within the zone for one round."""
        client_assignments = self.assign_clients_to_clusters()
        
        updated_cluster_weights = {}

        for cluster_id, assigned_clients in client_assignments.items():
            if not assigned_clients:
                # If a cluster has no clients, its model does not change
                updated_cluster_weights[cluster_id] = self.cluster_models[cluster_id].state_dict()
                continue

            cluster_model_state = self.cluster_models[cluster_id].state_dict()
            local_client_updates = []
            
            for client in assigned_clients:
                updated_weights = client.train(cluster_model_state, local_epochs, lr, momentum)
                local_client_updates.append(updated_weights)
            
            if local_client_updates:
                # Aggregate updates for this cluster
                aggregated_weights = self._aggregate_weights(local_client_updates)
                self.cluster_models[cluster_id].load_state_dict(aggregated_weights)
                updated_cluster_weights[cluster_id] = aggregated_weights
        
        return updated_cluster_weights

    def get_zone_summary_model(self):
        """
        Create a summary of the zone's models for the satellite.
        (Simplification) Averages all cluster models in the zone.
        """
        # In case a zone has no clients and thus no cluster models were trained
        if not self.cluster_models:
            return None
        
        cluster_model_states = [model.state_dict() for model in self.cluster_models.values()]
        zone_summary_state = self._aggregate_weights(cluster_model_states)
        
        zone_summary_model = Net()
        zone_summary_model.load_state_dict(zone_summary_state)
        return zone_summary_model

    def _aggregate_weights(self, client_weights):
        """Helper for FedAvg."""
        if not client_weights:
            return None
        
        agg_weights = copy.deepcopy(client_weights[0])
        for k in agg_weights.keys():
            agg_weights[k] = torch.stack([cw[k].float() for cw in client_weights], 0).mean(0)
        return agg_weights

class Satellite:
    """Represents the Global Server"""
    def __init__(self, num_zones, num_clusters_per_zone):
        self.global_model = Net()
        self.zones = [UAV(i, num_clusters_per_zone) for i in range(num_zones)]

    def distribute_clients(self, clients, client_zone_mapping):
        for client_id, zone_id in client_zone_mapping.items():
            self.zones[zone_id].add_client(clients[client_id])

    def train_round(self, epoch, args):
        """Execute one full round of hierarchical training."""
        print(f"--- Round {epoch+1}/{args.epochs} ---")
        
        # 1. Broadcast global model to all UAVs (which then update their cluster models)
        global_model_state = self.global_model.state_dict()
        for zone in self.zones:
            zone.update_cluster_models_from_global(global_model_state)

        # 2. Trigger parallel training in all zones
        zone_summary_models = []
        for zone in self.zones:
            zone.train_zone(args.local_epochs, args.lr, args.momentum)
            # 3. Collect zone summaries for global aggregation
            summary_model = zone.get_zone_summary_model()
            if summary_model:
                zone_summary_models.append(summary_model.state_dict())
        
        # 4. Aggregate zone summaries to update the global model
        if zone_summary_models:
            self._aggregate_global_model(zone_summary_models)

    def _aggregate_global_model(self, zone_model_weights):
        """Aggregate models from zones to update the global model (FedAvg)."""
        global_dict = self.global_model.state_dict()
        for k in global_dict.keys():
            global_dict[k] = torch.stack([zone_weights[k].float() for zone_weights in zone_model_weights], 0).mean(0)
        self.global_model.load_state_dict(global_dict)

    def test(self, test_loader):
        """Evaluate the global model."""
        self.global_model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = self.global_model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        return test_loss, accuracy

# --- 4. Data Preparation ---
def get_data_and_entities(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    client_zone_mapping = {i: random.randint(0, args.num_zones - 1) for i in range(args.num_clients)}
    
    # Non-IID data distribution
    num_labels_per_client = 2
    labels = train_dataset.targets
    client_data_indices = {i: np.empty(0, dtype=np.int64) for i in range(args.num_clients)}
    label_indices = [np.where(labels == i)[0] for i in range(10)]
    client_labels = {i: np.random.choice(10, num_labels_per_client, replace=False) for i in range(args.num_clients)}

    for client_id, labels_for_client in client_labels.items():
        for label in labels_for_client:
            num_clients_with_label = sum(1 for cid in range(args.num_clients) if label in client_labels[cid])
            num_samples = len(label_indices[label]) // num_clients_with_label
            rand_idx = np.random.choice(len(label_indices[label]), num_samples, replace=False)
            client_data_indices[client_id] = np.concatenate((client_data_indices[client_id], label_indices[label][rand_idx]))
    
    clients = []
    for i in range(args.num_clients):
        subset = Subset(train_dataset, client_data_indices[i])
        loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True)
        clients.append(UE(i, loader))

    print("--- Client & Zone & Data Distribution (Sample) ---")
    zone_counts = {z:0 for z in range(args.num_zones)}
    for cid, zid in client_zone_mapping.items():
        zone_counts[zid] += 1
    print(f"Zone client counts: {zone_counts}")

    return clients, test_loader, client_zone_mapping

# --- 5. Main Execution Loop ---
if __name__ == '__main__':
    print("Initializing Hierarchical Federated Learning Simulation...")
    
    clients, test_loader, client_zone_mapping = get_data_and_entities(args)

    satellite = Satellite(args.num_zones, args.num_clusters_per_zone)
    satellite.distribute_clients(clients, client_zone_mapping)
    
    test_losses = []
    accuracies = []

    print("\nStarting Training...\n")
    for epoch in range(args.epochs):
        satellite.train_round(epoch, args)
        
        test_loss, accuracy = satellite.test(test_loader)
        test_losses.append(test_loss)
        accuracies.append(accuracy)
        
        print(f"Round {epoch+1}/{args.epochs} -> Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%\n")

    # --- 6. Results Visualization ---
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(args.epochs), test_losses, marker='o')
    plt.title("Test Loss vs. Communication Rounds")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Test Loss")
    
    plt.subplot(1, 2, 2)
    plt.plot(range(args.epochs), accuracies, marker='o', color='r')
    plt.title("Accuracy vs. Communication Rounds")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Accuracy (%)")
    
    plt.tight_layout()
    plt.show()

    print("\nHierarchical Federated Learning Simulation Finished.")
    torch.save(satellite.global_model.state_dict(), "hierarchical_federated_model.pt")
    print("Final global model saved as 'hierarchical_federated_model.pt'")
