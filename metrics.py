
import torch
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np

def compute_personalized_accuracy(client, device, test_loader=None):
    """Evaluates the accuracy of a client's personalized model on their local test set."""
    client.backbone.eval()
    client.head.eval()
    
    if test_loader is None:
        # Create a DataLoader for the client's dataset if not provided
        # Note: In a real scenario, clients should have separate train/test splits.
        # For this simulation, we can evaluate on their training data as a proxy.
        print("using client's own dataset for evaluation. consider providing a separate test_loader.")
        test_loader = DataLoader(client.dataset, batch_size=128)

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            # Move the data batch to the device
            images, labels = images.to(device), labels.to(device)    

            features = client.backbone(images)
            outputs = client.head(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total if total > 0 else 0
    return accuracy

def compute_global_accuracy(cluster_model_state, clients, global_test_dataset, device):
    """Evaluates the accuracy of a global (cluster) model on a global test set."""
    # Create a temporary model to load the state
    from models import CNNBackbone, PersonalizedHead
    backbone = CNNBackbone().to(device)
    head = PersonalizedHead().to(device) # A generic head for evaluation
    
    # The global model only has a backbone
    backbone.load_state_dict(cluster_model_state)
    backbone.eval()
    head.eval()

    test_loader = DataLoader(global_test_dataset, batch_size=128)
    
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            features = backbone(images)
            outputs = head(features) # Use the generic head
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total if total > 0 else 0
    return accuracy

def estimate_comm_cost(client_payloads, server_payloads):
    """Estimates the total communication cost for a round.

    Args:
        client_payloads (list): List of byte sizes of client-to-UAV uploads.
        server_payloads (list): List of byte sizes of UAV-to-satellite and back.

    Returns:
        int: Total bytes transferred.
    """
    return sum(client_payloads) + sum(server_payloads)

class MetricsLogger:
    """A simple class to store and manage metrics over rounds."""
    def __init__(self):
        self.metrics = {
            'round': [],
            'personalized_accuracy': [],
            'global_accuracy': [],
            'loss': [],
            'comm_cost': [],
            'time_cost': []
        }

    def log_round(self, round_idx, personalized_acc, global_acc, avg_loss, comm_cost, time_cost):
        self.metrics['round'].append(round_idx)
        self.metrics['personalized_accuracy'].append(personalized_acc)
        self.metrics['global_accuracy'].append(global_acc)
        self.metrics['loss'].append(avg_loss)
        self.metrics['comm_cost'].append(comm_cost)
        self.metrics['time_cost'].append(time_cost)
        
        print(f"Round {round_idx:3d} | Pers. Acc: {personalized_acc:6.2f}% | "
              f"Global Acc: {global_acc:6.2f}% | Avg Loss: {avg_loss:.4f} | "
              f"Comm Cost: {comm_cost/1024:8.2f} KB | Time: {time_cost:.2f}s")

    def get_metrics(self):
        return self.metrics

    def save_to_file(self, filename):
        """Saves the logged metrics to a CSV file."""
        import pandas as pd
        df = pd.DataFrame(self.metrics)
        df.to_csv(filename, index=False)

if __name__ == '__main__':
    # This is a placeholder for example usage and basic testing.
    # We need mock clients and data to test the metrics functions.
    from client import Client
    from data import FEMNISTDataset
    from torchvision.transforms import Compose, ToTensor, Normalize
    from datasets import Dataset as HFDataset
    import random

    # 1. Create a mock client with a dataset
    def generate_dummy_data(num_samples, writer_id):
        data = {
            'writer_id': [writer_id] * num_samples,
            'image': [torch.randn(28, 28).numpy() for _ in range(num_samples)],
            'label': [random.randint(0, 61) for _ in range(num_samples)]
        }
        return HFDataset.from_dict(data)

    transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
    dummy_hf_dataset = generate_dummy_data(50, 'f0000_00')
    client_dataset = FEMNISTDataset(dummy_hf_dataset, transform=transform)
    mock_client = Client(client_id=1, dataset=client_dataset)

    # 2. Test Personalized Accuracy
    pers_acc = compute_personalized_accuracy(mock_client)
    print(f"Personalized accuracy for mock client: {pers_acc:.2f}%")
    # Accuracy will be low as the model is untrained.

    # 3. Test Global Accuracy
    # Create a dummy global test set
    global_test_hf = generate_dummy_data(200, 'global_test')
    global_test_dataset = FEMNISTDataset(global_test_hf, transform=transform)
    
    # Get a model state (e.g., from a UAV or satellite)
    cluster_model_state = mock_client.backbone.state_dict()
    global_acc = compute_global_accuracy(cluster_model_state, [mock_client], global_test_dataset)
    print(f"Global accuracy for mock model: {global_acc:.2f}%")

    # 4. Test MetricsLogger
    logger = MetricsLogger()
    logger.log_round(round_idx=1, personalized_acc=pers_acc, global_acc=global_acc, 
                     avg_loss=2.5, comm_cost=12345, time_cost=30.5)
    logger.log_round(round_idx=2, personalized_acc=pers_acc + 5, global_acc=global_acc + 2, 
                     avg_loss=2.1, comm_cost=12350, time_cost=31.2)

    metrics_data = logger.get_metrics()
    print("\nLogged metrics:")
    print(metrics_data)
    assert len(metrics_data['round']) == 2

    # logger.save_to_file('test_metrics.csv')
    # print("\nMetrics saved to test_metrics.csv")

    print("\nMetrics module tests passed.")
