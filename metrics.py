
import torch
from torch.utils.data import DataLoader, ConcatDataset, random_split
import numpy as np

def compute_personalized_accuracy(client, device, test_loader):
    """Evaluates the accuracy of a client's personalized model on their local test set."""
    client.backbone.eval()
    client.head.eval()
    
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
    head = PersonalizedHead().to(device)
    
    # The global model only has a backbone
    backbone.load_state_dict(cluster_model_state)
    backbone.eval()
    head.eval()

    test_loader = DataLoader(
        global_test_dataset, 
        batch_size=128, 
        num_workers=2, 
        pin_memory=True
    )
    
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

class MetricsLogger:
    def __init__(self):
        self.metrics = {
            'round': [], 'personalized_accuracy': [], 'global_accuracy': [],
            'loss': [], 'comm_cost': [], 'time_cost': []
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

    def save_to_file(self, filename):
        import pandas as pd
        pd.DataFrame(self.metrics).to_csv(filename, index=False)

if __name__ == '__main__':
    from client import Client
    from data import FEMNISTDataset
    from torchvision.transforms import Compose, ToTensor, Normalize
    from datasets import Dataset as HFDataset
    import random

    def generate_dummy_data(num_samples, writer_id):
        return HFDataset.from_dict({
            'writer_id': [writer_id] * num_samples,
            'image': [torch.randn(28, 28).numpy() for _ in range(num_samples)],
            'label': [random.randint(0, 61) for _ in range(num_samples)]
        })

    transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
    dummy_hf_dataset = generate_dummy_data(100, 'f0000_00')
    full_dataset = FEMNISTDataset(dummy_hf_dataset, transform=transform)
    
    train_size = int(0.9 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    mock_client = Client(client_id=1, train_dataset=train_dataset, test_dataset=test_dataset, device='cpu')

    pers_acc = compute_personalized_accuracy(mock_client, 'cpu', mock_client.test_dataloader)
    print(f"Personalized accuracy for mock client: {pers_acc:.2f}%")

    global_test_hf = generate_dummy_data(200, 'global_test')
    global_test_dataset = FEMNISTDataset(global_test_hf, transform=transform)
    
    cluster_model_state = mock_client.backbone.state_dict()
    global_acc = compute_global_accuracy(cluster_model_state, [mock_client], global_test_dataset, 'cpu')
    print(f"Global accuracy for mock model: {global_acc:.2f}%")

    logger = MetricsLogger()
    logger.log_round(1, pers_acc, global_acc, 2.5, 12345, 30.5)
    print("\nMetrics module tests passed.")
