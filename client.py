
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import numpy as np

from models import CNNBackbone, PersonalizedHead, get_model_parameters, set_model_parameters

class Client:
    """Represents a ground client in the HPFL simulation."""
    def __init__(self, client_id, dataset, compute_power=1.0, comm_quality=1.0, device='cpu'):
        self.client_id = client_id
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=32, shuffle=True) # 배치로 학습.. client.local_train()에서 사용됨.
        
        self.device = device

        # Hardware and network attributes
        self.compute_power = compute_power # e.g., 1.0 for baseline, <1.0 for slower
        self.comm_quality = comm_quality   # e.g., 1.0 for baseline, <1.0 for worse
        
        # Data attributes
        self.data_significance = len(dataset)
        
        # Model components
        self.backbone = CNNBackbone().to(self.device)
        self.head = PersonalizedHead().to(self.device)
        
        # Training state
        self.last_loss = -1
        self.optimizer = optim.SGD(list(self.backbone.parameters()) + list(self.head.parameters()), lr=0.01)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

    def local_train(self, shared_state_dict, epochs, lr):
        """Performs local training on the client's data.

        Args:
            shared_state_dict (OrderedDict): The state dict of the shared backbone from the UAV.
            epochs (int): The number of local training epochs.
            lr (float): The learning rate for the local optimizer.

        Returns:
            tuple: A tuple containing:
                - list: The updated backbone parameters (numpy arrays).
                - float: The training time in seconds.
                - int: The number of bytes transmitted (approximated).
        """
        # Update local backbone with the shared state
        self.backbone.load_state_dict(shared_state_dict)
        
        # Set optimizer learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.backbone.train()
        self.head.train()

        start_time = time.time()
        total_loss = 0.0
        num_batches = 0

        printed_batch_info = False # for debugging

        for epoch in range(epochs):
            for images, labels in self.dataloader:
                # --- Debug TEST CODE ---
                if not printed_batch_info and self.client_id == 0: #only print for client 0 once
                    print(f"\n[TEST] Client {self.client_id} (Epoch {epoch+1})")
                    print(f"  - DataLoader provided a batch of images with shape: {images.shape}")
                    print(f"  - DataLoader provided a batch of labels with shape: {labels.shape}")
                    printed_batch_info = True
                # --- END OF TEST CODE ---
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                
                # Forward pass
                features = self.backbone(images)
                outputs = self.head(features)
                
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1

        end_time = time.time()
        training_time = (end_time - start_time) / self.compute_power # Simulate compute power

        # Update last_loss for contribution scoring
        if num_batches > 0:
            self.last_loss = total_loss / num_batches

        # Get updated backbone parameters
        updated_backbone_params = get_model_parameters(self.backbone)
        
        # Estimate communication cost (size of backbone parameters)
        comm_cost_bytes = sum(p.nbytes for p in updated_backbone_params)

        return updated_backbone_params, training_time, comm_cost_bytes

    def compute_score(self, weights):
        """Computes the client's selection score based on multiple factors.
        Score S = alpha*q + beta*c + gamma*d + delta*g
        """
        alpha, beta, gamma, delta = weights['alpha'], weights['beta'], weights['gamma'], weights['delta']
        
        # For now, we use the raw values. Normalization should happen at the UAV level.
        q = self.comm_quality
        c = self.compute_power
        d = self.data_significance
        
        # Contribution score (g) - higher is better
        # Use inverse of loss. Add a small epsilon to avoid division by zero.
        g = 1.0 / (self.last_loss + 1e-6) if self.last_loss != -1 else 0
        
        score = (alpha * q) + (beta * c) + (gamma * d) + (delta * g)
        return score

    def get_head_params(self):
        """Returns the parameters of the personalized head."""
        return get_model_parameters(self.head)

if __name__ == '__main__':
    # This is a placeholder for example usage and basic testing.
    # To run this, we would need a dummy dataset from data.py
    from data import FEMNISTDataset
    from torchvision.transforms import Compose, ToTensor, Normalize
    from datasets import Dataset as HFDataset
    import random

    # 1. Create a dummy dataset for a single client
    def generate_dummy_data(num_samples):
        data = {
            'writer_id': ['f0000_00'] * num_samples,
            'image': [torch.randn(28, 28).numpy() for _ in range(num_samples)],
            'label': [random.randint(0, 61) for _ in range(num_samples)]
        }
        return HFDataset.from_dict(data)

    dummy_hf_dataset = generate_dummy_data(100)
    transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
    client_dataset = FEMNISTDataset(dummy_hf_dataset, transform=transform)

    # 2. Initialize a client
    client = Client(client_id=1, dataset=client_dataset, compute_power=0.8, comm_quality=0.9)
    print(f"Initialized client 1 with {client.data_significance} samples.")

    # 3. Simulate a training round
    # Get initial backbone state (usually from a server/UAV)
    initial_backbone_state = client.backbone.state_dict()
    
    print("\nStarting local training...")
    updated_params, train_time, comm_bytes = client.local_train(initial_backbone_state, epochs=1, lr=0.01)
    
    print(f"Local training finished in {train_time:.4f} seconds (simulated). ")
    print(f"Last training loss: {client.last_loss:.4f}")
    print(f"Estimated communication cost: {comm_bytes} bytes.")

    # 4. Compute selection score
    # These weights would typically come from the config
    dcs_weights = {'alpha': 0.25, 'beta': 0.25, 'gamma': 0.25, 'delta': 0.25}
    score = client.compute_score(dcs_weights)
    print(f"\nClient selection score: {score:.4f}")

    # 5. Get head parameters (for personalization)
    head_params = client.get_head_params()
    print(f"Extracted {len(head_params)} parameter tensors from the personalized head.")
