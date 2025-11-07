
import numpy as np
from collections import OrderedDict
import torch

from client import Client # Assuming client.py is in the same directory
from models import CNNBackbone

class UAV:
    """Represents a UAV that aggregates updates from a group of clients."""
    def __init__(self, uav_id, clients):
        self.uav_id = uav_id
        self.clients = clients
        self.model = CNNBackbone() # Each UAV maintains a model

    def select_clients(self, m, weights):
        """Selects the top m clients based on their DCS scores."""
        if m >= len(self.clients):
            return self.clients

        scores = [client.compute_score(weights) for client in self.clients]
        
        # Normalize scores before selection (optional but good practice)
        # For simplicity, we select top-m based on raw scores here.
        
        # Get indices of top m clients
        top_m_indices = np.argsort(scores)[-m:]
        selected_clients = [self.clients[i] for i in top_m_indices]
        
        return selected_clients

    def aggregate_updates(self, client_updates):
        """Aggregates model updates from selected clients using Federated Averaging (FedAvg).

        Args:
            client_updates (list of tuples): Each tuple contains (client_state_dict, num_samples).

        Returns:
            OrderedDict: The aggregated model state_dict.
        """
        if not client_updates:
            return self.model.state_dict() # Return current state

        total_samples = sum(num_samples for _, num_samples in client_updates)
        
        # Get the keys from the first model
        state_dicts = [sd for sd, _ in client_updates]
        keys = state_dicts[0].keys()
        
        avg_state_dict = OrderedDict()

        for key in keys:
            # Sum weighted tensors for the current key
            summed_tensor = torch.zeros_like(state_dicts[0][key]) # Init on CPU
            for sd, num_samples in client_updates:
                weight = num_samples / total_samples
                summed_tensor += sd[key] * weight
            
            avg_state_dict[key] = summed_tensor
            
        return avg_state_dict

    def get_model_state(self):
        """Returns the state dictionary of the UAV's model."""
        return self.model.state_dict()

    def set_model_state(self, state_dict):
        """Sets the state of the UAV's model."""
        self.model.load_state_dict(state_dict)

if __name__ == '__main__':
    # This is a placeholder for example usage and basic testing.
    # We need dummy clients to test the UAV functionality.
    from data import FEMNISTDataset
    from torchvision.transforms import Compose, ToTensor, Normalize
    from datasets import Dataset as HFDataset
    import random

    # 1. Create dummy data and clients
    def generate_dummy_data(num_samples, writer_id):
        data = {
            'writer_id': [writer_id] * num_samples,
            'image': [torch.randn(28, 28).numpy() for _ in range(num_samples)],
            'label': [random.randint(0, 61) for _ in range(num_samples)]
        }
        return HFDataset.from_dict(data)

    transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
    
    clients = []
    for i in range(5):
        dummy_hf_dataset = generate_dummy_data(100 + i*10, f'f000{i}_00')
        client_dataset = FEMNISTDataset(dummy_hf_dataset, transform=transform)
        client = Client(client_id=i, dataset=client_dataset, compute_power=0.5 + i*0.1, comm_quality=0.7 + i*0.05)
        clients.append(client)
        # Simulate one training round to get a loss value for scoring
        client.local_train(client.backbone.state_dict(), epochs=1, lr=0.01)

    print(f"Created {len(clients)} dummy clients.")

    # 2. Initialize a UAV
    uav = UAV(uav_id=1, clients=clients)
    print("Initialized UAV 1.")

    # 3. Select clients
    dcs_weights = {'alpha': 0.25, 'beta': 0.25, 'gamma': 0.25, 'delta': 0.25}
    num_to_select = 3
    selected_clients = uav.select_clients(num_to_select, dcs_weights)
    print(f"\nSelected {len(selected_clients)} clients out of {len(clients)}.")
    for c in selected_clients:
        print(f"  - Client {c.client_id} with score {c.compute_score(dcs_weights):.4f}")

    # 4. Simulate local training and aggregation
    client_updates = []
    uav_model_state = uav.get_model_state()

    for client in selected_clients:
        print(f"\nTraining client {client.client_id}...")
        updated_state_dict, _, _ = client.local_train(uav_model_state, epochs=1, lr=0.01)
        client_updates.append((updated_state_dict, len(client.dataset)))
        print(f"Client {client.client_id} finished training.")

    print(f"\nAggregating updates from {len(client_updates)} clients...")
    aggregated_state_dict = uav.aggregate_updates(client_updates)
    print("Aggregation complete.")

    # 5. Update UAV model with aggregated parameters
    uav.set_model_state(aggregated_state_dict)
    print("UAV model state has been updated.")

    # Verify the update
    updated_uav_params = uav.model.state_dict()
    assert torch.allclose(updated_uav_params['fc.weight'], aggregated_state_dict['fc.weight']), "UAV model update failed!"
    print("UAV model update verified.")
