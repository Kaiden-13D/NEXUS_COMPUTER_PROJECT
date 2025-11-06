"""
Client representation for the federated learning simulator.

This module defines the Client class, which represents a ground client.
Each client has its own local dataset, compute and network attributes, and is responsible
for local training and computing its selection score.
"""

import torch
import torch.optim as optim
import torch.nn as nn
import time
import numpy as np

from models import PersonalizedModel, get_model_parameters, set_model_parameters

class Client:
    """Represents a single client in the federated learning system."""
    def __init__(self, client_id, dataloader, backbone, head):
        self.client_id = client_id
        self.dataloader = dataloader
        self.model = PersonalizedModel(backbone, head)
        
        # Simulate client heterogeneity
        self.compute_power = np.random.uniform(0.5, 1.5) # Relative speed
        self.comm_quality = np.random.uniform(0.5, 1.5)  # Relative quality
        self.data_significance = len(self.dataloader.dataset)
        self.last_loss = None

    def local_train(self, shared_state, epochs, lr):
        """Performs local training on the client's data.

        Args:
            shared_state (dict): The state_dict of the shared backbone model.
            epochs (int): The number of local epochs to train for.
            lr (float): The learning rate for the optimizer.

        Returns:
            A tuple containing:
            - The updated shared model parameter deltas.
            - The updated personal head parameters.
            - The time taken for training.
            - The number of data samples used.
        """
        set_model_parameters(self.model.backbone, shared_state)
        self.model.train()
        
        optimizer = optim.SGD(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        start_time = time.time()
        
        initial_loss = 0
        for epoch in range(epochs):
            epoch_loss = 0.0
            for images, labels in self.dataloader:
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            if epoch == 0:
                initial_loss = epoch_loss / len(self.dataloader)

        end_time = time.time()
        training_time = (end_time - start_time) / self.compute_power
        
        final_loss = epoch_loss / len(self.dataloader)
        self.last_loss = initial_loss - final_loss # Contribution score

        # Calculate parameter deltas for the backbone
        updated_backbone_params = get_model_parameters(self.model.backbone)
        deltas = {key: updated_backbone_params[key] - shared_state[key] for key in shared_state}
        
        return deltas, get_model_parameters(self.model.head), training_time, len(self.dataloader.dataset)

    def compute_score(self, weights):
        """Computes the selection score for this client.

        The score is a weighted sum of communication quality, compute capacity,
        data significance, and contribution (loss improvement).
        
        S = alpha * q + beta * c + gamma * d + delta * g

        Args:
            weights (dict): A dictionary with weights for each component.

        Returns:
            The client's selection score.
        """
        # These values should be normalized across all clients by the UAV
        q = self.comm_quality
        c = self.compute_power
        d = self.data_significance
        g = self.last_loss if self.last_loss is not None else 0
        
        score = (
            weights['alpha'] * q +
            weights['beta'] * c +
            weights['gamma'] * d +
            weights['delta'] * g
        )
        
        return score
