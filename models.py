"""
Model definitions for the federated learning simulator.

This module defines the shared CNN backbone and the personalized head for the clients.
It also includes helper functions to get and set model parameters for aggregation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class SimpleCNN(nn.Module):
    """A simple CNN to act as the shared backbone.
    
    Consists of two convolutional layers and one fully connected layer.
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 512) # 1024 = 64 * 4 * 4

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 1024) # Flatten the tensor
        x = F.relu(self.fc1(x))
        return x

class PersonalizedHead(nn.Module):
    """A personalized classifier head for each client.
    
    A simple fully-connected layer that takes the output of the backbone.
    """
    def __init__(self, input_dim=512, output_dim=10):
        super(PersonalizedHead, self).__init__()
        self.fc2 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc2(x)

class PersonalizedModel(nn.Module):
    """Combines the shared backbone and a personalized head.
    """
    def __init__(self, backbone, head):
        super(PersonalizedModel, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        features = self.backbone(x)
        output = self.head(features)
        return output

def get_model_parameters(model):
    """Returns a model's state_dict.
    """
    return model.state_dict()

def set_model_parameters(model, params):
    """Sets a model's parameters from a state_dict.
    """
    model.load_state_dict(params)
