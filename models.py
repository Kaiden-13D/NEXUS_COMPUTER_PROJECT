
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class SimpleCNN(nn.Module):
    """A simple CNN backbone for FEMNIST.
    Matches the architecture often used in federated learning benchmarks.
    Input: 1x28x28 image
    Output: 62 classes (10 digits, 26 lowercase, 26 uppercase)
    """
    def __init__(self, num_classes=62):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=1)
        
        # Calculate the flattened size after conv layers
        # Input is 28x28. After conv1 (padding=1): 28x28. After pool: 14x14
        # After conv2 (padding=1): 14x14. After pool: 7x7
        self.fc1 = nn.Linear(64 * 5 * 5, 2048) # Adjusted flattened size
        self.fc2 = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5) # Adjusted flattened size
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class PersonalizedModel(nn.Module):
    """Combines a shared backbone with a personalized head.
    The backbone is the SimpleCNN, and the head is a simple linear layer.
    """
    def __init__(self, shared_backbone, personalized_head):
        super(PersonalizedModel, self).__init__()
        self.backbone = shared_backbone
        self.head = personalized_head

    def forward(self, x):
        features = self.backbone(x) # The backbone should output features
        output = self.head(features)
        return output

def get_model_parameters(model):
    """Extracts model parameters as a list of numpy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_model_parameters(model, parameters):
    """Sets model parameters from a list of numpy arrays."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

# Redefine the backbone to separate feature extraction from classification
class CNNBackbone(nn.Module):
    def __init__(self):
        super(CNNBackbone, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=1)
        self.fc1 = nn.Linear(64 * 5 * 5, 512) # Feature layer

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        return x

class PersonalizedHead(nn.Module):
    """A personalized classifier head."""
    def __init__(self, num_classes=62):
        super(PersonalizedHead, self).__init__()
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.fc2(x)

if __name__ == '__main__':
    # Example usage
    # 1. Create the shared backbone and personalized head
    backbone = CNNBackbone()
    head = PersonalizedHead()

    # 2. Combine them into a single model for a client
    client_model = PersonalizedModel(backbone, head)

    print("Client model architecture:")
    print(client_model)

    # 3. Get and set parameters
    params = get_model_parameters(client_model)
    print(f"\nExtracted {len(params)} parameter tensors.")

    # Simulate sending to a server and receiving back
    # Create a new model instance
    new_backbone = CNNBackbone()
    new_head = PersonalizedHead()
    new_client_model = PersonalizedModel(new_backbone, new_head)

    # Set the parameters
    set_model_parameters(new_client_model, params)
    print("Parameters set on a new model instance.")

    # Verify that the parameters are the same
    new_params = get_model_parameters(new_client_model)
    for p1, p2 in zip(params, new_params):
        assert (p1 == p2).all(), "Parameters do not match!"
    
    print("Parameter get/set functionality verified.")

    # Test forward pass
    dummy_input = torch.randn(10, 1, 28, 28) # Batch of 10 images
    output = client_model(dummy_input)
    print(f"\nOutput shape on dummy input: {output.shape}")
    assert output.shape == (10, 62), "Output shape is incorrect!"
    print("Forward pass successful.")
