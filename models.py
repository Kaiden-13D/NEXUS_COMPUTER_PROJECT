
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

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
