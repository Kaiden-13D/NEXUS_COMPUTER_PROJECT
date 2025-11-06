"""
Data loading and partitioning module.

This module provides functions to load datasets (e.g., MNIST, FEMNIST) and partition them
to simulate Non-IID data distributions among clients.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

def load_dataset(root, dataset_name, train=True):
    """Loads the specified dataset.

    Args:
        root (str): The root directory where the dataset is stored.
        dataset_name (str): The name of the dataset to load (e.g., "MNIST").
        train (bool): Whether to load the training or test set.

    Returns:
        A PyTorch Dataset.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if dataset_name == "MNIST":
        dataset = torchvision.datasets.MNIST(
            root=root,
            train=train,
            download=True,
            transform=transform
        )
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    return dataset

def partition_data(dataset, num_clients, scenario="strong"):
    """Partitions the dataset for a number of clients to simulate Non-IID data.

    Args:
        dataset: The dataset to partition.
        num_clients (int): The number of clients.
        scenario (str): The Non-IID scenario ("strong", "medium", "weak").

    Returns:
        A list of DataLoaders, one for each client.
    """
    if scenario == "strong":
        # Strong Non-IID: Each client gets data from only a few classes.
        # Simple implementation: Sort data by label and distribute chunks.
        labels = dataset.targets.numpy()
        sorted_indices = np.argsort(labels)
        
        # Shuffle within sorted chunks to add some randomness
        # This is a simple way to create shards.
        shards = np.array_split(sorted_indices, num_clients * 2) # Create more shards than clients
        np.random.shuffle(shards)
        
        client_data_indices = [np.concatenate(shards[i*2:(i+1)*2]) for i in range(num_clients)]

    else:
        # For now, only strong Non-IID is implemented
        raise NotImplementedError(f"Scenario '{scenario}' is not implemented yet.")

    client_dataloaders = []
    for indices in client_data_indices:
        client_images = dataset.data[indices]
        client_labels = dataset.targets[indices]
        
        # Add a channel dimension for grayscale images
        if len(client_images.shape) == 3:
            client_images = client_images.unsqueeze(1)
            
        # Normalize manually as transform is not applied on subset
        client_images = client_images.float() / 255.0
        client_images = (client_images - 0.5) / 0.5

        tensor_dataset = TensorDataset(client_images, client_labels)
        dataloader = DataLoader(tensor_dataset, batch_size=32, shuffle=True)
        client_dataloaders.append(dataloader)

    return client_dataloaders
