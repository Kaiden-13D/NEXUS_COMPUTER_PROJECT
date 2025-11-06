"""
Evaluation metrics and logging utilities.

This module provides functions to compute various metrics such as personalized accuracy,
global accuracy, loss, communication cost, and time cost. It also handles logging
of these metrics for analysis.
"""

import torch
import numpy as np

def compute_personalized_accuracy(client, test_loader):
    """Computes the accuracy of a client's personalized model on a test set.

    Args:
        client (Client): The client whose model should be evaluated.
        test_loader (DataLoader): The DataLoader for the test set.

    Returns:
        The accuracy of the model on the test set.
    """
    client.model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = client.model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return correct / total

def compute_global_accuracy(global_model, global_test_loader):
    """Computes the accuracy of a global model on a global test set.

    Args:
        global_model (nn.Module): The global model to evaluate.
        global_test_loader (DataLoader): The DataLoader for the global test set.

    Returns:
        The accuracy of the model on the test set.
    """
    global_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in global_test_loader:
            outputs = global_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return correct / total

def estimate_comm_cost(payload_size_bytes, num_clients, is_uplink=True):
    """A simple function to estimate communication cost.
    
    (This is a placeholder for a more complex model.)
    """
    # Simple assumption: cost is proportional to payload size and number of clients
    base_cost_per_byte = 1e-6 # Arbitrary cost unit
    return payload_size_bytes * num_clients * base_cost_per_byte

def log_round_metrics(round_idx, metrics):
    """Logs the metrics for a given round to the console.
    
    Args:
        round_idx (int): The current round number.
        metrics (dict): A dictionary of metrics to log.
    """
    log_message = f"Round {round_idx+1}:"
    for key, value in metrics.items():
        log_message += f" | {key}: {value:.4f}"
    print(log_message)
