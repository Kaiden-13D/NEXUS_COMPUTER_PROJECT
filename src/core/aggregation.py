"""
Federated learning aggregation functions
"""
from typing import List, Dict, Optional
import torch


def fedavg(
    state_dicts: List[Dict], 
    weights: Optional[List[float]] = None
) -> Dict:
    """
    Federated Averaging (FedAvg) aggregation
    
    Args:
        state_dicts: List of client model state_dicts
        weights: List of client weights (e.g., data size)
                 None for uniform weights (1/n)
    
    Returns:
        Weighted averaged state_dict
    """
    if not state_dicts:
        raise ValueError("Empty state_dicts list")
    
    n = len(state_dicts)
    
    # Use uniform weights if not provided
    if weights is None:
        weights = [1.0 / n] * n
    else:
        if len(weights) != n:
            raise ValueError(f"weights length ({len(weights)}) must match state_dicts length ({n})")
        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            raise ValueError("Total weight cannot be zero")
        weights = [w / total_weight for w in weights]
    
    # Calculate weighted average
    result = {}
    for k in state_dicts[0]:
        result[k] = sum(
            state_dicts[i][k] * weights[i] 
            for i in range(n)
        )
    
    return result

