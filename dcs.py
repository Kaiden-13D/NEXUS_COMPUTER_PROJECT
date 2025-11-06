"""
Dynamic Client Selection (DCS) utilities.

This module centralizes the scoring formula and normalization for client selection.
It provides the weights for different components of the score.
"""

import numpy as np

def normalize_features(values):
    """Scales a list of values to the range [0, 1].

    Args:
        values (list or np.ndarray): The values to normalize.

    Returns:
        A numpy array of normalized values.
    """
    values = np.array(values)
    min_val = values.min()
    max_val = values.max()
    
    if max_val - min_val > 0:
        return (values - min_val) / (max_val - min_val)
    else:
        return np.zeros(values.shape) # All values are the same

def compute_scores(clients, weights):
    """Computes selection scores for a list of clients.

    Args:
        clients (list): A list of Client objects.
        weights (dict): A dictionary of weights for scoring (alpha, beta, gamma, delta).

    Returns:
        A list of tuples, where each tuple contains a client and their score.
    """
    if not clients:
        return []

    # Extract features for normalization
    comm_qualities = [client.comm_quality for client in clients]
    compute_powers = [client.compute_power for client in clients]
    data_significances = [client.data_significance for client in clients]
    contributions = [client.last_loss if client.last_loss is not None else 0 for client in clients]

    # Normalize features
    norm_q = normalize_features(comm_qualities)
    norm_c = normalize_features(compute_powers)
    norm_d = normalize_features(data_significances)
    norm_g = normalize_features(contributions)

    scores = []
    for i, client in enumerate(clients):
        score = (
            weights['alpha'] * norm_q[i] +
            weights['beta'] * norm_c[i] +
            weights['gamma'] * norm_d[i] +
            weights['delta'] * norm_g[i]
        )
        scores.append((client, score))
    
    return scores
