
import numpy as np

def normalize_features(feature_values):
    """Normalizes a list of feature values to the [0, 1] range."""
    min_val = np.min(feature_values)
    max_val = np.max(feature_values)
    
    if max_val == min_val:
        return np.zeros_like(feature_values)
        
    return (feature_values - min_val) / (max_val - min_val)

def compute_scores(clients, weights):
    """Computes the DCS score for each client and returns a list of (client, score) tuples.

    The score is a weighted sum of normalized features:
    S = alpha*q + beta*c + gamma*d + delta*g
    
    Args:
        clients (list of Client): The clients to be scored.
        weights (dict): A dictionary with keys 'alpha', 'beta', 'gamma', 'delta'.

    Returns:
        list: A list of tuples, where each tuple is (client, score).
    """
    if not clients:
        return []

    alpha = weights.get('alpha', 0.25)
    beta = weights.get('beta', 0.25)
    gamma = weights.get('gamma', 0.25)
    delta = weights.get('delta', 0.25)

    # 1. Extract raw feature values from all clients
    comm_qualities = np.array([client.comm_quality for client in clients])
    compute_powers = np.array([client.compute_power for client in clients])
    data_significance = np.array([client.data_significance for client in clients])
    
    # Contribution score (g) is inverse of loss. Higher is better.
    # Add a small epsilon to avoid division by zero.
    contributions = np.array([1.0 / (client.last_loss + 1e-6) if client.last_loss != -1 else 0 for client in clients])

    # 2. Normalize each feature across the clients
    norm_q = normalize_features(comm_qualities)
    norm_c = normalize_features(compute_powers)
    norm_d = normalize_features(data_significance)
    norm_g = normalize_features(contributions)

    # 3. Compute the final weighted score for each client
    scores = (alpha * norm_q) + (beta * norm_c) + (gamma * norm_d) + (delta * norm_g)
    
    client_scores = list(zip(clients, scores))
    
    return client_scores

if __name__ == '__main__':
    # This is a placeholder for example usage and basic testing.
    # We need a mock Client class to test the DCS functionality.

    class MockClient:
        def __init__(self, client_id, comm, compute, data, loss):
            self.client_id = client_id
            self.comm_quality = comm
            self.compute_power = compute
            self.data_significance = data
            self.last_loss = loss

        def __repr__(self):
            return f"Client(id={self.client_id})"

    # 1. Create a list of mock clients with varying attributes
    clients = [
        MockClient(client_id=1, comm=0.9, compute=1.0, data=100, loss=0.5),
        MockClient(client_id=2, comm=0.5, compute=0.8, data=200, loss=0.8),
        MockClient(client_id=3, comm=0.7, compute=1.2, data=50, loss=0.2), # High contribution
        MockClient(client_id=4, comm=0.8, compute=0.9, data=150, loss=-1), # No contribution yet
        MockClient(client_id=5, comm=0.6, compute=0.7, data=300, loss=1.2), # High data
    ]
    print(f"Created {len(clients)} mock clients.")

    # 2. Define DCS weights
    dcs_weights = {'alpha': 0.1, 'beta': 0.2, 'gamma': 0.4, 'delta': 0.3}
    print("\nDCS Weights:", dcs_weights)

    # 3. Compute scores
    client_scores = compute_scores(clients, dcs_weights)
    print("\nComputed client scores:")
    for client, score in client_scores:
        print(f"  - {client}: {score:.4f}")

    # 4. Sort clients by score to see the ranking
    sorted_clients = sorted(client_scores, key=lambda x: x[1], reverse=True)
    print("\nClient ranking based on score:")
    for client, score in sorted_clients:
        print(f"  - {client} (Score: {score:.4f})")

    # Verification
    assert len(client_scores) == len(clients)
    # Check that the highest score went to client 3, who has the best loss
    assert sorted_clients[0][0].client_id == 3
    print("\nDCS functionality verified.")
