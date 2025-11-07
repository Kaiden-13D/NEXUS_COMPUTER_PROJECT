
import numpy as np
from collections import OrderedDict
import torch

# We will need clustering utilities, which we assume will be in clustering.py
from clustering import model_to_vector, cluster_assignment
from models import CNNBackbone, get_model_parameters

class Satellite:
    """Represents the satellite responsible for global clustering and aggregation."""
    def __init__(self, num_clusters_k):
        self.num_clusters_k = num_clusters_k
        self.cluster_models = {} # {cluster_id: model_state_dict}

    def cluster_and_aggregate(self, uav_models):
        """Performs model clustering and aggregation for each cluster.

        Args:
            uav_models (dict): A dictionary mapping uav_id to the UAV's model state_dict.

        Returns:
            dict: A dictionary mapping cluster_id to the aggregated cluster model state_dict.
        """
        if not uav_models:
            return {}

        uav_ids = list(uav_models.keys())
        model_states = list(uav_models.values())

        # 1. Convert model states to vectors for clustering
        model_vectors = [model_to_vector(state) for state in model_states]

        # 2. Assign models to clusters
        # The clustering function should return a dictionary {cluster_id: [uav_indices]}
        cluster_assignments = cluster_assignment(model_vectors, self.num_clusters_k)
        
        # 3. Aggregate models within each cluster
        new_cluster_models = {}
        for cluster_id, uav_indices in cluster_assignments.items():
            if not uav_indices:
                continue
            
            cluster_uav_states = [model_states[i] for i in uav_indices]
            
            # Simple FedAvg aggregation within the cluster
            aggregated_state_dict = self._federated_averaging(cluster_uav_states)
            new_cluster_models[cluster_id] = aggregated_state_dict

        self.cluster_models = new_cluster_models
        return self.cluster_models, cluster_assignments

    def _federated_averaging(self, state_dicts):
        """Averages the parameters of multiple model state_dicts."""
        if not state_dicts:
            return None

        # Get the keys from the first model
        keys = state_dicts[0].keys()
        num_models = len(state_dicts)
        
        # Initialize a new state_dict to store the average
        avg_state_dict = OrderedDict()

        for key in keys:
            # Sum the tensors for the current key from all models
            summed_tensor = torch.stack([sd[key] for sd in state_dicts]).sum(0)
            avg_state_dict[key] = summed_tensor / num_models
            
        return avg_state_dict

if __name__ == '__main__':
    # This is a placeholder for example usage and basic testing.
    # We need dummy UAV models and clustering functions.

    # Mock clustering functions for testing purposes
    def model_to_vector(model_state):
        # Flatten all parameters into a single vector
        return np.concatenate([p.numpy().flatten() for p in model_state.values()])

    def cluster_assignment(vectors, k):
        # Simple mock clustering: assign to clusters based on vector norm (not a real algorithm)
        norms = [np.linalg.norm(v) for v in vectors]
        assignments = {i: [] for i in range(k)}
        for i, norm in enumerate(norms):
            cluster_id = int(norm % k) # Simple deterministic assignment for testing
            assignments[cluster_id].append(i)
        return assignments

    # 1. Create dummy UAV models (state_dicts)
    num_uavs = 6
    uav_models = {}
    for i in range(num_uavs):
        model = CNNBackbone()
        # Slightly perturb weights to simulate different models
        for param in model.parameters():
            param.data += torch.randn_like(param.data) * 0.1 * i
        uav_models[f'uav_{i}'] = model.state_dict()

    print(f"Created {len(uav_models)} dummy UAV models.")

    # 2. Initialize the Satellite
    num_clusters = 3
    satellite = Satellite(num_clusters_k=num_clusters)
    print(f"Initialized Satellite with k={num_clusters}.")

    # 3. Perform clustering and aggregation
    print("\nPerforming satellite-level clustering and aggregation...")
    cluster_models, assignments = satellite.cluster_and_aggregate(uav_models)

    print("Clustering and aggregation complete.")
    print(f"Created {len(cluster_models)} cluster models.")
    print("Cluster assignments (UAV index -> Cluster ID):")
    # Reverse the assignment dict for easier reading
    uav_to_cluster = {}
    for cluster_id, uav_indices in assignments.items():
        for uav_idx in uav_indices:
            uav_to_cluster[list(uav_models.keys())[uav_idx]] = cluster_id
    print(uav_to_cluster)

    # 4. Verify the output
    assert len(cluster_models) <= num_clusters
    if cluster_models:
        first_cluster_id = list(cluster_models.keys())[0]
        first_cluster_model = cluster_models[first_cluster_id]
        # Check that the aggregated model has the correct structure
        assert first_cluster_model.keys() == uav_models['uav_0'].keys()
        print("\nOutput format verified.")

    print("\nSatellite functionality test passed.")
