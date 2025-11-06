"""
Satellite representation for the federated learning simulator.

This module defines the Satellite class, which is responsible for global operations.
The satellite performs global clustering of UAV models, adapts the number of clusters,
and aggregates models within each cluster.
"""

from collections import OrderedDict
import numpy as np
from clustering import model_to_vector, compute_similarity_matrix, cluster_assignment

class Satellite:
    """Represents the satellite responsible for global clustering and aggregation."""
    def __init__(self, initial_clusters_k):
        self.initial_clusters_k = initial_clusters_k
        self.uav_models = [] # List of model state_dicts from UAVs

    def cluster_models(self, uav_models):
        """Clusters UAV models based on parameter similarity.

        Args:
            uav_models (list): A list of model state_dicts from the UAVs.

        Returns:
            A dictionary mapping cluster labels to lists of UAV models.
        """
        self.uav_models = uav_models
        if not self.uav_models:
            return {}

        # Convert models to vectors
        model_vectors = [model_to_vector(model) for model in self.uav_models]
        
        # Compute similarity matrix
        sim_matrix = compute_similarity_matrix(model_vectors)
        
        # Get cluster assignments
        cluster_labels = cluster_assignment(sim_matrix, self.initial_clusters_k)
        
        # Group models by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(self.uav_models[i])
            
        return clusters

    def aggregate_clusters(self, clusters):
        """Aggregates models within each cluster to create global cluster models.

        Args:
            clusters (dict): A dictionary mapping cluster labels to lists of models.

        Returns:
            A dictionary mapping cluster labels to the aggregated global model state_dict.
        """
        aggregated_cluster_models = {}
        for label, models in clusters.items():
            if not models:
                continue

            # Initialize with the first model's structure
            aggregated_model = OrderedDict()
            for key in models[0].keys():
                aggregated_model[key] = 0.0

            # Simple averaging of model parameters
            num_models = len(models)
            for model in models:
                for key in model:
                    aggregated_model[key] += model[key] / num_models
            
            aggregated_cluster_models[label] = aggregated_model
            
        return aggregated_cluster_models

    def broadcast_global_model(self, cluster_models):
        """Distributes cluster-specific global models back to UAVs.
        
        (This is a conceptual step; in the simulation, the trainer will handle this.)
        """
        pass
