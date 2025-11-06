"""
Clustering utilities for model similarity.

This module provides functions for measuring model similarity and performing clustering.
It includes functions to convert models to vectors, compute a similarity matrix,
and assign models to clusters.
"""

import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

def model_to_vector(model_state):
    """Flattens a model's state_dict into a single vector.

    Args:
        model_state (OrderedDict): The model's state_dict.

    Returns:
        A 1D numpy array representing the model's parameters.
    """
    # Concatenate all parameters into a single tensor
    params = [param.cpu().numpy().flatten() for param in model_state.values()]
    return np.concatenate(params)

def compute_similarity_matrix(vectors):
    """Computes the pairwise cosine similarity between a list of vectors.

    Args:
        vectors (list or np.ndarray): A list of 1D numpy arrays.

    Returns:
        A 2D numpy array representing the similarity matrix.
    """
    return cosine_similarity(vectors)

def cluster_assignment(sim_matrix, max_k):
    """Groups models into clusters based on their similarity matrix.

    This implementation uses K-means clustering on the similarity vectors.

    Args:
        sim_matrix (np.ndarray): The similarity matrix.
        max_k (int): The maximum number of clusters to form.

    Returns:
        A numpy array of cluster labels for each model.
    """
    if sim_matrix.shape[0] <= max_k:
        # If there are fewer models than max_k, assign each to its own cluster
        return np.arange(sim_matrix.shape[0])

    kmeans = KMeans(n_clusters=max_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(sim_matrix)
    
    return cluster_labels
