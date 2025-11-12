"""
Model similarity-based clustering
"""
import torch
import torch.nn.functional as F
from typing import List, Dict
from collections import defaultdict
import numpy as np
from sklearn.decomposition import PCA

from ..utils.progress_logger import get_progress_logger


def extract_model_features(state_dict: Dict) -> torch.Tensor:
    """
    Extract feature vector from model state_dict (lightweight)
    
    Args:
        state_dict: PyTorch model's state_dict
    
    Returns:
        Flattened feature vector
    """
    # Flatten and concatenate all parameters
    features = []
    for key, param in state_dict.items():
        if 'weight' in key:  # Use only weights (lightweight)
            features.append(param.flatten())
    
    return torch.cat(features)


def compute_cosine_similarity(vec1: torch.Tensor, vec2: torch.Tensor) -> float:
    """
    Calculate cosine similarity
    
    Args:
        vec1, vec2: Feature vectors
    
    Returns:
        Cosine similarity (0~1)
    """
    vec1_norm = F.normalize(vec1.unsqueeze(0), p=2, dim=1)
    vec2_norm = F.normalize(vec2.unsqueeze(0), p=2, dim=1)
    similarity = torch.mm(vec1_norm, vec2_norm.t()).item()
    return max(0.0, min(1.0, (similarity + 1) / 2))  # Normalize -1~1 to 0~1


def compute_similarity_matrix(
    state_dicts_or_tuples: List[Dict | tuple]
) -> np.ndarray:
    """
    Compute similarity matrix between models
    
    Args:
        state_dicts_or_tuples: List of model state_dicts or (client_id, state_dict) tuples
    
    Returns:
        Similarity matrix (n x n)
    """
    # Extract state_dict only if tuples
    if state_dicts_or_tuples and isinstance(state_dicts_or_tuples[0], tuple):
        state_dicts = [sd for _, sd in state_dicts_or_tuples]
    else:
        state_dicts = state_dicts_or_tuples
    
    n = len(state_dicts)
    
    logger = get_progress_logger()
    if logger:
        logger.log(f"Extracting features from {n} models...", print_to_console=False)
    
    # Show feature extraction progress (logged to progress log only)
    try:
        from tqdm import tqdm
        import sys
        if logger:
            tqdm_file = open(logger.log_file, 'a')
            features = [extract_model_features(sd) for sd in tqdm(state_dicts, desc="Extracting features", file=tqdm_file)]
            tqdm_file.close()
        else:
            features = [extract_model_features(sd) for sd in tqdm(state_dicts, desc="Extracting features")]
    except ImportError:
        features = [extract_model_features(sd) for sd in state_dicts]
    
    # Dimensionality reduction (PCA) - lightweight
    if len(features[0]) > 1000:
        features_array = torch.stack(features).numpy()
        n_samples = features_array.shape[0]
        n_features = features_array.shape[1]
        # n_components must be less than min(n_samples, n_features)
        max_components = min(100, n_features, max(1, n_samples - 1))
        if max_components > 0:
            pca = PCA(n_components=max_components)
            features_reduced = pca.fit_transform(features_array)
            features = [torch.from_numpy(f) for f in features_reduced]
    
    # Compute similarity matrix
    if logger:
        logger.log(f"Computing similarity matrix ({n}x{n})...", print_to_console=False)
    
    similarity_matrix = np.zeros((n, n))
    
    # Show progress (logged to progress log only)
    try:
        from tqdm import tqdm
        import sys
        if logger:
            tqdm_file = open(logger.log_file, 'a')
            outer_iter = tqdm(range(n), desc="Computing similarities", file=tqdm_file)
        else:
            outer_iter = tqdm(range(n), desc="Computing similarities")
    except ImportError:
        outer_iter = range(n)
        tqdm_file = None
    
    try:
        for i in outer_iter:
            for j in range(i, n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    sim = compute_cosine_similarity(features[i], features[j])
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim
    finally:
        if 'tqdm_file' in locals() and tqdm_file:
            tqdm_file.close()
            if logger:
                import sys
                sys.stdout = sys.__stdout__
    
    return similarity_matrix


def cluster_models(
    state_dicts_or_tuples: List[Dict | tuple],
    num_clusters: int = None,
    similarity_matrix: np.ndarray = None
) -> List[int]:
    """
    Cluster models based on similarity
    
    Args:
        state_dicts_or_tuples: List of model state_dicts or (client_id, state_dict) tuples
        num_clusters: Number of clusters (None for auto-determination)
        similarity_matrix: Pre-computed similarity matrix (None to compute)
    
    Returns:
        Cluster assignments for each model (list)
    """
    # Extract state_dict only if tuples
    if state_dicts_or_tuples and isinstance(state_dicts_or_tuples[0], tuple):
        state_dicts = [sd for _, sd in state_dicts_or_tuples]
    else:
        state_dicts = state_dicts_or_tuples
    
    if similarity_matrix is None:
        similarity_matrix = compute_similarity_matrix(state_dicts)
    
    n = len(state_dicts)
    
    # Auto-determine number of clusters (simple heuristic)
    if num_clusters is None:
        # Determine number of clusters based on average similarity
        avg_similarity = np.mean(similarity_matrix[np.triu_indices(n, k=1)])
        num_clusters = max(2, min(int(np.sqrt(n)), int(n * (1 - avg_similarity) * 2)))
    
    # Convert to distance matrix (1 - similarity)
    distance_matrix = 1 - similarity_matrix
    
    # Agglomerative clustering (distance matrix based)
    # Simple implementation: similarity-based grouping
    from sklearn.cluster import AgglomerativeClustering
    
    clustering = AgglomerativeClustering(
        n_clusters=num_clusters,
        metric='precomputed',
        linkage='average'
    )
    cluster_labels = clustering.fit_predict(distance_matrix)
    
    return cluster_labels.tolist()


def cluster_based_aggregation(
    state_dicts: List[Dict],
    cluster_labels: List[int],
    weights: List[float] = None
) -> Dict[int, Dict]:
    """
    Aggregate models by cluster
    
    Args:
        state_dicts: List of model state_dicts
        cluster_labels: Cluster assignment for each model
        weights: List of model weights (e.g., data size, None for uniform)
    
    Returns:
        Dictionary {cluster_id: aggregated_state_dict}
    """
    from .aggregation import fedavg
    
    # Group by cluster (state_dict and weights together)
    cluster_groups = defaultdict(lambda: {'state_dicts': [], 'weights': []})
    for idx, cluster_id in enumerate(cluster_labels):
        cluster_groups[cluster_id]['state_dicts'].append(state_dicts[idx])
        if weights:
            cluster_groups[cluster_id]['weights'].append(weights[idx])
    
    # Perform weighted FedAvg for each cluster
    cluster_models = {}
    for cluster_id, group_data in cluster_groups.items():
        group_state_dicts = group_data['state_dicts']
        group_weights = group_data['weights'] if weights else None
        cluster_models[cluster_id] = fedavg(group_state_dicts, group_weights)
    
    return cluster_models

