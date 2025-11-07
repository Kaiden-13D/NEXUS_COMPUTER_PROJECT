
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

def model_to_vector(model_state):
    """Converts a model's state_dict to a flattened numpy vector."""
    # Ensure tensors are on the CPU and converted to numpy
    return np.concatenate([p.cpu().numpy().flatten() for p in model_state.values()])

def compute_similarity_matrix(vectors):
    """Computes the pairwise cosine similarity matrix for a list of vectors."""
    return cosine_similarity(vectors)

def cluster_assignment(model_vectors, num_clusters):
    """Assigns models to clusters using K-Means.

    Args:
        model_vectors (list of np.ndarray): The list of model vectors.
        num_clusters (int): The number of clusters to create.

    Returns:
        dict: A dictionary mapping cluster_id to a list of model indices in that cluster.
    """
    if not model_vectors or num_clusters <= 0:
        return {}

    # Stack vectors into a matrix for K-Means
    X = np.array(model_vectors)
    
    # If the number of samples is less than clusters, we can't form k clusters.
    # Assign each sample to its own cluster.
    if X.shape[0] < num_clusters:
        assignments = {i: [i] for i in range(X.shape[0])}
        return assignments

    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    # Group indices by cluster label
    assignments = {i: [] for i in range(num_clusters)}
    for i, label in enumerate(labels):
        assignments[label].append(i)
        
    return assignments

if __name__ == '__main__':
    # Example usage and basic testing

    # 1. Create dummy model vectors
    num_models = 10
    vector_dim = 128
    # Create 3 distinct groups of vectors for clear clustering
    model_vectors = []
    for i in range(num_models):
        if i < 3: # Group 1
            model_vectors.append(np.random.rand(vector_dim) * 1 + 0)
        elif i < 6: # Group 2
            model_vectors.append(np.random.rand(vector_dim) * 1 + 5)
        else: # Group 3
            model_vectors.append(np.random.rand(vector_dim) * 1 + 10)

    print(f"Created {len(model_vectors)} dummy model vectors.")

    # 2. Compute similarity matrix (optional, but good to show)
    sim_matrix = compute_similarity_matrix(model_vectors)
    print("\nComputed similarity matrix (shape):", sim_matrix.shape)
    assert sim_matrix.shape == (num_models, num_models)

    # 3. Perform clustering
    num_clusters = 3
    assignments = cluster_assignment(model_vectors, num_clusters)
    print(f"\nPerformed K-Means clustering with k={num_clusters}.")
    print("Cluster assignments (Cluster ID -> Model Indices):")
    print(assignments)

    # Verify the output
    assert len(assignments) == num_clusters
    total_assigned = sum(len(indices) for indices in assignments.values())
    assert total_assigned == num_models
    print("\nClustering assignment verified.")

    # Test edge case: fewer models than clusters
    print("\nTesting edge case: num_models < num_clusters")
    few_model_vectors = model_vectors[:2]
    assignments_edge = cluster_assignment(few_model_vectors, num_clusters)
    print("Cluster assignments for edge case:")
    print(assignments_edge)
    assert len(assignments_edge) == len(few_model_vectors)
    print("Edge case test passed.")
