"""
모델 유사도 기반 클러스터링
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
    모델 state_dict에서 특징 벡터 추출 (경량화)
    
    Args:
        state_dict: PyTorch 모델의 state_dict
    
    Returns:
        평탄화된 특징 벡터
    """
    # 모든 파라미터를 평탄화하여 연결
    features = []
    for key, param in state_dict.items():
        if 'weight' in key:  # weight만 사용 (경량화)
            features.append(param.flatten())
    
    return torch.cat(features)


def compute_cosine_similarity(vec1: torch.Tensor, vec2: torch.Tensor) -> float:
    """
    코사인 유사도 계산
    
    Args:
        vec1, vec2: 특징 벡터
    
    Returns:
        코사인 유사도 (0~1)
    """
    vec1_norm = F.normalize(vec1.unsqueeze(0), p=2, dim=1)
    vec2_norm = F.normalize(vec2.unsqueeze(0), p=2, dim=1)
    similarity = torch.mm(vec1_norm, vec2_norm.t()).item()
    return max(0.0, min(1.0, (similarity + 1) / 2))  # -1~1 -> 0~1로 정규화


def compute_similarity_matrix(
    state_dicts_or_tuples: List[Dict | tuple]
) -> np.ndarray:
    """
    모델 간 유사도 행렬 계산
    
    Args:
        state_dicts_or_tuples: 모델 state_dict 리스트 또는 (client_id, state_dict) 튜플 리스트
    
    Returns:
        유사도 행렬 (n x n)
    """
    # 튜플인 경우 state_dict만 추출
    if state_dicts_or_tuples and isinstance(state_dicts_or_tuples[0], tuple):
        state_dicts = [sd for _, sd in state_dicts_or_tuples]
    else:
        state_dicts = state_dicts_or_tuples
    
    n = len(state_dicts)
    
    logger = get_progress_logger()
    if logger:
        logger.log(f"Extracting features from {n} models...", print_to_console=False)
    
    # 특징 추출 진행 상황 표시 (progress 로그에만 기록)
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
    
    # 차원 축소 (PCA) - 경량화
    if len(features[0]) > 1000:
        features_array = torch.stack(features).numpy()
        n_samples = features_array.shape[0]
        n_features = features_array.shape[1]
        # n_components는 샘플 수와 특징 수 중 작은 값보다 작아야 함
        max_components = min(100, n_features, max(1, n_samples - 1))
        if max_components > 0:
            pca = PCA(n_components=max_components)
            features_reduced = pca.fit_transform(features_array)
            features = [torch.from_numpy(f) for f in features_reduced]
    
    # 유사도 행렬 계산
    if logger:
        logger.log(f"Computing similarity matrix ({n}x{n})...", print_to_console=False)
    
    similarity_matrix = np.zeros((n, n))
    
    # 진행 상황 표시 (progress 로그에만 기록)
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
    모델 유사도 기반 클러스터링
    
    Args:
        state_dicts_or_tuples: 모델 state_dict 리스트 또는 (client_id, state_dict) 튜플 리스트
        num_clusters: 클러스터 수 (None이면 자동 결정)
        similarity_matrix: 미리 계산된 유사도 행렬 (None이면 계산)
    
    Returns:
        각 모델의 클러스터 할당 (리스트)
    """
    # 튜플인 경우 state_dict만 추출
    if state_dicts_or_tuples and isinstance(state_dicts_or_tuples[0], tuple):
        state_dicts = [sd for _, sd in state_dicts_or_tuples]
    else:
        state_dicts = state_dicts_or_tuples
    
    if similarity_matrix is None:
        similarity_matrix = compute_similarity_matrix(state_dicts)
    
    n = len(state_dicts)
    
    # 클러스터 수 자동 결정 (간단한 휴리스틱)
    if num_clusters is None:
        # 유사도 행렬의 평균을 기반으로 클러스터 수 결정
        avg_similarity = np.mean(similarity_matrix[np.triu_indices(n, k=1)])
        num_clusters = max(2, min(int(np.sqrt(n)), int(n * (1 - avg_similarity) * 2)))
    
    # 거리 행렬로 변환 (1 - similarity)
    distance_matrix = 1 - similarity_matrix
    
    # K-means 클러스터링 (거리 행렬 기반)
    # 간단한 구현: 유사도 기반 그룹핑
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
    클러스터별 모델 집계
    
    Args:
        state_dicts: 모델 state_dict 리스트
        cluster_labels: 각 모델의 클러스터 할당
        weights: 각 모델의 가중치 리스트 (데이터 크기 등, None이면 균등 가중치)
    
    Returns:
        {cluster_id: aggregated_state_dict} 딕셔너리
    """
    from .aggregation import fedavg
    
    # 클러스터별로 그룹화 (state_dict와 가중치 함께)
    cluster_groups = defaultdict(lambda: {'state_dicts': [], 'weights': []})
    for idx, cluster_id in enumerate(cluster_labels):
        cluster_groups[cluster_id]['state_dicts'].append(state_dicts[idx])
        if weights:
            cluster_groups[cluster_id]['weights'].append(weights[idx])
    
    # 각 클러스터별로 가중 FedAvg 수행
    cluster_models = {}
    for cluster_id, group_data in cluster_groups.items():
        group_state_dicts = group_data['state_dicts']
        group_weights = group_data['weights'] if weights else None
        cluster_models[cluster_id] = fedavg(group_state_dicts, group_weights)
    
    return cluster_models

