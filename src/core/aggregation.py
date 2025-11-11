"""
연합학습 집계 함수
"""
from typing import List, Dict, Optional
import torch


def fedavg(
    state_dicts: List[Dict], 
    weights: Optional[List[float]] = None
) -> Dict:
    """
    Federated Averaging (FedAvg) 집계
    
    Args:
        state_dicts: 클라이언트들의 모델 state_dict 리스트
        weights: 각 클라이언트의 가중치 리스트 (데이터 크기 등)
                 None이면 균등 가중치 (1/n)
    
    Returns:
        가중 평균화된 state_dict
    """
    if not state_dicts:
        raise ValueError("Empty state_dicts list")
    
    n = len(state_dicts)
    
    # 가중치가 없으면 균등 가중치
    if weights is None:
        weights = [1.0 / n] * n
    else:
        if len(weights) != n:
            raise ValueError(f"weights length ({len(weights)}) must match state_dicts length ({n})")
        # 가중치 정규화
        total_weight = sum(weights)
        if total_weight == 0:
            raise ValueError("Total weight cannot be zero")
        weights = [w / total_weight for w in weights]
    
    # 가중 평균 계산
    result = {}
    for k in state_dicts[0]:
        result[k] = sum(
            state_dicts[i][k] * weights[i] 
            for i in range(n)
        )
    
    return result

