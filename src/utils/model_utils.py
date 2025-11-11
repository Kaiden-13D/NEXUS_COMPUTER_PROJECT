"""
모델 관련 유틸리티 함수
"""
import io
import torch
from typing import Dict


def get_state_dict_bytes(state_dict: Dict) -> bytes:
    """
    모델 state_dict를 bytes로 변환
    
    Args:
        state_dict: PyTorch 모델의 state_dict
    
    Returns:
        직렬화된 bytes
    """
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    return buf.getvalue()


def bytes_to_state_dict(data: bytes) -> Dict:
    """
    bytes를 모델 state_dict로 변환
    
    Args:
        data: 직렬화된 bytes
    
    Returns:
        PyTorch 모델의 state_dict
    """
    return torch.load(io.BytesIO(data), map_location="cpu")

