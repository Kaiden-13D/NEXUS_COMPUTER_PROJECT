"""
HPFL Core Modules
"""
from .data import FEMNISTDataset, setup_femnist_by_writer
from .model import SimpleCNN
from .network import Link, CostMeter
from .fl_nodes import Packet, FLClient, UAV, SatAgg
from .aggregation import fedavg
from .clustering import (
    compute_similarity_matrix,
    cluster_models,
    cluster_based_aggregation
)

__all__ = [
    'FEMNISTDataset', 'setup_femnist_by_writer',
    'SimpleCNN',
    'Link', 'CostMeter',
    'Packet', 'FLClient', 'UAV', 'SatAgg',
    'fedavg',
    'compute_similarity_matrix',
    'cluster_models',
    'cluster_based_aggregation'
]

