"""
Federated learning nodes: clients, UAVs, satellite aggregator
"""
import asyncio
import random
import time
import json
import base64
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .model import SimpleCNN
from .network import Link
from .aggregation import fedavg
from ..utils.model_utils import bytes_to_state_dict, get_state_dict_bytes
from ..config.default_config import Config


@dataclass
class Packet:
    """Network packet"""
    src: str
    dst: str
    ts: float
    data: bytes
    type: str = "model_update"


@dataclass
class FLClient:
    """Federated learning client"""
    id: str
    uav_id: str
    link: Link
    train_loader: DataLoader
    
    # DCS state information (for simulation, normalized to 0~1)
    comm_quality: float = field(default_factory=lambda: random.random())
    comp_capability: float = field(default_factory=lambda: random.random())
    data_significance: float = field(default_factory=lambda: random.random())
    contribution: float = field(default_factory=lambda: random.random())
    
    # Cluster assignment (managed by UAV)
    cluster_id: Optional[int] = None
    
    def calculate_dcs_score(self) -> float:
        """
        Calculate DCS (Dynamic Client Selection) score
        
        Si = α*qi + β*ci + γ*di + δ*gi
        
        Returns:
            DCS score
        """
        return (Config.ALPHA * self.comm_quality +
                Config.BETA * self.comp_capability +
                Config.GAMMA * self.data_significance +
                Config.DELTA * self.contribution)
    
    def train(self, global_model: nn.Module) -> dict:
        """
        Perform local training
        
        Args:
            global_model: Global model
        
        Returns:
            Trained model's state_dict
        """
        model = SimpleCNN(62).to(Config.DEVICE)
        model.load_state_dict(global_model.state_dict())
        
        opt = torch.optim.SGD(
            model.parameters(), 
            lr=Config.LR, 
            momentum=Config.MOMENTUM
        )
        
        model.train()
        initial_loss = 0.0
        final_loss = 0.0
        
        # Perform local training for LOCAL_EPOCHS
        for epoch in range(Config.LOCAL_EPOCHS):
            for i, (x, y) in enumerate(self.train_loader):
                x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
                opt.zero_grad()
                out = model(x)
                loss = F.cross_entropy(out, y)
                loss.backward()
                opt.step()
                
                # Record loss from first batch of first epoch and last batch of last epoch
                if epoch == 0 and i == 0:
                    initial_loss = loss.item()
                if epoch == Config.LOCAL_EPOCHS - 1:
                    final_loss = loss.item()
        
        # Update contribution (gi) based on loss reduction
        if initial_loss > 0:
            self.contribution = max(0.0, min(1.0, 
                (initial_loss - final_loss) / (initial_loss + 1e-9)))
        
        return model.state_dict()


@dataclass
class UAV:
    """UAV (Regional aggregator) with local cluster-based aggregation"""
    id: str
    link: Link
    in_q: asyncio.Queue
    out_q: asyncio.Queue
    assigned_clients: List[FLClient] = field(default_factory=list)
    
    # Client to cluster mapping (managed by this UAV)
    client_to_cluster: Dict[str, int] = field(default_factory=dict)
    
    # Round buffer: cluster_id -> [(client_id, state_dict, data_weight), ...]
    cluster_buffers: Dict[int, List[Tuple[str, Dict, float]]] = field(default_factory=lambda: defaultdict(list))
    
    def select_clients_dcs(self, sample_ratio: float) -> List[FLClient]:
        """
        Select clients using DCS
        
        Args:
            sample_ratio: Selection ratio
        
        Returns:
            List of selected clients
        """
        num_to_select = max(1, int(len(self.assigned_clients) * sample_ratio))
        
        # Calculate latest score for each client
        client_scores = [
            (c, c.calculate_dcs_score()) 
            for c in self.assigned_clients
        ]
        
        # Sort by score in descending order
        client_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top m clients
        selected = [c for c, score in client_scores[:num_to_select]]
        return selected
    
    def update_client_cluster_mapping(self, mapping: Dict[str, int]):
        """
        Update client-to-cluster mapping from satellite
        
        Args:
            mapping: {client_id: cluster_id} mapping
        """
        self.client_to_cluster.update(mapping)
        # Also update client objects
        for client in self.assigned_clients:
            if client.id in mapping:
                client.cluster_id = mapping[client.id]
    
    def reset_round_buffer(self):
        """Reset cluster buffers for new round"""
        self.cluster_buffers.clear()
    
    def add_client_update(self, client_id: str, state_dict: Dict, data_weight: float):
        """
        Add client update to appropriate cluster buffer
        
        Args:
            client_id: Client identifier
            state_dict: Model state_dict
            data_weight: Data size weight for aggregation
        """
        # Get cluster_id for this client (default to 0 if not assigned yet)
        cluster_id = self.client_to_cluster.get(client_id, 0)
        self.cluster_buffers[cluster_id].append((client_id, state_dict, data_weight))
    
    def aggregate_cluster_models(self) -> Dict[int, Dict]:
        """
        Aggregate models per cluster using FedAvg
        
        Returns:
            Dictionary {cluster_id: aggregated_state_dict}
        """
        cluster_models = {}
        for cluster_id, updates in self.cluster_buffers.items():
            if not updates:
                continue
            
            state_dicts = [sd for _, sd, _ in updates]
            weights = [w for _, _, w in updates]
            cluster_models[cluster_id] = fedavg(state_dicts, weights)
        
        return cluster_models
    
    async def run(self):
        """UAV run loop (placeholder - packets are processed by experiment script)"""
        # Note: Packets are processed directly by experiment scripts via uav.in_q.get_nowait()
        # This method is kept as a background task placeholder but doesn't consume packets
        while True:
            await asyncio.sleep(1.0)  # Just keep the task alive
    
    async def send_cluster_models_to_satellite(self, cluster_models: Dict[int, Dict], 
                                               client_id_to_data_size: Dict[str, int],
                                               cost_meter=None):
        """
        Send cluster models to satellite
        
        Args:
            cluster_models: {cluster_id: state_dict} dictionary
            client_id_to_data_size: {client_id: data_size} for metadata
            cost_meter: CostMeter instance for tracking
        """
        # Build payload with cluster models and metadata
        payload_data = {
            "uav_id": self.id,
            "clusters": []
        }
        
        for cluster_id, state_dict in cluster_models.items():
            # Get client IDs in this cluster
            client_ids = [cid for cid, _, _ in self.cluster_buffers.get(cluster_id, [])]
            data_weight = sum(w for _, _, w in self.cluster_buffers.get(cluster_id, []))
            
            # Encode state_dict bytes as base64 for JSON serialization
            state_dict_bytes = get_state_dict_bytes(state_dict)
            state_dict_b64 = base64.b64encode(state_dict_bytes).decode('utf-8')
            
            cluster_info = {
                "cluster_id": cluster_id,
                "state_dict_b64": state_dict_b64,
                "data_weight": data_weight,
                "client_ids": client_ids
            }
            payload_data["clusters"].append(cluster_info)
        
        # Serialize payload
        payload_json = json.dumps(payload_data, default=str)
        payload_bytes = payload_json.encode('utf-8')
        
        # Transmit to satellite
        if await self.link.transmit(payload_bytes, cost_meter=cost_meter):
            # Create packet for satellite
            packet = Packet(
                src=self.id,
                dst="satellite",
                ts=time.time(),
                data=payload_bytes,
                type="uav_cluster_models"
            )
            await self.out_q.put(packet)


@dataclass
class SatAgg:
    """Satellite aggregator with global cluster management"""
    in_q: asyncio.Queue
    buffer: List = field(default_factory=list)  # List of UAV cluster model payloads
    
    # Global cluster models: {cluster_id: state_dict}
    cluster_models: Dict[int, Dict] = field(default_factory=dict)
    
    # Global client-to-cluster mapping: {client_id: cluster_id}
    client_to_cluster: Dict[str, int] = field(default_factory=dict)
    
    # Number of clusters (fixed after round 0)
    num_clusters: Optional[int] = None
    
    async def run(self):
        """Satellite run loop (collect UAV cluster model updates)"""
        while True:
            pkt = await self.in_q.get()
            if isinstance(pkt, Packet) and pkt.type == "uav_cluster_models":
                # Parse UAV cluster models payload
                try:
                    payload_json = pkt.data.decode('utf-8')
                    payload_data = json.loads(payload_json)
                    # Decode base64 state_dicts back to bytes
                    for cluster_info in payload_data.get("clusters", []):
                        if "state_dict_b64" in cluster_info:
                            state_dict_bytes = base64.b64decode(cluster_info["state_dict_b64"].encode('utf-8'))
                            cluster_info["state_dict"] = bytes_to_state_dict(state_dict_bytes)
                            del cluster_info["state_dict_b64"]
                    self.buffer.append(payload_data)
                except Exception as e:
                    print(f"Error parsing UAV payload: {e}")
            else:
                # Backward compatibility: old format
                if isinstance(pkt, Packet):
                    client_id = pkt.src
                    state_dict = bytes_to_state_dict(pkt.data)
                    self.buffer.append((client_id, state_dict))
                else:
                    state_dict = bytes_to_state_dict(pkt)
                    self.buffer.append((None, state_dict))

