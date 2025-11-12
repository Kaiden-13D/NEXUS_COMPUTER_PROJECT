"""
Federated learning nodes: clients, UAVs, satellite aggregator
"""
import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .model import SimpleCNN
from .network import Link
from ..utils.model_utils import bytes_to_state_dict
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
    """UAV (Regional aggregator)"""
    id: str
    link: Link
    in_q: asyncio.Queue
    out_q: asyncio.Queue
    assigned_clients: List[FLClient] = field(default_factory=list)
    
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
    
    async def run(self):
        """UAV run loop (batch packet processing)"""
        batch = []
        while True:
            try:
                pkt = await asyncio.wait_for(self.in_q.get(), 0.8)
                batch.append(pkt)
                if len(batch) >= 8:
                    await self.flush(batch)
                    batch = []
            except asyncio.TimeoutError:
                if batch:
                    await self.flush(batch)
                    batch = []
    
    async def flush(self, batch: List[Packet]):
        """Send batch packets to satellite"""
        payload = b"".join(p.data for p in batch)
        if await self.link.transmit(payload):
            # Send entire Packet object (including client_id)
            for pkt in batch:
                await self.out_q.put(pkt)


@dataclass
class SatAgg:
    """Satellite aggregator"""
    in_q: asyncio.Queue
    buffer: List = field(default_factory=list)  # List of (client_id, state_dict) tuples
    
    async def run(self):
        """Satellite run loop (collect model updates)"""
        while True:
            pkt = await self.in_q.get()
            # Extract client_id and state_dict from Packet object
            if isinstance(pkt, Packet):
                client_id = pkt.src
                state_dict = bytes_to_state_dict(pkt.data)
                self.buffer.append((client_id, state_dict))
            else:
                # Backward compatibility: bytes only
                state_dict = bytes_to_state_dict(pkt)
                self.buffer.append((None, state_dict))

