"""
연합학습 노드: 클라이언트, UAV, 위성 집계자
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
    """네트워크 패킷"""
    src: str
    dst: str
    ts: float
    data: bytes
    type: str = "model_update"


@dataclass
class FLClient:
    """연합학습 클라이언트"""
    id: str
    uav_id: str
    link: Link
    train_loader: DataLoader
    
    # DCS용 상태 정보 (시뮬레이션용, 0~1 정규화 가정)
    comm_quality: float = field(default_factory=lambda: random.random())
    comp_capability: float = field(default_factory=lambda: random.random())
    data_significance: float = field(default_factory=lambda: random.random())
    contribution: float = field(default_factory=lambda: random.random())
    
    def calculate_dcs_score(self) -> float:
        """
        DCS (Dynamic Client Selection) 점수 계산
        
        Si = α*qi + β*ci + γ*di + δ*gi
        
        Returns:
            DCS 점수
        """
        return (Config.ALPHA * self.comm_quality +
                Config.BETA * self.comp_capability +
                Config.GAMMA * self.data_significance +
                Config.DELTA * self.contribution)
    
    def train(self, global_model: nn.Module) -> dict:
        """
        로컬 학습 수행
        
        Args:
            global_model: 전역 모델
        
        Returns:
            학습된 모델의 state_dict
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
        
        # LOCAL_EPOCHS만큼 로컬 학습 수행
        for epoch in range(Config.LOCAL_EPOCHS):
            for i, (x, y) in enumerate(self.train_loader):
                x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
                opt.zero_grad()
                out = model(x)
                loss = F.cross_entropy(out, y)
                loss.backward()
                opt.step()
                
                # 첫 에폭의 첫 배치와 마지막 에폭의 마지막 배치에서 loss 기록
                if epoch == 0 and i == 0:
                    initial_loss = loss.item()
                if epoch == Config.LOCAL_EPOCHS - 1:
                    final_loss = loss.item()
        
        # 기여도(gi) 업데이트: 손실 감소량 기반
        if initial_loss > 0:
            self.contribution = max(0.0, min(1.0, 
                (initial_loss - final_loss) / (initial_loss + 1e-9)))
        
        return model.state_dict()


@dataclass
class UAV:
    """UAV (구역별 집계자)"""
    id: str
    link: Link
    in_q: asyncio.Queue
    out_q: asyncio.Queue
    assigned_clients: List[FLClient] = field(default_factory=list)
    
    def select_clients_dcs(self, sample_ratio: float) -> List[FLClient]:
        """
        DCS를 사용한 클라이언트 선택
        
        Args:
            sample_ratio: 선택 비율
        
        Returns:
            선택된 클라이언트 리스트
        """
        num_to_select = max(1, int(len(self.assigned_clients) * sample_ratio))
        
        # 각 클라이언트의 최신 점수 계산
        client_scores = [
            (c, c.calculate_dcs_score()) 
            for c in self.assigned_clients
        ]
        
        # 점수 내림차순 정렬
        client_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 상위 m개 선택
        selected = [c for c, score in client_scores[:num_to_select]]
        return selected
    
    async def run(self):
        """UAV 실행 루프 (패킷 배치 처리)"""
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
        """배치 패킷을 위성으로 전송"""
        payload = b"".join(p.data for p in batch)
        if await self.link.transmit(payload):
            # Packet 객체 전체를 전송 (client_id 포함)
            for pkt in batch:
                await self.out_q.put(pkt)


@dataclass
class SatAgg:
    """위성 집계자 (Satellite Aggregator)"""
    in_q: asyncio.Queue
    buffer: List = field(default_factory=list)  # (client_id, state_dict) 튜플 리스트
    
    async def run(self):
        """위성 실행 루프 (모델 업데이트 수집)"""
        while True:
            pkt = await self.in_q.get()
            # Packet 객체에서 client_id와 state_dict 추출
            if isinstance(pkt, Packet):
                client_id = pkt.src
                state_dict = bytes_to_state_dict(pkt.data)
                self.buffer.append((client_id, state_dict))
            else:
                # 하위 호환성: bytes만 받는 경우
                state_dict = bytes_to_state_dict(pkt)
                self.buffer.append((None, state_dict))

