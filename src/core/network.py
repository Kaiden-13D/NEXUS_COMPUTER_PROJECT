"""
네트워크 시뮬레이션: 링크 및 비용 측정
"""
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional


@dataclass
class Link:
    """네트워크 링크 시뮬레이션"""
    name: str
    latency_ms: int
    jitter_ms: int
    bandwidth_bps: int
    loss: float  # 패킷 손실률 (0.0 ~ 1.0)
    
    async def transmit(self, packet: bytes, cost_meter=None) -> Optional[bytes]:
        """
        패킷 전송 시뮬레이션
        
        Args:
            packet: 전송할 패킷 (bytes)
            cost_meter: 비용 측정기 (CostMeter 인스턴스)
        
        Returns:
            성공 시 packet, 실패 시 None
        """
        delay = (self.latency_ms + random.randint(0, self.jitter_ms)) / 1000 + \
                len(packet) / max(1, self.bandwidth_bps)
        
        if random.random() < self.loss:
            if cost_meter:
                cost_meter.note(self.name, False, 0, delay)
            return None
        
        if cost_meter:
            cost_meter.note(self.name, True, len(packet), delay)
        return packet


class CostMeter:
    """통신 비용 및 시간 비용 측정"""
    
    def __init__(self):
        self.reset_all()
        self.total_cum_bytes = 0
        self.total_cum_time = 0.0  # 시뮬레이션된 논리적 시간
    
    def reset_all(self):
        """라운드별 통계 초기화"""
        self._round = defaultdict(lambda: {"bytes": 0, "max_delay": 0.0})
    
    def note(self, link_name: str, ok: bool, nbytes: int, delay_s: float):
        """
        링크 사용 기록
        
        Args:
            link_name: 링크 이름
            ok: 전송 성공 여부
            nbytes: 전송된 바이트 수
            delay_s: 지연 시간 (초)
        """
        if ok:
            self._round[link_name]["bytes"] += nbytes
            self._round[link_name]["max_delay"] = max(
                self._round[link_name]["max_delay"], delay_s
            )
    
    def end_round(self):
        """
        라운드 종료 및 비용 정산
        
        Returns:
            (round_bytes, round_max_delay)
        """
        round_bytes = sum(d["bytes"] for d in self._round.values())
        self.total_cum_bytes += round_bytes
        
        # 이번 라운드의 소요 시간 = 모든 링크에서 발생한 최대 지연 시간의 합
        round_max_delay = sum(d["max_delay"] for d in self._round.values())
        self.total_cum_time += round_max_delay
        
        self.reset_all()
        return round_bytes, round_max_delay

