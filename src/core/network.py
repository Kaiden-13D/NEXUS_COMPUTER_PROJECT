"""
Network simulation: links and cost measurement
"""
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional


@dataclass
class Link:
    """Network link simulation"""
    name: str
    latency_ms: int
    jitter_ms: int
    bandwidth_bps: int
    loss: float  # Packet loss rate (0.0 ~ 1.0)
    
    async def transmit(self, packet: bytes, cost_meter=None) -> Optional[bytes]:
        """
        Simulate packet transmission
        
        Args:
            packet: Packet to transmit (bytes)
            cost_meter: Cost meter instance (CostMeter)
        
        Returns:
            packet on success, None on failure
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
    """Communication and time cost measurement"""
    
    def __init__(self):
        self.reset_all()
        self.total_cum_bytes = 0
        self.total_cum_time = 0.0  # Simulated logical time
    
    def reset_all(self):
        """Reset round-by-round statistics"""
        self._round = defaultdict(lambda: {"bytes": 0, "max_delay": 0.0})
    
    def note(self, link_name: str, ok: bool, nbytes: int, delay_s: float):
        """
        Record link usage
        
        Args:
            link_name: Link name
            ok: Transmission success status
            nbytes: Number of bytes transmitted
            delay_s: Delay time (seconds)
        """
        if ok:
            self._round[link_name]["bytes"] += nbytes
            self._round[link_name]["max_delay"] = max(
                self._round[link_name]["max_delay"], delay_s
            )
    
    def end_round(self):
        """
        End round and finalize costs
        
        Returns:
            (round_bytes, round_max_delay)
        """
        round_bytes = sum(d["bytes"] for d in self._round.values())
        self.total_cum_bytes += round_bytes
        
        # Round time = sum of max delays from all links
        round_max_delay = sum(d["max_delay"] for d in self._round.values())
        self.total_cum_time += round_max_delay
        
        self.reset_all()
        return round_bytes, round_max_delay

