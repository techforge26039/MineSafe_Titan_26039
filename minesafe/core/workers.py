from __future__ import annotations
from dataclasses import dataclass
from typing import List
from .hazard import calculate_trpi
from .telemetry import TelemetryData

@dataclass
class Worker:
    tag: str
    name: str
    zone: str
    hr_bpm: float
    spo2_pct: float
    temp_c: float
    fall_detected: bool = False
    source: str = "SIMULATED"

    def triage(self, t: TelemetryData, route_blocked: bool):
        score, _ = calculate_trpi(t, self.hr_bpm, self.spo2_pct, route_blocked, self.fall_detected)
        return score


def demo_workers() -> List[Worker]:
    return [
        Worker("TAG-108", "Worker A", "ZONE_A", 128, 94, 37.8, False),
        Worker("TAG-114", "Worker B", "ZONE_B", 118, 95, 37.4, False),
        Worker("TAG-121", "Worker C", "ZONE_B", 88, 97, 36.8, False),
    ]