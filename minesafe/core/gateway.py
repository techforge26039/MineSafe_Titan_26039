from __future__ import annotations
from typing import Any, Dict, Optional, Tuple, List
import requests
from .telemetry import TelemetryData, sanitize_and_validate
from .config import REQUEST_TIMEOUT

class GatewayStatus:
    def __init__(self, online: bool, message: str, latency_ms: float = 0.0):
        self.online = online
        self.message = message
        self.latency_ms = latency_ms

class GatewayClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def get_telemetry(self, sequence: int) -> Tuple[Optional[TelemetryData], GatewayStatus]:
        url = f"{self.base_url}/api/telemetry"
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                data = resp.json()
                telemetry = sanitize_and_validate(data, sequence, source="LIVE_GATEWAY")
                status = GatewayStatus(online=True, message="OPERATIONAL", latency_ms=resp.elapsed.total_seconds() * 1000.0)
                return telemetry, status
            return None, GatewayStatus(online=False, message=f"HTTP {resp.status_code}")
        except requests.exceptions.Timeout:
            return None, GatewayStatus(online=False, message="TIMEOUT")
        except requests.exceptions.ConnectionError:
            return None, GatewayStatus(online=False, message="CONNECTION_REFUSED")
        except Exception as e:
            return None, GatewayStatus(online=False, message=str(e)[:24])

    def send_rover_command(self, cmd: str) -> bool:
        url = f"{self.base_url}/api/command"
        try:
            resp = requests.post(url, json={"command": cmd}, timeout=1.0)
            return resp.status_code == 200
        except Exception:
            return False

    def get_workers(self) -> Tuple[Optional[List[Dict[str, Any]]], GatewayStatus]:
        url = f"{self.base_url}/api/workers"
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                return resp.json(), GatewayStatus(online=True, message="OK")
            return None, GatewayStatus(online=False, message=f"HTTP {resp.status_code}")
        except Exception:
            return None, GatewayStatus(online=False, message="OFFLINE")