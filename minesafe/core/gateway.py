"""
MineSafe Network Gateway Client
-------------------------------
Polls telemetry and worker states from the ESP32-S3-CAM over HTTP/Wi-Fi.
"""
from __future__ import annotations
import requests
from typing import Any, Dict, List, Optional, Tuple

from .telemetry import sanitize_and_validate, TelemetryData

class CommandResult:
    def __init__(self, ok: bool, message: str):
        self.ok = ok
        self.message = message

class GatewayClient:
    """HTTP client communicating with the ESP32-S3-CAM."""
    def __init__(self, base_url: str = "http://192.168.137.89", timeout: float = 1.5):
        # Normalize base URL (strip trailing slashes)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.online = False
        self.message = "Gateway disconnected"

    def get_telemetry(self, sequence: int = 0) -> Tuple[Optional[TelemetryData], GatewayClient]:
        endpoint = f"{self.base_url}/api/telemetry"
        try:
            resp = requests.get(endpoint, timeout=self.timeout)
            if resp.status_code == 200:
                payload = resp.json()
                if isinstance(payload, dict):
                    self.online = True
                    self.message = "ESP32-S3 Gateway Online"
                    # Pass through canonical validation pipeline
                    telemetry = sanitize_and_validate(payload, sequence, "LIVE · ESP32-S3")
                    return telemetry, self
            
            self.online = False
            self.message = f"HTTP Error {resp.status_code}"
            return None, self

        except Exception as exc:
            self.online = False
            self.message = f"FAIL-SAFE · {type(exc).__name__}"
            return None, self

    def get_workers(self) -> Tuple[Optional[List[Dict[str, Any]]], GatewayClient]:
        endpoint = f"{self.base_url}/api/workers"
        try:
            resp = requests.get(endpoint, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                workers = data if isinstance(data, list) else data.get("workers", [])
                return workers, self
            return None, self
        except Exception:
            return None, self

    def send_rover_command(self, cmd: str) -> CommandResult:
        endpoint = f"{self.base_url}/api/command"
        try:
            resp = requests.post(endpoint, json={"command": cmd}, timeout=self.timeout)
            if resp.status_code == 200:
                ack = resp.json().get("ack", "OK")
                return CommandResult(True, f"ACK: {ack}")
            return CommandResult(False, f"HTTP {resp.status_code}")
        except Exception as exc:
            return CommandResult(False, f"Network Error: {type(exc).__name__}")