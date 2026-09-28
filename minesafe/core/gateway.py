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
        url = f"{self.base_url}/data"
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200:
                data = resp.json()

                # Process vibration input (handles boolean or integer)
                vibe_raw = data.get("vibration", False)
                vibe_g = 2.5 if (vibe_raw is True or vibe_raw == 1 or str(vibe_raw).upper() in ["DETECTED", "HIGH"]) else 0.05

                # Process water input (handles analog waterValue or boolean water)
                water_raw = data.get("waterValue", 0)
                try:
                    water_val = float(water_raw)
                    water_cm = round((water_val / 4095.0) * 15.0, 1) if water_val > 0 else 0.0
                except (ValueError, TypeError):
                    water_cm = 15.0 if data.get("water", False) else 0.0

                # Convert raw gas readings to percentages/PPMs
                mq2_raw = float(data.get("mq2", 0) or 0)
                ch4_pct = round(min(max((mq2_raw / 4095.0) * 2.5, 0.08), 2.5), 2)

                mq9_raw = float(data.get("mq9", 0) or 0)
                co_ppm = round((mq9_raw / 4095.0) * 100.0, 1)

                mq135_raw = float(data.get("mq135", 0) or 0)
                co2_ppm = round(380.0 + ((mq135_raw / 4095.0) * 4620.0), 1)

                formatted_data = {
                    "o2": 20.8,
                    "co": co_ppm,
                    "ch4": ch4_pct,
                    "co2": co2_ppm,
                    "temp": float(data.get("temperature", 27.0) or 27.0),
                    "humidity": float(data.get("humidity", 55.0) or 55.0),
                    "strata_vibe_g": vibe_g,
                    "water_level_cm": water_cm
                }

                telemetry = sanitize_and_validate(formatted_data, sequence, source="LIVE_GATEWAY")
                status = GatewayStatus(
                    online=True,
                    message="OPERATIONAL",
                    latency_ms=resp.elapsed.total_seconds() * 1000.0,
                )
                return telemetry, status
            return None, GatewayStatus(online=False, message=f"HTTP {resp.status_code}")
        except requests.exceptions.Timeout:
            return None, GatewayStatus(online=False, message="TIMEOUT")
        except requests.exceptions.ConnectionError:
            return None, GatewayStatus(online=False, message="CONNECTION_REFUSED")
        except Exception as e:
            return None, GatewayStatus(online=False, message=str(e)[:24])

    def get_frame(self) -> Tuple[Optional[bytes], GatewayStatus]:
        """Fetches ONE still JPEG from the ESP32-S3-CAM's /capture endpoint.

        Important: this is NOT a continuous video stream. The current ESP32
        firmware (handleCapture()) only serves single snapshots on request —
        there is no MJPEG /stream route. "Live video" in the dashboard has to
        be built by calling this repeatedly (see the app.py fragment example
        below) — the live-feed illusion comes from fast polling, not from a
        true multipart stream. This was the missing piece causing the vision
        panel to show a permanent loading spinner: nothing in gateway.py was
        ever fetching a frame at all.
        """
        url = f"{self.base_url}/capture"
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 200 and resp.content:
                return resp.content, GatewayStatus(online=True, message="FRAME_OK", latency_ms=resp.elapsed.total_seconds() * 1000.0)
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
        return None, GatewayStatus(online=False, message="OFFLINE")