from __future__ import annotations

APP_NAME = "MineSafe Titan"
VERSION = "v7.3.0-SIH-2026-FINAL"
NODE_ID = "TITAN-COMMAND-NODE-01"

REGULATORY_BOUNDS = {
    "o2": {"statutory_min": 19.0, "warning_band": 19.5, "unit": "%"},
    "co": {"operational_limit": 50.0, "warning_band": 25.0, "unit": "PPM"},
    "ch4": {"return_air_max": 0.75, "general_body_max": 1.25, "unit": "%"},
    "co2": {"statutory_max": 5000.0, "statutory_pct": 0.5, "unit": "PPM"},
    "wet_bulb": {"action_threshold": 30.5, "ceiling_limit": 33.5, "unit": "°C"},
    "water_level_cm": {"operational_limit": 15.0, "ceiling_limit": 30.0, "unit": "cm"},
}

PHYSICAL_LIMITS = {
    "o2": (0.0, 30.0),
    "co": (0.0, 2000.0),
    "ch4": (0.0, 10.0),
    "co2": (0.0, 50000.0),
    "temp": (-20.0, 80.0),
    "humidity": (0.0, 100.0),
    "strata_vibe_g": (0.0, 20.0),
    "water_level_cm": (0.0, 1000.0),
}

REQUIRED_TELEMETRY = list(PHYSICAL_LIMITS)
DEFAULT_GATEWAY = "http://192.168.137.89"
REQUEST_TIMEOUT = 1.5

MINE_GRAPH = {
    "BASE": {"ZONE_A": 20},
    "ZONE_A": {"BASE": 20, "ZONE_B": 45, "SHAFT_NORTH": 30},
    "SHAFT_NORTH": {"ZONE_A": 30, "ZONE_B": 25},
    "ZONE_B": {"ZONE_A": 45, "SHAFT_NORTH": 25},
}

NODE_COORDS = {
    "BASE": (0.4, 5.0),
    "ZONE_A": (2.6, 5.0),
    "SHAFT_NORTH": (5.0, 6.5),
    "ZONE_B": (7.0, 5.0),
}