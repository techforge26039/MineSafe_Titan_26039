from __future__ import annotations

APP_NAME = "MineSafe Titan"
VERSION = "v7.0.0-SIH-2026-FINAL"
NODE_ID = "TITAN-COMMAND-NODE-01"

REGULATORY_BOUNDS = {
    "o2": {"statutory_min": 19.0, "warning_band": 19.5, "unit": "%"},
    "co": {"operational_limit": 50.0, "warning_band": 25.0, "unit": "PPM"},
    "ch4": {"return_air_max": 0.75, "general_body_max": 1.25, "unit": "%"},
    "co2": {"statutory_max": 5000.0, "statutory_pct": 0.5, "unit": "PPM"},
    "wet_bulb": {"action_threshold": 30.5, "ceiling_limit": 33.5, "unit": "°C"},
}

# Engineering plausibility bounds. Out-of-range data is REJECTED, never silently clamped.
PHYSICAL_LIMITS = {
    "o2": (0.0, 30.0),
    "co": (0.0, 2000.0),
    "ch4": (0.0, 10.0),
    "co2": (0.0, 50000.0),
    "temp": (-20.0, 80.0),
    "humidity": (0.0, 100.0),
    "strata_vibe_g": (0.0, 20.0),
    "cgr": (0.0, 20.0),
    "water_level_cm": (0.0, 10000.0),
}

REQUIRED_TELEMETRY = list(PHYSICAL_LIMITS)
DEFAULT_GATEWAY = "http://192.168.137.89"
REQUEST_TIMEOUT = 1.5

# Coal mine geometry aligned from BASE to ZONE_A with bypass shafts and crosscuts.
MINE_GRAPH = {
    "BASE": {"ZONE_A": 45, "SHAFT_NORTH": 35, "SHAFT_SOUTH": 50},
    "SHAFT_NORTH": {"BASE": 35, "ZONE_A": 25, "AUX_BYPASS": 30},
    "SHAFT_SOUTH": {"BASE": 50, "ZONE_A": 40, "VENT_CROSS": 35},
    "AUX_BYPASS": {"SHAFT_NORTH": 30, "ZONE_A": 20},
    "VENT_CROSS": {"SHAFT_SOUTH": 35, "ZONE_A": 25},
    "ZONE_A": {"BASE": 45, "SHAFT_NORTH": 25, "SHAFT_SOUTH": 40, "AUX_BYPASS": 20, "VENT_CROSS": 25},
}

NODE_COORDS = {
    "BASE": (1.0, 5.0),
    "SHAFT_NORTH": (4.5, 6.5),
    "SHAFT_SOUTH": (4.5, 3.5),
    "AUX_BYPASS": (6.8, 6.2),
    "VENT_CROSS": (6.8, 3.8),
    "ZONE_A": (8.5, 5.0),
}