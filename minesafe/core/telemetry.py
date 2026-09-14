from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import math
from typing import Any, Dict, Optional, Tuple
from .config import PHYSICAL_LIMITS, REQUIRED_TELEMETRY

def wet_bulb_celsius(temp_c: float, rh_pct: float) -> float:
    tw = (
        temp_c * math.atan(0.151977 * math.sqrt(rh_pct + 8.313659))
        + math.atan(temp_c + rh_pct)
        - math.atan(rh_pct - 1.676331)
        + 0.00391838 * (rh_pct ** 1.5) * math.atan(0.023101 * rh_pct)
        - 4.686035
    )
    return round(tw, 1)

def _to_float(name: str, value: Any) -> Tuple[Optional[float], str]:
    if name not in PHYSICAL_LIMITS:
        return None, "PARSE_ERROR"
    lo, hi = PHYSICAL_LIMITS[name]
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None, "PARSE_ERROR"
    if not math.isfinite(v):
        return None, "FAULT_NAN"
    if not lo <= v <= hi:
        return None, "OUT_OF_BOUNDS"
    return v, "VALID"

@dataclass(frozen=True)
class TelemetryData:
    o2: float
    co: float
    ch4: float
    co2: float
    temp: float
    humidity: float
    strata_vibe_g: float
    water_level_cm: float
    wet_bulb: float
    sequence: int
    timestamp: str
    source: str
    is_valid: bool
    status_map: Dict[str, str]
    raw: Dict[str, Any]
    cgr: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

def sanitize_and_validate(raw: Dict[str, Any], sequence: int, source: str) -> TelemetryData:
    status: Dict[str, str] = {}
    values: Dict[str, float] = {}

    for key in REQUIRED_TELEMETRY:
        if key not in raw or raw[key] is None:
            status[key] = "MISSING_KEY"
            continue
        val, state = _to_float(key, raw[key])
        status[key] = state
        if val is not None:
            values[key] = val

    if "temp" in values and "humidity" in values:
        wb = wet_bulb_celsius(values["temp"], values["humidity"])
        status["wet_bulb"] = "DERIVED_VALID"
    else:
        wb = float("nan")
        status["wet_bulb"] = "DERIVED_UNAVAILABLE"

    is_valid = all(v in {"VALID", "DERIVED_VALID"} for v in status.values())

    v_raw = raw.get("strata_vibe_g", 0.05)
    try:
        clean_vibe = float(values.get("strata_vibe_g", v_raw))
        if not math.isfinite(clean_vibe): clean_vibe = 0.05
    except Exception:
        clean_vibe = 0.05

    return TelemetryData(
        o2=values.get("o2", float(raw.get("o2", 20.8))),
        co=values.get("co", float(raw.get("co", 5.0))),
        ch4=values.get("ch4", float(raw.get("ch4", 0.10))),
        co2=values.get("co2", float(raw.get("co2", 450.0))),
        temp=values.get("temp", float(raw.get("temp", 27.0))),
        humidity=values.get("humidity", float(raw.get("humidity", 55.0))),
        strata_vibe_g=clean_vibe,
        water_level_cm=values.get("water_level_cm", float(raw.get("water_level_cm", 2.0))),
        wet_bulb=wb if math.isfinite(wb) else 22.0,
        sequence=sequence,
        timestamp=datetime.now(timezone.utc).isoformat(),
        source=source,
        is_valid=is_valid,
        status_map=status,
        raw=dict(raw),
        cgr=float(raw.get("cgr", 0.0)),
    )

def demo_telemetry(scenario: str) -> Dict[str, Any]:
    base = {
        "o2": 20.8, "co": 5.0, "ch4": 0.10, "co2": 450.0, "temp": 27.0,
        "humidity": 55.0, "strata_vibe_g": 0.05, "water_level_cm": 2.0
    }
    cases = {
        "Critical Oxygen Deficiency": {"o2": 16.5, "co": 12.0, "ch4": 0.25, "co2": 2200.0, "temp": 28.0, "humidity": 60.0, "strata_vibe_g": 0.05, "water_level_cm": 2.5},
        "Carbon Monoxide Poisoning": {"o2": 19.8, "co": 85.0, "ch4": 0.30, "co2": 1800.0, "temp": 31.0, "humidity": 65.0, "strata_vibe_g": 0.08, "water_level_cm": 3.0},
        "Explosive Methane Ingress": {"o2": 18.9, "co": 20.0, "ch4": 1.45, "co2": 1500.0, "temp": 30.0, "humidity": 70.0, "strata_vibe_g": 0.12, "water_level_cm": 3.0},
        "Dynamic Strata Roof Collapse": {"o2": 20.1, "co": 10.0, "ch4": 0.20, "co2": 1100.0, "temp": 28.0, "humidity": 65.0, "strata_vibe_g": 2.8, "water_level_cm": 4.0},
        "Thermal Ceiling Exceeded": {"o2": 20.2, "co": 10.0, "ch4": 0.15, "co2": 950.0, "temp": 42.0, "humidity": 90.0, "strata_vibe_g": 0.08, "water_level_cm": 5.0},
        "Water Sump Inundation": {"o2": 20.4, "co": 6.0, "ch4": 0.12, "co2": 600.0, "temp": 25.0, "humidity": 92.0, "strata_vibe_g": 0.10, "water_level_cm": 28.5},
        "Toxic CO₂ Blackdamp Ingress": {"o2": 18.2, "co": 15.0, "ch4": 0.20, "co2": 6500.0, "temp": 29.0, "humidity": 75.0, "strata_vibe_g": 0.06, "water_level_cm": 4.5},
        "Compound Disaster": {"o2": 15.8, "co": 120.0, "ch4": 1.80, "co2": 6200.0, "temp": 38.0, "humidity": 88.0, "strata_vibe_g": 3.5, "water_level_cm": 35.0},
    }
    base.update(cases.get(scenario, {}))
    return base