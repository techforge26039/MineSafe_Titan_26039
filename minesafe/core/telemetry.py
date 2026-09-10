from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import math
from typing import Any, Dict, Optional, Tuple

from .config import PHYSICAL_LIMITS, REQUIRED_TELEMETRY


def wet_bulb_celsius(temp_c: float, rh_pct: float) -> float:
    # Stull approximation, adequate for dashboard engineering visualization.
    tw = (
        temp_c * math.atan(0.151977 * math.sqrt(rh_pct + 8.313659))
        + math.atan(temp_c + rh_pct)
        - math.atan(rh_pct - 1.676331)
        + 0.00391838 * (rh_pct ** 1.5) * math.atan(0.023101 * rh_pct)
        - 4.686035
    )
    return round(tw, 1)


def _to_float(name: str, value: Any) -> Tuple[Optional[float], str]:
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
    wet_bulb: float
    strata_vibe_g: float
    cgr: float
    water_level_cm: float
    sequence: int
    timestamp: str
    source: str
    is_valid: bool
    status_map: Dict[str, str]
    raw: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def sanitize_and_validate(raw: Dict[str, Any], sequence: int, source: str) -> TelemetryData:
    status: Dict[str, str] = {}
    values: Dict[str, float] = {}

    for key in REQUIRED_TELEMETRY:
        if key not in raw:
            status[key] = "MISSING_KEY"
            continue
        val, state = _to_float(key, raw[key])
        status[key] = state
        if val is not None:
            values[key] = val

    vibe_raw = raw.get("strata_vibe", False)
    if isinstance(vibe_raw, str):
        strata_vibe = vibe_raw.strip().lower() in {"1", "true", "yes", "on"}
    else:
        strata_vibe = bool(vibe_raw)
    status["strata_vibe"] = "VALID"

    # Derived value is only valid when its source values are valid.
    if "temp" in values and "humidity" in values:
        wb = wet_bulb_celsius(values["temp"], values["humidity"])
        status["wet_bulb"] = "DERIVED_VALID"
    else:
        wb = float("nan")
        status["wet_bulb"] = "DERIVED_UNAVAILABLE"

    is_valid = all(v in {"VALID", "DERIVED_VALID"} for v in status.values())
    # Keep a safe display value for missing channels, but preserve invalid state.
    d = {
        "o2": values.get("o2", float("nan")), "co": values.get("co", float("nan")),
        "ch4": values.get("ch4", float("nan")), "co2": values.get("co2", float("nan")),
        "temp": values.get("temp", float("nan")), "humidity": values.get("humidity", float("nan")),
        "strata_vibe_g": values.get("strata_vibe_g", float("nan")), "cgr": values.get("cgr", float("nan")),
        "water_level_cm": values.get("water_level_cm", float("nan")),
    }
    return TelemetryData(
        **d, wet_bulb=wb, sequence=sequence,
        timestamp=datetime.now(timezone.utc).isoformat(), source=source,
        is_valid=is_valid, status_map=status, raw=dict(raw),
    )


def demo_telemetry(scenario: str) -> Dict[str, Any]:
    base = {"o2": 20.8, "co": 5.0, "ch4": 0.10, "co2": 800.0, "temp": 27.0,
            "humidity": 55.0, "strata_vibe_g": 0.05, "cgr": 0.05, "water_level_cm": 4.0, "strata_vibe": False}
    cases = {
        "Critical Oxygen Deficiency": {"o2": 17.2, "co": 15, "ch4": .40, "co2": 2200, "temp": 29, "humidity": 60, "cgr": .2, "water_level_cm": 8, "strata_vibe_g": .08, "strata_vibe": False},
        "Carbon Monoxide Poisoning": {"o2": 19.8, "co": 85, "ch4": .30, "co2": 1800, "temp": 31, "humidity": 65, "cgr": .1, "water_level_cm": 8, "strata_vibe_g": .08, "strata_vibe": False},
        "Explosive Methane Ingress": {"o2": 18.9, "co": 20, "ch4": 1.45, "co2": 1500, "temp": 30, "humidity": 70, "cgr": .3, "water_level_cm": 9, "strata_vibe_g": .12, "strata_vibe": False},
        "Dynamic Strata Roof Collapse": {"o2": 20.1, "co": 10, "ch4": .20, "co2": 1100, "temp": 28, "humidity": 65, "cgr": 3.8, "water_level_cm": 7, "strata_vibe_g": 2.5, "strata_vibe": True},
        "Trapped Workers / Route Blocked": {"o2": 19.4, "co": 35, "ch4": .60, "co2": 3100, "temp": 32, "humidity": 80, "cgr": 1.1, "water_level_cm": 15, "strata_vibe_g": .25, "strata_vibe": False},
        "Thermal Ceiling Exceeded": {"o2": 20.2, "co": 10, "ch4": .15, "co2": 950, "temp": 42, "humidity": 90, "cgr": .1, "water_level_cm": 7, "strata_vibe_g": .08, "strata_vibe": False},
        "Compound Disaster": {"o2": 16.8, "co": 120, "ch4": 1.80, "co2": 6200, "temp": 38, "humidity": 85, "cgr": 4.5, "water_level_cm": 40, "strata_vibe_g": 4.0, "strata_vibe": True},
    }
    base.update(cases.get(scenario, {}))
    return base
