from __future__ import annotations
from typing import Dict, List, Tuple
import math
from .telemetry import TelemetryData


def _safe(v: float) -> float:
    return 0.0 if not math.isfinite(v) else v


def evaluate_hazard(t: TelemetryData) -> Tuple[int, str, str, List[str], Dict[str, float]]:
    if not t.is_valid:
        return 100, 'DATA QUALITY', 'FAIL-SAFE', ['Invalid telemetry packet — decision engine held in fail-safe state'], {'ATMOSPHERIC': 100.0, 'GEOTECHNICAL': 100.0, 'THERMAL': 100.0}
    
    breaches: List[str] = []
    o2, co, ch4, co2, wb, vibe = map(_safe, [t.o2, t.co, t.ch4, t.co2, t.wet_bulb, t.strata_vibe_g])
    
    o2_pen = max(0, (19.5 - o2) / 4.5 * 100) if o2 < 19.5 else 0
    co_pen = min(100, co / 50.0 * 100)
    ch4_pen = min(100, ch4 / 1.25 * 100)
    co2_pen = min(100, co2 / 5000.0 * 100)
    
    atm = .35 * o2_pen + .30 * co_pen + .25 * ch4_pen + .10 * co2_pen
    geo = min(100.0, (vibe / 2.0) * 100.0)
    therm = 100.0 if wb >= 33.5 else (50.0 + (wb - 30.5) / 3.0 * 50.0 if wb >= 30.5 else max(0.0, (wb - 20.0) / 10.5 * 40.0))
    
    if o2 < 19.0: breaches.append(f'O₂ critical deficiency ({o2:.1f}% < 19.0%)')
    if co >= 50.0: breaches.append(f'CO operational ceiling ({co:.0f} PPM ≥ 50 PPM)')
    if ch4 >= 0.75: breaches.append(f'CH₄ return-air threshold ({ch4:.2f}% ≥ 0.75%)')
    if ch4 >= 1.25: breaches.append(f'CH₄ general-body ceiling ({ch4:.2f}% ≥ 1.25%)')
    if co2 >= 5000.0: breaches.append(f'CO₂ threshold ({co2:.0f} PPM ≥ 5000 PPM)')
    if wb >= 33.5: breaches.append(f'Wet-bulb ceiling ({wb:.1f}°C ≥ 33.5°C)')
    if vibe >= 2.0: breaches.append(f'High strata vibration ({vibe:.2f} g ≥ 2.0 g)')
    
    total = int(min(100.0, .50 * atm + .35 * geo + .15 * therm))
    total = max(total, 70) if breaches else total
    
    scores = {'ATMOSPHERIC': atm, 'GEOTECHNICAL': geo, 'THERMAL': therm}
    ordered = sorted(scores, key=scores.get, reverse=True)
    return total, ordered[0], ordered[1], breaches, scores


def risk_label(score: float) -> str:
    return 'CRITICAL' if score >= 80 else 'HIGH' if score >= 60 else 'ELEVATED' if score >= 35 else 'LOW'


def calculate_trpi(t: TelemetryData, hr: float, spo2: float, route_blocked: bool, fall_detected: bool = False) -> Tuple[float, Dict[str, float]]:
    if not t.is_valid:
        return 100.0, {'Data quality fail-safe': 100.0}
    
    o2, co, ch4, co2, vibe = map(_safe, [t.o2, t.co, t.ch4, t.co2, t.strata_vibe_g])
    s_atm = max(0, min(100, (19.5 - o2) / 4.5 * 100)) if o2 < 19.5 else 0
    s_tox = min(100, max(co / 50.0, ch4 / 1.25, co2 / 5000.0) * 100)
    s_phys = max(0, min(100, (hr - 75.0) / 65.0 * 100))
    s_oxy = max(0, min(100, (95.0 - spo2) / 10.0 * 100)) if spo2 < 95.0 else 0
    s_route = 100.0 if route_blocked else 0.0
    s_geo = min(100.0, (vibe / 2.0) * 100.0)
    s_fall = 100.0 if fall_detected else 0.0
    
    trpi = .22 * s_atm + .22 * s_tox + .13 * s_phys + .10 * s_oxy + .13 * s_route + .10 * s_geo + .10 * s_fall
    return round(min(100.0, trpi), 1), {
        'Atmospheric (22%)': round(.22 * s_atm, 1),
        'Toxicity (22%)': round(.22 * s_tox, 1),
        'Physiological (13%)': round(.13 * s_phys, 1),
        'SpO₂ (10%)': round(.10 * s_oxy, 1),
        'Route (13%)': round(.13 * s_route, 1),
        'Strata (10%)': round(.10 * s_geo, 1),
        'Fall flag (10%)': round(.10 * s_fall, 1)
    }