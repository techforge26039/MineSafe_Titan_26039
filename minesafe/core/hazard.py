from __future__ import annotations
from typing import Dict, List, Tuple
import math
from .telemetry import TelemetryData


def _safe(v: float) -> float:
    return 0.0 if not math.isfinite(v) else v


def evaluate_hazard(t: TelemetryData) -> Tuple[int, str, str, List[str], Dict[str, float]]:
    if not t.is_valid:
        return 100, 'DATA QUALITY', 'FAIL-SAFE', ['Invalid telemetry packet — decision engine held in fail-safe state'], {'ATMOSPHERIC':100.0,'GEOTECHNICAL':100.0,'THERMAL':100.0}
    breaches: List[str] = []
    o2,co,ch4,co2,wb,cgr,vibe=map(_safe,[t.o2,t.co,t.ch4,t.co2,t.wet_bulb,t.cgr,t.strata_vibe_g])
    o2_pen=max(0,(19.5-o2)/4.5*100) if o2<19.5 else 0
    co_pen=min(100,co/100*100); ch4_pen=min(100,ch4/1.25*100); co2_pen=min(100,co2/5000*100)
    atm=.35*o2_pen+.30*co_pen+.25*ch4_pen+.10*co2_pen
    geo=min(100,.55*(vibe/3*100)+.45*(cgr/3*100))
    therm=100 if wb>=33.5 else (50+(wb-30.5)/3*50 if wb>=30.5 else max(0,(wb-20)/10.5*40))
    if o2<19: breaches.append(f'O₂ critical deficiency ({o2:.1f}% < 19.0%)')
    if co>=50: breaches.append(f'CO operational ceiling ({co:.0f} PPM ≥ 50 PPM)')
    if ch4>=.75: breaches.append(f'CH₄ return-air threshold ({ch4:.2f}% ≥ 0.75%)')
    if ch4>=1.25: breaches.append(f'CH₄ general-body ceiling ({ch4:.2f}% ≥ 1.25%)')
    if co2>=5000: breaches.append(f'CO₂ threshold ({co2:.0f} PPM ≥ 5000 PPM)')
    if wb>=33.5: breaches.append(f'Wet-bulb ceiling ({wb:.1f}°C ≥ 33.5°C)')
    if vibe>=2: breaches.append(f'High strata vibration ({vibe:.2f} g)')
    if cgr>=3: breaches.append(f'High crack-growth rate ({cgr:.2f} mm/min)')
    total=int(min(100,.50*atm+.35*geo+.15*therm)); total=max(total,70) if breaches else total
    scores={'ATMOSPHERIC':atm,'GEOTECHNICAL':geo,'THERMAL':therm}; ordered=sorted(scores,key=scores.get,reverse=True)
    return total,ordered[0],ordered[1],breaches,scores


def risk_label(score: float)->str:
    return 'CRITICAL' if score>=80 else 'HIGH' if score>=60 else 'ELEVATED' if score>=35 else 'LOW'


def calculate_trpi(t:TelemetryData,hr:float,spo2:float,route_blocked:bool,fall_detected:bool=False)->Tuple[float,Dict[str,float]]:
    if not t.is_valid: return 100.0,{'Data quality fail-safe':100.0}
    o2,co,ch4,co2=map(_safe,[t.o2,t.co,t.ch4,t.co2])
    s_atm=max(0,min(100,(19.5-o2)/4.5*100)) if o2<19.5 else 0
    s_tox=min(100,max(co/100,ch4/1.25,co2/5000)*100); s_phys=max(0,min(100,(hr-75)/65*100)); s_oxy=max(0,min(100,(95-spo2)/10*100)) if spo2<95 else 0
    s_route=100 if route_blocked else 0; s_geo=min(100,max(t.strata_vibe_g/3*100,t.cgr/3*100))
    s_fall=100 if fall_detected else 0
    trpi=.22*s_atm+.22*s_tox+.13*s_phys+.10*s_oxy+.13*s_route+.10*s_geo+.10*s_fall
    return round(min(100,trpi),1),{'Atmospheric (22%)':round(.22*s_atm,1),'Toxicity (22%)':round(.22*s_tox,1),'Physiological (13%)':round(.13*s_phys,1),'SpO₂ (10%)':round(.10*s_oxy,1),'Route (13%)':round(.13*s_route,1),'Strata (10%)':round(.10*s_geo,1),'Fall flag (10%)':round(.10*s_fall,1)}
