"""
MineSafe Titan — statutory reference layer and engineering methods.

The values below are presented as a configurable reference baseline for a
prototype. They are not a substitute for site-specific statutory instructions,
approved ventilation schemes, certified instrumentation, or mine-manager
authority.

Primary public source:
DGMS, Coal Mines Regulations, 2017:
https://www.dgms.gov.in/writereaddata/UploadFile/Coal_Mines_Regulation_2017_Noti.pdf
Relevant ventilation text appears in Regulation 53(2)(b)-(e) in the published
2017 regulation text.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class StandardReference:
    name: str
    value: str
    classification: str
    note: str


CMR_2017_REFERENCES = (
    StandardReference(
        "Oxygen",
        "≥ 19.0% by volume",
        "STATUTORY REFERENCE",
        "At places where persons are required to work or pass, under the cited ventilation provision.",
    ),
    StandardReference(
        "Carbon dioxide",
        "≤ 0.5% by volume (5,000 ppm)",
        "STATUTORY REFERENCE",
        "Prototype converts 0.5% to 5,000 ppm for sensor display.",
    ),
    StandardReference(
        "Inflammable gas / methane",
        "≤ 0.75% return-air general body; ≤ 1.25% at any place",
        "STATUTORY REFERENCE",
        "Use the applicable mine/gassiness context and approved procedures.",
    ),
    StandardReference(
        "Wet-bulb temperature",
        "≤ 33.5°C; >30.5°C requires ventilation arrangement",
        "STATUTORY REFERENCE",
        "The cited provision specifies at least 1 m/s air movement when wet-bulb exceeds 30.5°C.",
    ),
    StandardReference(
        "Carbon monoxide",
        "50 ppm",
        "CONFIGURED OPERATIONAL THRESHOLD",
        "Displayed as a prototype alert threshold; do not present this value as a universal CMR statutory ceiling.",
    ),
)

# The screenshot/demo method uses this as a transparent atmospheric proxy.
# It estimates time to the 19% oxygen threshold under simplified assumptions.
# It must never be described as survival time.
def o2_buffer_proxy_minutes(
    chamber_volume_m3: float,
    current_o2_pct: float,
    workers: int,
    o2_consumption_l_min_per_worker: float = 0.25,
) -> float:
    if chamber_volume_m3 <= 0:
        raise ValueError("chamber_volume_m3 must be > 0")
    if workers <= 0:
        raise ValueError("workers must be > 0")
    if o2_consumption_l_min_per_worker <= 0:
        raise ValueError("o2_consumption_l_min_per_worker must be > 0")
    if current_o2_pct <= 19.0:
        return 0.0
    # V[m3] -> L; oxygen fraction difference -> decimal fraction.
    available_o2_l = chamber_volume_m3 * 1000.0 * ((current_o2_pct - 19.0) / 100.0)
    total_consumption_l_min = workers * o2_consumption_l_min_per_worker
    return round(max(0.0, available_o2_l / total_consumption_l_min), 1)


def reference_rows():
    return [
        {
            "Parameter": x.name,
            "Reference": x.value,
            "Class": x.classification,
            "Method note": x.note,
        }
        for x in CMR_2017_REFERENCES
    ]


def methodology_rows():
    return [
        ("Telemetry validation", "Missing, malformed, NaN and physically implausible values are rejected; invalid LIVE data enters fail-safe state."),
        ("Hazard fusion", "Atmospheric + geotechnical + thermal components are combined into an explainable 0–100 prototype risk score."),
        ("Sensor AI", "TinyMLP baseline is explicitly demo-trained until labelled historical mine data is supplied and evaluated on a held-out split."),
        ("Computer vision", "Only explicit model classes create mapped safety events. Generic YOLO does not infer missing PPE/falls/fire from class absence."),
        ("TRPI", "Weighted worker prioritization using atmosphere, toxicity, physiology, SpO₂, route, strata and fall evidence."),
        ("Routing", "Candidate paths are ranked by distance plus explicit hazard penalties; blocked edges are removed."),
        ("O₂ buffer proxy", "Simplified chamber/worker consumption calculation to the 19% threshold; not a medical, survivability or evacuation-time guarantee."),
        ("Human authority", "Rover control remains human-approved and is restricted when the live gateway is unavailable."),
        ("Audit", "Telemetry state, model metadata, hazard components, workers, routes and limitations can be exported as JSON."),
    ]
