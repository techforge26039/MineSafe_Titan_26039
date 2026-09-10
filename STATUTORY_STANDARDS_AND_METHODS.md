# MineSafe Titan — Statutory Standards & Methods

## DGMS reference

The prototype uses the **Coal Mines Regulations, 2017** as a reference baseline for the
dashboard's ventilation-related display logic.

Official DGMS source:
https://www.dgms.gov.in/writereaddata/UploadFile/Coal_Mines_Regulation_2017_Noti.pdf

The cited ventilation provision states, among other things:
- oxygen not less than 19% by volume where persons work or pass;
- carbon dioxide not more than 0.5% by volume;
- inflammable gas not more than 0.75% in the general body of return air and 1.25% at any place;
- wet-bulb temperature not exceeding 33.5°C, with ventilation arrangements specified when it exceeds 30.5°C.

### Important classification

MineSafe labels these as **statutory/reference values**, not as a substitute for a
site's approved ventilation scheme.

The 50 ppm CO value displayed by MineSafe is a **configured operational alert threshold**.
It is deliberately not labelled as a universal CMR statutory ceiling.

## O₂ buffer proxy

The dashboard provides a transparent atmospheric proxy:

Δt = V_chamber × (O₂_current − 19%) / (N_workers × O₂_consumption)

with consistent units.

This is **not survival time, not a medical prediction, and not a guaranteed evacuation window**.
It ignores CO uptake, CO₂ effects, ventilation changes, leakage, stratification, temperature,
individual metabolic variation and many mine-specific factors.

## Decision pipeline

Telemetry → validation → hazard fusion → sensor AI → vision evidence → worker TRPI →
risk-aware routing → human-approved rover command → forensic audit.

## Engineering integrity

- Invalid LIVE telemetry is fail-safe rather than silently replaced by simulation.
- Worker-channel failures are not silently substituted with simulated workers.
- Generic YOLO classes are not treated as mine-specific safety classes.
- Synthetic model training is explicitly labelled until real-data validation is completed.
