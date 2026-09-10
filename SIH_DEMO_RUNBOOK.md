# SIH 2026 Demo Runbook — PS 26039

## 3-minute judge flow

### 1. Start nominal
Select `SIMULATION LAB` → `Nominal Operations`.
Show the digital twin, worker cards, route and green baseline.

### 2. Trigger compound incident
Select `Compound Disaster` and enable blockage.
Point out:
- O2 deficiency
- methane rise
- CO/CO2 burden
- thermal load
- strata vibration/crack-growth rate
- blocked rescue edge

The system raises multimodal risk, recomputes TRPI and ranks safe route candidates.

### 3. Add computer vision
Enable `IMAGE UPLOAD` or `CAMERA SNAPSHOT`.
Use `yolov8n.pt` for generic person detection or upload `models/weights/minevision_best.pt` after custom training.

With a custom model, explicitly mapped classes such as `no_helmet`, `fallen_person`, `smoke`, `fire` or `obstacle` can contribute to the vision score.

### 4. Explain the AI honestly
Say:
> "Our baseline sensor model is demo-trained on synthetic engineering scenarios because validated mine telemetry is not available at sufficient scale. The pipeline is designed for retraining and held-out validation on real mine data. Our baseline YOLO model is generic; mine-specific safety events require our custom labelled model."

### 5. Demonstrate human-in-the-loop rover control
Use the rover controls only after showing the recommendation. In LIVE mode, commands are sent only through a healthy gateway. The prototype does not claim autonomous certified control.

## Strong judge questions

**Where is the AI?**
- Sensor anomaly MLP + YOLO computer vision + multimodal fusion.

**How do you avoid false live data?**
- Invalid/missing/out-of-range telemetry is rejected. A failed LIVE gateway enters fail-safe state; simulated data is never substituted.

**Why YOLO?**
- Vision adds a second sensing modality for worker/PPE/environmental visual events and can complement gas/geotechnical telemetry.

**How will you validate it?**
- Permissioned historical mine telemetry and labelled underground imagery, with train/validation/test separation and held-out metrics.

**Is this certified?**
- No. It is a research/decision-support prototype intended to augment, not replace, certified mine-safety systems and statutory procedures.


## 30-second regulatory-methodology proof

Open **STATUTORY STANDARDS & METHODS** and show:
1. 19% O₂, 0.5% CO₂, 0.75%/1.25% methane and 30.5°C/33.5°C wet-bulb reference logic.
2. The explicit note that 50 ppm CO is a configured operational threshold, not presented as a universal statutory CMR ceiling.
3. The O₂ buffer proxy and its limitations.
4. The traceable command console.
5. The official DGMS source button.

This is a strong answer if a judge asks: **"Where did your thresholds come from?"**
