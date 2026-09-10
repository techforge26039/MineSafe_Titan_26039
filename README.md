# MineSafe Titan — SIH 2026 Final Prototype

**Problem statement target:** SIH 2026 PS #26039 — AI-Powered Underground Mine Safety

MineSafe Titan is a research/prototype decision-support system that fuses:

1. validated environmental telemetry,
2. a transparent sensor anomaly model,
3. worker wearable telemetry + TRPI triage,
4. YOLOv8 computer vision,
5. hazard-weighted rescue routing, and
6. rover command/ACK interfaces.

## Architecture

```text
ESP32 telemetry ─────┐
Worker wearables ────┼──> validation ─> hazard fusion ─> TRPI / priority
YOLOv8 vision ───────┘                         │
                                              ├─> safest route
                                              ├─> rover recommendation
                                              └─> forensic JSON
```

## Important model-integrity rule

The baseline `yolov8n.pt` is a generic pretrained object detector. It can detect
classes it was trained on, such as `person`, but it does **not** magically detect
`no_helmet`, `fallen_person`, `fire`, etc. The app therefore maps only explicit
classes from the loaded model into safety events.

For the SIH demonstration, use a custom mine/PPE model trained on permissioned
underground imagery. The included dataset template recommends:

`person, helmet, no_helmet, safety_vest, no_vest, fallen_person, smoke, fire, obstacle, blocked_exit`

## Run

### Windows

```text
start.bat
```

or:

```bash
python -m pip install -r requirements.txt
python -m streamlit run minesafe/app.py
```

### Linux/macOS

```bash
chmod +x start.sh
./start.sh
```

On first baseline inference, Ultralytics may need to download/cache `yolov8n.pt`.
Alternatively upload a custom `.pt` model in the sidebar.

## Train the mine-specific YOLO model

1. Create `dataset/data.yaml` from `dataset/data.yaml.example`.
2. Put labelled images under your dataset's train/validation paths.
3. Use one YOLO label file per image with normalized bounding boxes.
4. Run:

```bash
python train_yolov8.py
```

The script trains, validates and copies the best checkpoint to:

```text
models/weights/minevision_best.pt
```

Use that checkpoint in Titan V6 through the custom-model uploader.

## Live ESP32 API contract

### `GET /api/telemetry`

Expected JSON fields:

```json
{
  "o2": 20.8,
  "co": 5,
  "ch4": 0.10,
  "co2": 800,
  "temp": 27,
  "humidity": 55,
  "strata_vibe_g": 0.05,
  "cgr": 0.05,
  "water_level_cm": 4,
  "strata_vibe": false
}
```

Common aliases such as `oxygen_pct`, `co_ppm`, `methane`, `co2_ppm`,
`temperature`, `humidity_pct`, `vibration_g`, `roof_cgr`, and `water_cm`
are normalized automatically.

### `GET /api/workers`

Returns either a JSON list or `{ "workers": [...] }`. Each worker should expose
`tag`, `name`, `zone`, `hr_bpm`, `spo2_pct`, `temp_c`, and optionally
`fall_detected`.

### `POST /api/command`

The app sends a JSON command and accepts an explicit `ack` field from the
rover gateway. No fake sequence verification is displayed.

## Safety / engineering boundaries

- Invalid, missing, NaN and physically implausible telemetry is rejected; it is
  never silently clamped into a believable value.
- A failed LIVE gateway becomes a fail-safe data-quality state. Simulation is
  never silently substituted for a requested LIVE channel.
- A failed LIVE worker channel does not silently substitute demo workers.
- The sensor MLP is labelled `DEMO-TRAINED` until field data and validation are
  available.
- TRPI is a prototype prioritization score, not a medical prediction,
  survivability model, or certified safety metric.
- Vision events are decision-support signals, not a replacement for certified
  mine instrumentation or statutory emergency procedures.
- This project is not certified mine-control equipment.

## Final engineering upgrades

- Persistent YOLO tracking via ByteTrack when supported; automatic detection-only fallback if a tracker is unavailable.
- Route candidates are ranked with separate distance and hazard-penalty terms so the recommendation is explainable.
- TRPI exposes weighted contributors rather than a single opaque number.
- Forensic JSON captures telemetry state, multimodal hazard components, worker triage, route candidates, rover state, AI metadata and vision detections.
- `validate_yolo_dataset.py` validates image/label pairing, class IDs and normalized bounding boxes before training.
- `evaluate_sensor_model.py` provides held-out precision/recall/F1/confusion-matrix metrics for labelled sensor data.
- `SIH_DEMO_RUNBOOK.md` contains a concise demo sequence and judge-safe answers.


## Statutory standards & methods tab

The final UI includes a dedicated **STATUTORY STANDARDS & METHODS** tab containing:
- DGMS Coal Mines Regulations, 2017 reference thresholds used by the prototype;
- explicit distinction between statutory/reference values and configured operational thresholds;
- transparent O₂ buffer proxy formula and limitations;
- system-wide telemetry/dispatch console;
- decision-methodology table;
- official DGMS source link.

See `STATUTORY_STANDARDS_AND_METHODS.md` for the same engineering notes.

## Final-round positioning

This is an SIH internal-round prototype. It is intentionally transparent about what is
implemented versus what still requires field validation. Do not claim certification,
medical prediction, survival-time prediction, or real-mine deployment unless those claims
are independently validated.
