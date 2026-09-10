# MineSafe Titan V6 — YOLOv8 Vision Module

## What is integrated

Titan V6 now has a real YOLOv8 inference path in `minesafe/models/vision.py`.

- Baseline: `yolov8n.pt` for lightweight generic detection.
- Image upload and webcam snapshot are supported in the Streamlit sidebar.
- A custom YOLOv8 `.pt` model can be uploaded without changing the app.
- Detections are annotated and exported in the forensic JSON.
- Only explicitly mapped safety classes can influence the hazard score.
- The baseline COCO model is **not** treated as a PPE/fall/fire detector.

Ultralytics documents YOLOv8 detection outputs as boxes containing coordinates, confidence and class IDs, and supports model export for deployment formats such as ONNX/TensorRT. See the official docs before field deployment.

## Recommended SIH custom classes

For a mine-specific dataset, start with a compact class list:

1. `person`
2. `helmet`
3. `no_helmet`
4. `safety_vest`
5. `no_vest`
6. `fallen_person`
7. `smoke`
8. `fire`
9. `obstacle`
10. `blocked_exit`

Do not add a class unless you have representative labelled data for it. Underground scenes should cover low light, dust, occlusion, reflective PPE, different camera heights and worker distances.

## Training example

```bash
pip install -r requirements.txt
python train_yolov8.py
```

The supplied training helper expects a YOLO-format dataset and writes a custom `best.pt`. Upload that file in the V6 sidebar under **Optional custom YOLOv8 .pt model**.

## Suggested dataset structure

```text
dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
  data.yaml
```

Example `data.yaml`:

```yaml
path: ./dataset
train: images/train
val: images/val
names:
  0: person
  1: helmet
  2: no_helmet
  3: safety_vest
  4: no_vest
  5: fallen_person
  6: smoke
  7: fire
  8: obstacle
  9: blocked_exit
```

## Judge-safe claims

Say:

> "Titan uses YOLOv8 as a vision sensing layer. The generic pretrained model is used only for classes it was actually trained on; mine-specific PPE and incident detection requires a custom model trained and validated on representative underground imagery."

Do **not** claim certified PPE compliance, crack detection, fall detection or fire detection from the baseline COCO model.

## Deployment path

For an edge rover, export the validated custom model to ONNX or TensorRT after hardware-in-the-loop testing. Keep environmental gas/ventilation safety logic independent of the vision model so a vision failure cannot disable the primary fail-safe path.
