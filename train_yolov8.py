"""Train a MineSafe-specific YOLOv8 detector.

Expected Ultralytics dataset layout is described in dataset/README.md.
Run from the project root:
    python train_yolov8.py

The script trains, validates, reports the best checkpoint and copies it to
models/weights/minevision_best.pt for easy upload/use in Titan.
"""
from pathlib import Path
import shutil
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATASET_YAML = ROOT / "dataset" / "data.yaml"
BASE_MODEL = "yolov8n.pt"
PROJECT = ROOT / "runs" / "minevision"
RUN_NAME = "yolov8n_minesafe"
OUT = ROOT / "models" / "weights" / "minevision_best.pt"

if not DATASET_YAML.exists():
    raise SystemExit("Missing dataset/data.yaml. Copy dataset/data.yaml.example and point it to your labelled dataset.")

model = YOLO(BASE_MODEL)
train_results = model.train(
    data=str(DATASET_YAML), epochs=80, imgsz=640, batch=16, patience=20,
    workers=2, project=str(PROJECT), name=RUN_NAME, pretrained=True,
    cache=False, verbose=True,
)

best = PROJECT / RUN_NAME / "weights" / "best.pt"
if not best.exists():
    raise SystemExit(f"Training finished but best.pt was not found at {best}")

val_model = YOLO(str(best))
metrics = val_model.val(data=str(DATASET_YAML), imgsz=640, verbose=False)
print("Validation complete.")
try:
    print(f"mAP50-95: {float(metrics.box.map):.4f}")
    print(f"mAP50:    {float(metrics.box.map50):.4f}")
except Exception:
    pass

OUT.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(best, OUT)
print(f"Best checkpoint copied to: {OUT}")
