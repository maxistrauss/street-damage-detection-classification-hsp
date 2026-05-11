"""
YOLOv11 training on RDD2022.
Run prepare_dataset.py first to generate data/yolo/.
"""

from pathlib import Path
from ultralytics import YOLO

YAML      = Path(__file__).parent.parent / "data" / "yolo" / "rdd2022.yaml"
MODEL     = "yolo11m.pt"   # medium — change to yolo11s.pt for faster iteration
EPOCHS    = 50
IMGSZ     = 640
BATCH     = 16
WORKERS   = 4
PROJECT   = Path(__file__).parent.parent / "runs"
NAME      = "rdd2022_yolo11m"


def main():
    model = YOLO(MODEL)

    results = model.train(
        data=str(YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        workers=WORKERS,
        project=str(PROJECT),
        name=NAME,
        # class weights compensate for imbalance (longitudinal/transverse >> pothole)
        # ultralytics applies these via cls_pw; tune after first run if needed
        patience=10,       # early stopping
        save_period=10,    # checkpoint every 10 epochs
        val=True,
        plots=True,
    )

    print(f"\nBest weights: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    main()
