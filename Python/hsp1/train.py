"""
YOLOv11 training on RDD2022.
Run prepare_dataset.py first to generate data/yolo/.
"""

from pathlib import Path
from ultralytics import YOLO

YAML      = Path(__file__).parent / "data" / "yolo" / "rdd2022.yaml"
MODEL     = "yolo11m.pt"
EPOCHS    = 100
IMGSZ     = 640
BATCH     = 16
WORKERS   = 4
PROJECT   = Path(__file__).parent / "runs"
NAME      = "rdd2022_yolo11m_damage"


def main():
    model = YOLO(MODEL)

    results = model.train(
        data=str(YAML.as_posix()),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        workers=WORKERS,
        project=str(PROJECT.as_posix()),
        name=NAME,
        patience=10,
        save_period=10,
        val=True,
        plots=True,
        device=0,
    )

    print(f"\nBest weights: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    main()
