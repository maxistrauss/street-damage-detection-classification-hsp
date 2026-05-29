"""
Run inference with a trained YOLOv11 model on new road images.
Usage: python predict.py --source path/to/image_or_folder --weights runs/rdd2022_yolo11m/weights/best.pt
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source",  required=True, help="Image, folder, or video path")
    parser.add_argument("--weights", default="runs/rdd2022_yolo11m/weights/best.pt")
    parser.add_argument("--conf",    type=float, default=0.25)
    parser.add_argument("--iou",     type=float, default=0.45)
    parser.add_argument("--save",    action="store_true", default=True)
    args = parser.parse_args()

    model = YOLO(args.weights)
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        save=args.save,
        project="runs/predict",
    )

    for r in results:
        print(r.path, "→", len(r.boxes), "detections")


if __name__ == "__main__":
    main()
