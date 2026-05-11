"""
Converts the RDD2022 Supervisely dataset to YOLO format.

Input structure (data/rdd2022/):
    train/img/*.jpg  +  train/ann/*.jpg.json
    test/img/*.jpg   +  test/ann/*.jpg.json

Output structure (data/yolo/):
    images/train/  images/val/  images/test/
    labels/train/  labels/val/  labels/test/
    rdd2022.yaml
"""

import json
import shutil
import random
from pathlib import Path

DATASET_DIR = Path(__file__).parent.parent / "data" / "rdd2022"
OUTPUT_DIR  = Path(__file__).parent.parent / "data" / "yolo"

CLASSES = [
    "alligator crack",
    "block crack",
    "longitudinal crack",
    "other corruption",
    "pothole",
    "repair",
    "transverse crack",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

VAL_RATIO = 0.2
SEED = 42


def supervisely_to_yolo(ann_path: Path, img_w: int, img_h: int) -> list[str]:
    """Return YOLO label lines for one annotation JSON."""
    data = json.loads(ann_path.read_text())
    lines = []
    for obj in data.get("objects", []):
        cls = obj.get("classTitle", "")
        if cls not in CLASS_TO_IDX:
            continue
        pts = obj["points"]["exterior"]
        x1, y1 = pts[0]
        x2, y2 = pts[1]
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        w  = abs(x2 - x1) / img_w
        h  = abs(y2 - y1) / img_h
        lines.append(f"{CLASS_TO_IDX[cls]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def process_split(src_split: str, dst_split: str):
    img_src = DATASET_DIR / src_split / "img"
    ann_src = DATASET_DIR / src_split / "ann"
    img_dst = OUTPUT_DIR / "images" / dst_split
    lbl_dst = OUTPUT_DIR / "labels" / dst_split
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    images = sorted(img_src.glob("*.jpg"))
    ok = skipped = 0
    for img_path in images:
        ann_path = ann_src / (img_path.name + ".json")
        if not ann_path.exists():
            skipped += 1
            continue

        ann_data = json.loads(ann_path.read_text())
        size = ann_data.get("size", {})
        img_w = size.get("width", 0)
        img_h = size.get("height", 0)
        if not img_w or not img_h:
            skipped += 1
            continue

        lines = supervisely_to_yolo(ann_path, img_w, img_h)
        shutil.copy2(img_path, img_dst / img_path.name)
        (lbl_dst / img_path.with_suffix(".txt").name).write_text("\n".join(lines))
        ok += 1

    print(f"  {dst_split}: {ok} images ({skipped} skipped)")
    return images


def split_train_val():
    img_src = DATASET_DIR / "train" / "img"
    ann_src = DATASET_DIR / "train" / "ann"

    images = sorted(img_src.glob("*.jpg"))
    random.seed(SEED)
    random.shuffle(images)
    split_idx = int(len(images) * (1 - VAL_RATIO))
    train_imgs = images[:split_idx]
    val_imgs   = images[split_idx:]

    for dst_split, img_list in [("train", train_imgs), ("val", val_imgs)]:
        img_dst = OUTPUT_DIR / "images" / dst_split
        lbl_dst = OUTPUT_DIR / "labels" / dst_split
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        ok = skipped = 0
        for img_path in img_list:
            ann_path = ann_src / (img_path.name + ".json")
            if not ann_path.exists():
                skipped += 1
                continue

            ann_data = json.loads(ann_path.read_text())
            size = ann_data.get("size", {})
            img_w = size.get("width", 0)
            img_h = size.get("height", 0)
            if not img_w or not img_h:
                skipped += 1
                continue

            lines = supervisely_to_yolo(ann_path, img_w, img_h)
            shutil.copy2(img_path, img_dst / img_path.name)
            (lbl_dst / img_path.with_suffix(".txt").name).write_text("\n".join(lines))
            ok += 1

        print(f"  {dst_split}: {ok} images ({skipped} skipped)")


def write_yaml():
    yaml_content = f"""path: {OUTPUT_DIR.as_posix()}
train: images/train
val:   images/val
test:  images/test

nc: {len(CLASSES)}
names: {CLASSES}
"""
    (OUTPUT_DIR / "rdd2022.yaml").write_text(yaml_content)
    print(f"  YAML: {OUTPUT_DIR / 'rdd2022.yaml'}")


def main():
    print("=== RDD2022: Supervisely -> YOLO conversion ===")

    print("\n[1/3] Splitting train → train/val ...")
    split_train_val()

    print("\n[2/3] Processing test split ...")
    img_dst = OUTPUT_DIR / "images" / "test"
    lbl_dst = OUTPUT_DIR / "labels" / "test"
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)
    img_src = DATASET_DIR / "test" / "img"
    ann_src = DATASET_DIR / "test" / "ann"
    ok = skipped = 0
    for img_path in sorted(img_src.glob("*.jpg")):
        ann_path = ann_src / (img_path.name + ".json")
        if not ann_path.exists():
            skipped += 1
            continue
        ann_data = json.loads(ann_path.read_text())
        size = ann_data.get("size", {})
        img_w = size.get("width", 0)
        img_h = size.get("height", 0)
        if not img_w or not img_h:
            skipped += 1
            continue
        lines = supervisely_to_yolo(ann_path, img_w, img_h)
        shutil.copy2(img_path, img_dst / img_path.name)
        (lbl_dst / img_path.with_suffix(".txt").name).write_text("\n".join(lines))
        ok += 1
    print(f"  test: {ok} images ({skipped} skipped)")

    print("\n[3/3] Writing dataset YAML ...")
    write_yaml()

    print("\nDone.")


if __name__ == "__main__":
    main()
