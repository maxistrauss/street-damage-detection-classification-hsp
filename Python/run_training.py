"""
Direct training script — run via: conda run -n rdd2022 python run_training.py
Progress is printed live to stdout. Results land in runs/rdd2022_yolo11m/.
"""
import os

# Must happen before any torch import — adds conda env DLLs to Windows search path
_ENV = r'C:\Users\phili\anaconda3\envs\rdd2022_clean'
for _d in [
    _ENV,
    os.path.join(_ENV, 'Library', 'bin'),
    os.path.join(_ENV, 'Library', 'mingw-w64', 'bin'),
    os.path.join(_ENV, 'Lib', 'site-packages', 'torch', 'lib'),
]:
    if os.path.exists(_d):
        os.add_dll_directory(_d)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONNOUSERSITE"] = "1"

from pathlib import Path
from ultralytics import YOLO

BASE     = Path(__file__).parent
YAML     = BASE / "data" / "yolo" / "rdd2022.yaml"
RUNS_DIR = BASE / "runs"
RUN_NAME = "rdd2022_yolo11m"

print(f"YAML  : {YAML}  (exists={YAML.exists()})")

import torch
print(f"Torch : {torch.__version__}")
print(f"CUDA  : {torch.cuda.is_available()} — {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print()

model = YOLO("yolo11m.pt")

results = model.train(
    data        = str(YAML.as_posix()),
    epochs      = 50,
    imgsz       = 640,
    batch       = 16,
    workers     = 4,
    device      = 0,
    project     = str(RUNS_DIR.as_posix()),
    name        = RUN_NAME,
    patience    = 10,
    save_period = 10,
    val         = True,
    plots       = True,
    exist_ok    = True,
    optimizer   = "AdamW",
    lr0         = 0.001,
    verbose     = True,
)

print(f"\nBeste Gewichte: {results.save_dir}/weights/best.pt")
