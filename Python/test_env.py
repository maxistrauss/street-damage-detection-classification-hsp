import os
os.environ['PYTHONNOUSERSITE'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from ultralytics import YOLO
print('torch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('VRAM:', round(torch.cuda.get_device_properties(0).total_memory/1e9, 1), 'GB')
print('All imports OK')
