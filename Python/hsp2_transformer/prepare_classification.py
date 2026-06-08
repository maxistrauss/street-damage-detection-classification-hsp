import json
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Paths
DATASET_DIR = Path("../data/dataset-ninja/rdd2022")
OUTPUT_DIR = Path("data/classification")

CLASSES = [
    "alligator crack",
    "block crack",
    "longitudinal crack",
    "other corruption",
    "pothole",
    "repair",
    "transverse crack",
    "normal"
]

TARGET_SIZE = (224, 224)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42

random.seed(SEED)

def get_crops(img_path, ann_path):
    """Extract crops for each object in the annotation."""
    if not ann_path.exists():
        return []
    
    with open(ann_path, 'r') as f:
        data = json.load(f)
    
    img_w = data['size']['width']
    img_h = data['size']['height']
    
    crops = []
    objects = data.get('objects', [])
    
    # Track occupied regions to find 'normal' patches later
    occupied_regions = []

    for i, obj in enumerate(objects):
        cls = obj.get('classTitle')
        if cls not in CLASSES:
            continue
            
        pts = obj['points']['exterior']
        x1, y1 = pts[0]
        x2, y2 = pts[1]
        
        # Add some padding (20%)
        w = x2 - x1
        h = y2 - y1
        pad_x = w * 0.1
        pad_y = h * 0.1
        
        x1_p = max(0, x1 - pad_x)
        y1_p = max(0, y1 - pad_y)
        x2_p = min(img_w, x2 + pad_x)
        y2_p = min(img_h, y2 + pad_y)
        
        occupied_regions.append((x1, y1, x2, y2))
        crops.append({
            'class': cls,
            'box': (x1_p, y1_p, x2_p, y2_p),
            'id': i
        })

    # Try to find 1 'normal' patch if the image is large enough
    if img_w > TARGET_SIZE[0] and img_h > TARGET_SIZE[1]:
        for _ in range(5): # Try 5 times to find a clean spot
            nx = random.randint(0, img_w - TARGET_SIZE[0])
            ny = random.randint(0, img_h - TARGET_SIZE[1])
            nboth = (nx, ny, nx + TARGET_SIZE[0], ny + TARGET_SIZE[1])
            
            overlap = False
            for ox1, oy1, ox2, oy2 in occupied_regions:
                # Check intersection
                if not (nboth[2] < ox1 or nboth[0] > ox2 or nboth[3] < oy1 or nboth[1] > oy2):
                    overlap = True
                    break
            
            if not overlap:
                crops.append({
                    'class': 'normal',
                    'box': nboth,
                    'id': 'n1'
                })
                break
                
    return crops

def process_split(split_name):
    print(f"Processing {split_name} split...")
    img_dir = DATASET_DIR / split_name / "img"
    ann_dir = DATASET_DIR / split_name / "ann"
    
    images = list(img_dir.glob("*.jpg"))
    
    all_samples = []
    
    for img_path in tqdm(images):
        ann_path = ann_dir / (img_path.name + ".json")
        crops = get_crops(img_path, ann_path)
        
        if not crops:
            continue
            
        # Open image once per file
        try:
            with Image.open(img_path) as img:
                for crop in crops:
                    # Crop and resize
                    box = crop['box']
                    patch = img.crop(box)
                    patch = patch.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
                    
                    # We'll save them later after splitting
                    all_samples.append({
                        'image': patch,
                        'class': crop['class'],
                        'name': f"{img_path.stem}_{crop['id']}.jpg"
                    })
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    return all_samples

def save_samples(samples, subset):
    print(f"Saving {len(samples)} samples to {subset}...")
    for sample in tqdm(samples):
        class_dir = OUTPUT_DIR / subset / sample['class'].replace(" ", "_")
        class_dir.mkdir(parents=True, exist_ok=True)
        sample['image'].save(class_dir / sample['name'], quality=95)

def main():
    # Process only the labeled Train set and split it into Train, Val, and Test
    all_samples = process_split("train")
    random.shuffle(all_samples)
    
    n = len(all_samples)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    
    train_part = all_samples[:train_end]
    val_part = all_samples[train_end:val_end]
    test_part = all_samples[val_end:]
    
    save_samples(train_part, "train")
    save_samples(val_part, "val")
    save_samples(test_part, "test")

if __name__ == "__main__":
    main()
