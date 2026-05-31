import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from pathlib import Path
import argparse

# Config
DATA_DIR = Path("hsp2_transformer/data/classification/val")
TRAIN_DIR = Path("hsp2_transformer/data/classification/train")
RESULTS_BASE_DIR = Path("hsp2_transformer/results")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_type, num_classes):
    if model_type == "vit":
        model = models.vit_b_16()
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        model_path = Path("hsp2_transformer/models/best_vit.pth")
    elif model_type == "swin":
        model = models.swin_t()
        model.head = nn.Linear(model.head.in_features, num_classes)
        model_path = Path("hsp2_transformer/models/best_swin.pth")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model file {model_path} not found.")

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    return model.to(DEVICE).eval()

def visualize_attention(model_type):
    print(f"Visualizing activity for {model_type}...")
    
    output_dir = RESULTS_BASE_DIR / model_type / "attention"
    output_dir.mkdir(parents=True, exist_ok=True)

    classes = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir()])
    num_classes = len(classes)
    
    model = load_model(model_type, num_classes)

    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Hook to capture data
    captured_data = {}
    def get_hook(name):
        def hook(module, input, output):
            captured_data[name] = output.detach()
        return hook

    if model_type == "vit":
        # Capture from the last encoder block's output
        target_layer = model.encoder.layers[-1]
        handle = target_layer.register_forward_hook(get_hook("feat"))
    elif model_type == "swin":
        # Capture from the last stage output
        target_layer = model.features[-1]
        handle = target_layer.register_forward_hook(get_hook("feat"))

    # Get some sample images
    all_images = list(DATA_DIR.rglob("*.jpg"))
    samples = np.random.choice(all_images, min(10, len(all_images)), replace=False)

    for img_path in samples:
        img_pill = Image.open(img_path).convert("RGB")
        img_tensor = transform(img_pill).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            _ = model(img_tensor)
        
        if "feat" not in captured_data: continue
        feat = captured_data["feat"]
        
        if model_type == "vit":
            # feat shape: [1, 197, 768] (1 CLS + 196 patches)
            # Remove CLS token and reshape
            patches = feat[0, 1:].reshape(14, 14, 768)
            heatmap = torch.norm(patches, dim=-1).cpu().numpy()
        else: # swin
            # Swin features after last block: [1, 7, 7, 768]
            heatmap = torch.norm(feat[0], dim=-1).cpu().numpy()

        # Normalize and Resize
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        heatmap = cv2.resize(heatmap, (224, 224))
        
        # Plot
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img_pill.resize((224, 224)))
        plt.title(f"Original: {img_path.parent.name}")
        plt.axis("off")
        
        plt.subplot(1, 2, 2)
        plt.imshow(img_pill.resize((224, 224)))
        plt.imshow(heatmap, cmap='jet', alpha=0.5)
        plt.title(f"{model_type.upper()} Activity Map")
        plt.axis("off")
        
        plt.savefig(output_dir / f"attention_{img_path.stem}.png")
        plt.close()

    handle.remove()
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="vit", choices=["vit", "swin"])
    args = parser.parse_args()
    
    visualize_attention(args.model)
