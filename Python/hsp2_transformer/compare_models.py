import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# Config
VAL_DIR = Path("hsp2_transformer/data/classification/val")
VIT_PATH = Path("hsp2_transformer/models/best_vit.pth")
SWIN_PATH = Path("hsp2_transformer/models/best_swin.pth")
RESULTS_DIR = Path("hsp2_transformer/results/comparison")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_vit(num_classes):
    model = models.vit_b_16()
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    model.load_state_dict(torch.load(VIT_PATH, map_location=DEVICE))
    return model.to(DEVICE).eval()

def load_swin(num_classes):
    model = models.swin_t()
    model.head = nn.Linear(model.head.in_features, num_classes)
    model.load_state_dict(torch.load(SWIN_PATH, map_location=DEVICE))
    return model.to(DEVICE).eval()

def compare():
    print(f"Comparing models on {DEVICE}...")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_ds = datasets.ImageFolder(VAL_DIR, transform=transform)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False) # Batch size 1 for easy tracking
    classes = val_ds.classes
    
    vit = load_vit(len(classes))
    swin = load_swin(len(classes))

    failures = []
    
    print("Running inference...")
    with torch.no_grad():
        for i, (img, label) in enumerate(tqdm(val_loader)):
            img, label = img.to(DEVICE), label.to(DEVICE)
            
            # Predict ViT
            out_vit = vit(img)
            pred_vit = torch.max(out_vit, 1)[1].item()
            
            # Predict Swin
            out_swin = swin(img)
            pred_swin = torch.max(out_swin, 1)[1].item()
            
            true_label = label.item()
            
            # Track failures or disagreements
            if pred_vit != true_label or pred_swin != true_label:
                # Store sample info
                img_path, _ = val_ds.samples[i]
                failures.append({
                    'path': img_path,
                    'true': classes[true_label],
                    'vit': classes[pred_vit],
                    'swin': classes[pred_swin]
                })
            
            if len(failures) >= 100: # Stop after finding enough examples
                break

    # Visualize 10 interesting cases (where they disagree or both fail)
    print(f"Generating visualization for {min(10, len(failures))} cases...")
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    axes = axes.flatten()
    
    # Filter for cases where they disagree if possible
    disagreements = [f for f in failures if f['vit'] != f['swin']]
    plot_samples = disagreements[:10] if len(disagreements) >= 10 else failures[:10]

    for idx, case in enumerate(plot_samples):
        img = Image.open(case['path'])
        axes[idx].imshow(img)
        
        color_vit = 'green' if case['vit'] == case['true'] else 'red'
        color_swin = 'green' if case['swin'] == case['true'] else 'red'
        
        title = f"True: {case['true']}\nViT: {case['vit']}\nSwin: {case['swin']}"
        axes[idx].set_title(title, fontsize=9)
        axes[idx].axis('off')
        
        # Highlight text based on correctness in a real report would be better, 
        # but here we just put the text.

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "error_analysis_samples.png")
    print(f"Visualization saved to {RESULTS_DIR / 'error_analysis_samples.png'}")

    # Summary Statistics
    vit_correct = sum(1 for f in failures if f['vit'] == f['true'])
    swin_correct = sum(1 for f in failures if f['swin'] == f['true'])
    print(f"\nSubset Analysis (first 100 samples with errors):")
    print(f"ViT correct: {vit_correct}/100")
    print(f"Swin correct: {swin_correct}/100")

if __name__ == "__main__":
    compare()
