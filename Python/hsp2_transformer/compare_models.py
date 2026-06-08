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
TEST_DIR = Path("data/classification/test")
VIT_PATH = Path("models/best_vit.pth")
SWIN_PATH = Path("models/best_swin.pth")
RESULTS_DIR = Path("results/comparison")
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

    val_ds = datasets.ImageFolder(TEST_DIR, transform=transform)
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
            prob_vit = torch.nn.functional.softmax(out_vit, dim=1)
            conf_vit, pred_vit_idx = torch.max(prob_vit, 1)
            pred_vit = pred_vit_idx.item()
            conf_vit = conf_vit.item()
            
            # Predict Swin
            out_swin = swin(img)
            prob_swin = torch.nn.functional.softmax(out_swin, dim=1)
            conf_swin, pred_swin_idx = torch.max(prob_swin, 1)
            pred_swin = pred_swin_idx.item()
            conf_swin = conf_swin.item()
            
            true_label = label.item()
            
            # Track failures or disagreements
            if pred_vit != true_label or pred_swin != true_label:
                # Store sample info
                img_path, _ = val_ds.samples[i]
                failures.append({
                    'path': img_path,
                    'true': classes[true_label],
                    'vit': classes[pred_vit],
                    'vit_conf': conf_vit,
                    'swin': classes[pred_swin],
                    'swin_conf': conf_swin
                })
            
            if len(failures) >= 100: # Stop after finding enough examples
                break

    # Visualize 10 interesting cases (where they disagree or both fail)
    print(f"Generating visualization for {min(10, len(failures))} cases...")
    
    fig, axes = plt.subplots(2, 5, figsize=(25, 12))
    axes = axes.flatten()
    
    # Filter for cases where they disagree if possible
    disagreements = [f for f in failures if f['vit'] != f['swin']]
    plot_samples = disagreements[:10] if len(disagreements) >= 10 else failures[:10]

    for idx, case in enumerate(plot_samples):
        img = Image.open(case['path'])
        axes[idx].imshow(img)
        
        title = f"True: {case['true']}\nViT: {case['vit']} ({case['vit_conf']:.2%})\nSwin: {case['swin']} ({case['swin_conf']:.2%})"
        axes[idx].set_title(title, fontsize=10)
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
