import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import argparse

# Config
DATA_DIR = Path("hsp2_transformer/data/classification/val")
TRAIN_DIR = Path("hsp2_transformer/data/classification/train")
RESULTS_BASE_DIR = Path("hsp2_transformer/results")

BATCH_SIZE = 64
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

def evaluate(model_type):
    print(f"Evaluating {model_type} on {DEVICE}...")
    
    results_dir = RESULTS_BASE_DIR / model_type
    results_dir.mkdir(exist_ok=True, parents=True)

    # Data Loading
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    classes = sorted([d.name for d in TRAIN_DIR.iterdir() if d.is_dir()])
    num_classes = len(classes)
    
    test_ds = datasets.ImageFolder(DATA_DIR, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = load_model(model_type, num_classes)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc=f"Testing {model_type}"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Classification Report
    report = classification_report(all_labels, all_preds, target_names=classes)
    print(f"\nClassification Report ({model_type}):")
    print(report)
    
    with open(results_dir / "classification_report.txt", "w") as f:
        f.write(report)

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix: {model_type.upper()}')
    plt.savefig(results_dir / "confusion_matrix.png")
    plt.close()
    print(f"Results saved to {results_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="vit", choices=["vit", "swin"])
    args = parser.parse_args()
    
    evaluate(args.model)