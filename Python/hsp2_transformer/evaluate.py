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

# Config
DATA_DIR = Path("hsp2_transformer/data/classification/val")
MODEL_PATH = Path("hsp2_transformer/models/best_vit.pth")
RESULTS_DIR = Path("hsp2_transformer/results")
RESULTS_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate():
    print(f"Evaluating on {DEVICE}...")
    
    # Data Loading
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Important: Ensure the test classes match the train classes
    train_dir = Path("hsp2_transformer/data/classification/train")
    classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    num_classes = len(classes)
    print(f"Detected {num_classes} classes: {classes}")

    test_ds = datasets.ImageFolder(DATA_DIR, transform=transform)
    # Filter test_ds to only include classes that were in training
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Load Model
    model = models.vit_b_16()
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    
    if not MODEL_PATH.exists():
        print(f"Error: Model file {MODEL_PATH} not found. Training might still be in progress.")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Testing"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Classification Report
    report = classification_report(all_labels, all_preds, target_names=classes)
    print("\nClassification Report:")
    print(report)
    
    with open(RESULTS_DIR / "classification_report.txt", "w") as f:
        f.write(report)

    # F1 Score
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"Weighted F1 Score: {f1:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix: Vision Transformer (ViT)')
    plt.savefig(RESULTS_DIR / "vit_confusion_matrix.png")
    print(f"Confusion Matrix saved to {RESULTS_DIR / 'vit_confusion_matrix.png'}")

if __name__ == "__main__":
    evaluate()
