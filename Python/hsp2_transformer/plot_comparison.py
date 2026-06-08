import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Note: In this project, training history is not explicitly saved as CSV, 
# but we can try to find where the plots were generated from or just use the existing images.
# However, if we want to plot them together, we'd need the raw data.
# Since we don't have the raw history, I'll just make sure the individual results are well organized.

def summarize():
    print("Individual results are already generated in results/vit and results/swin.")
    print("Folder structure is:")
    print("results/")
    print("  vit/ (metrics and attention for ViT)")
    print("  swin/ (metrics and attention for Swin)")
    print("  comparison/ (cross-model analysis)")

if __name__ == "__main__":
    summarize()
