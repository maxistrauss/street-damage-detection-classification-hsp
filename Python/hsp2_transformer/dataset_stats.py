import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

# Paths
DATA_DIR = Path("data/classification/train")

def generate_stats():
    print("Gathering dataset statistics...")
    
    if not DATA_DIR.exists():
        print(f"Error: Directory {DATA_DIR} not found.")
        return
        
    class_counts = {}
    for class_dir in DATA_DIR.iterdir():
        if class_dir.is_dir():
            count = len(list(class_dir.glob("*.jpg")))
            class_counts[class_dir.name] = count
            
    df = pd.DataFrame(list(class_counts.items()), columns=['Class', 'Count'])
    df = df.sort_values(by='Count', ascending=False)
    
    total = df['Count'].sum()
    df['Percentage'] = (df['Count'] / total) * 100
    
    print("\nDataset Composition:")
    print(df.to_string(index=False))
    
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    ax = sns.barplot(x='Count', y='Class', data=df, palette='viridis')
    
    for i, v in enumerate(df['Count']):
        ax.text(v + 100, i, f"{v} ({df['Percentage'].iloc[i]:.1f}%)", color='black', va='center')
        
    plt.title('Class Distribution: RDD2022 Classification Patches (Train Set)')
    plt.xlabel('Number of Images')
    plt.ylabel('Damage Class')
    plt.tight_layout()
    
    save_path = Path("results/dataset_distribution.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    print(f"\nDistribution plot saved to {save_path}")

if __name__ == "__main__":
    generate_stats()
