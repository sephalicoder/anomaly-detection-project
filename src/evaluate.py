import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Explicitly point Python to look inside the src directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from dataset import MVTecDataset
from model import ConvAutoencoder

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📊 Running evaluation on device: {device}")
    
    # 1. Setup Test Data (Test split contains both normal and anomaly folders)
    test_dataset = MVTecDataset(root_dir="data", category="bottle", split="test", image_size=128)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # 2. Load the Trained Model Weights
    model = ConvAutoencoder().to(device)
    model_path = "checkpoints/bottle_autoencoder.pt"
    
    # Load the full checkpoint dictionary safely
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract the actual model weights from the dictionary if wrapped
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    
    criterion = nn.MSELoss(reduction='none') # Compute raw per-pixel errors
    
    gt_labels = []
    anomaly_scores = []
    
    print("Processing test set images...")
    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            
            # Forward Pass
            reconstructed, _ = model(images)
            
            # Anomaly Score = Mean Squared Error across all pixels
            loss = criterion(reconstructed, images)
            img_score = torch.mean(loss).cpu().item()
            
            anomaly_scores.append(img_score)
            gt_labels.append(labels.item())
            
    gt_labels = np.array(gt_labels)
    anomaly_scores = np.array(anomaly_scores)
    
    # 3. Calculate Performance Metrics
    auroc = roc_auc_score(gt_labels, anomaly_scores)
    print(f"\n🚀 Evaluation Metrics:")
    print(f"🥇 Image-level AUROC: {auroc:.4f}")
    
    # 4. Plot and Save the ROC Curve
    fpr, tpr, _ = roc_curve(gt_labels, anomaly_scores)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auroc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) - Bottle')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/roc_curve.png")
    print("📈 Saved ROC curve plot to 'results/roc_curve.png'")

if __name__ == "__main__":
    evaluate()