"""
Self-contained script to train and evaluate multiple categories sequentially.
Optimized for high-fidelity 50-epoch benchmarking.
"""
import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score

# Ensure local imports work cleanly from any execution directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from dataloaders import get_dataloaders
from model import ConvAutoencoder

# Configured for complete multi-category benchmarking
CATEGORIES = ["bottle", "screw"]

def run_pipeline_for_category(cat, epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Establish dynamic data directory pathing
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    if not os.path.exists(root_dir):
        root_dir = "data"
        
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = f"checkpoints/{cat}_autoencoder.pt"
    
    train_loader, test_loader = get_dataloaders(root_dir=root_dir, category=cat, image_size=128, batch_size=16)
    
    # ------------------ 1. TRAINING LOOP ------------------
    print(f"🏋️ Starting high-fidelity training on normal '{cat}' images...")
    model = ConvAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, _, _ in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            reconstructions, _ = model(images)
            loss = criterion(reconstructions, images)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Log metrics frequently for clear tracking over 50 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == (epochs - 1):
            print(f"  Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.6f}")
            
    # Save the finalized structural weights dict
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
    print(f"💾 Checkpoint successfully saved to {ckpt_path}")
    
    # ------------------ 2. STATISTICAL EVALUATION ------------------
    print(f"📊 Running evaluation mapping for '{cat}'...")
    model.eval()
    eval_criterion = nn.MSELoss(reduction='none')
    
    gt_labels = []
    anomaly_scores = []
    
    # Force batch_size=1 to cleanly isolate specific sample metrics
    _, eval_loader = get_dataloaders(root_dir=root_dir, category=cat, image_size=128, batch_size=1)
    
    with torch.no_grad():
        for images, labels, _ in eval_loader:
            images = images.to(device)
            reconstructed, _ = model(images)
            loss = eval_criterion(reconstructed, images)
            img_score = torch.mean(loss).cpu().item()
            
            anomaly_scores.append(img_score)
            gt_labels.append(labels.item())
            
    auroc = roc_auc_score(np.array(gt_labels), np.array(anomaly_scores))
    print(f"🥇 {cat.upper()} Optimized AUROC Score: {auroc:.4f}\n")
    return auroc

def main():
    results = {}
    print(f"{'='*50}\n🚀 Launching Deep Autoencoder 50-Epoch Benchmark\n{'='*50}")
    
    for cat in CATEGORIES:
        print(f"\n▶️ Running production loop for category: {cat.upper()}")
        try:
            auroc = run_pipeline_for_category(cat, epochs=50)
            results[cat] = auroc
        except Exception as e:
            print(f"❌ Pipeline execution block failed for {cat}: {e}")
            
    # --- Print Executive Production Summary Table ---
    print(f"\n{'='*50}\n📊 FINAL PRODUCTION BENCHMARK SUMMARY (50 EPOCHS)\n{'='*50}")
    print(f"{'Category':<15}{'Model AUROC':<12}")
    print("-" * 30)
    for cat, auroc in results.items():
        print(f"{cat:<15}{auroc:<12.4f}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()