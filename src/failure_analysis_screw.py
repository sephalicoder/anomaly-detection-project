"""
Isolates and prints the specific structural failure cases for the screw category
using the exact operational target-recall threshold logic.
"""
import sys
import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import recall_score
from dataloaders import get_dataloaders
from model import ConvAutoencoder

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

def analyze_screw_failures(target_recall=0.95):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ckpt_path = "checkpoints/screw_autoencoder.pt"
    if not os.path.exists(ckpt_path):
        print("❌ Screw checkpoint not found! Please run the training benchmark first.")
        return

    checkpoint = torch.load(ckpt_path, map_location=device)
    model = ConvAutoencoder().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    if not os.path.exists(root_dir):
        root_dir = "data"

    _, test_loader = get_dataloaders(root_dir=root_dir, category="screw", image_size=128, batch_size=1)
    
    criterion = nn.MSELoss(reduction='none')
    scores, labels, paths = [], [], []

    print("Extracting test scores for failure diagnostics...")
    with torch.no_grad():
        for images, batch_labels, batch_paths in test_loader:
            images = images.to(device)
            reconstructed, _ = model(images)
            loss = criterion(reconstructed, images)
            img_score = torch.mean(loss).cpu().item()
            
            scores.extend([img_score])
            labels.extend(batch_labels.numpy())
            paths.extend(batch_paths)

    scores = np.array(scores)
    labels = np.array(labels)

    # --- Replicate strict validation split logic to determine operational threshold ---
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(scores))
    half = len(idx) // 2
    val_idx, hold_idx = idx[:half], idx[half:]

    val_scores, val_labels = scores[val_idx], labels[val_idx]
    hold_scores, hold_labels = scores[hold_idx], labels[hold_idx]
    hold_paths = [paths[i] for i in hold_idx]

    # Target threshold from split
    candidate_thresholds = np.sort(val_scores[val_labels == 1])
    best_thresh = candidate_thresholds[0]
    for t in candidate_thresholds:
        preds = (val_scores >= t).astype(int)
        r = recall_score(val_labels, preds, zero_division=0)
        if r >= target_recall:
            best_thresh = t
        else:
            break

    hold_preds = (hold_scores >= best_thresh).astype(int)

    print(f"\n=== 🔍 SCREW QUALITATIVE FAILURE BOUNDARY ANALYSIS ===")
    print(f"Operational Boundary Threshold: {best_thresh:.6f}")
    
    # Extract structural failure classes
    false_negatives = [(hold_paths[i], hold_scores[i]) for i in range(len(hold_labels))
                        if hold_labels[i] == 1 and hold_preds[i] == 0]
    false_positives = [(hold_paths[i], hold_scores[i]) for i in range(len(hold_labels))
                        if hold_labels[i] == 0 and hold_preds[i] == 1]

    print(f"\n❌ Missed Defects / False Negatives (Count: {len(false_negatives)}):")
    for p, s in false_negatives[:4]:
        parts = p.split(os.sep)
        display_path = os.path.join(parts[-2], parts[-1]) if len(parts) > 1 else p
        print(f"  - {display_path} (score={s:.6f})")

    print(f"\n⚠️ Normal Flagged as Defective / False Positives (Count: {len(false_positives)}):")
    for p, s in false_positives[:4]:
        print(f"  - {os.path.basename(p)} (score={s:.6f})")

if __name__ == "__main__":
    analyze_screw_failures()