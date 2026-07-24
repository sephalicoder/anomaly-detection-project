"""
Picks a business-driven operating threshold (target: 95% recall on anomalies)
and reports Precision/Recall/F1 at that threshold.
"""
import sys
import os

# Fix path resolution so imports work from any execution directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score
from dataloaders import get_dataloaders
from model import ConvAutoencoder


def threshold_analysis(category="bottle", checkpoint_dir="checkpoints", image_size=128, target_recall=0.95):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ckpt_path = os.path.join(checkpoint_dir, f"{category}_autoencoder.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Instantiate parameter-free model architecture
    model = ConvAutoencoder().to(device)
    
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Dynamic root folder tracking
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    if not os.path.exists(root_dir):
        root_dir = "data"

    _, test_loader = get_dataloaders(root_dir=root_dir, category=category, image_size=image_size, batch_size=1)
    
    criterion = nn.MSELoss(reduction='none')
    scores, labels, paths = [], [], []

    print("Extracting reconstruction scores across all test images...")
    with torch.no_grad():
        for images, batch_labels, batch_paths in test_loader:
            images = images.to(device)
            reconstructed, _ = model(images)
            
            loss = criterion(reconstructed, images)
            img_scores = torch.mean(loss, dim=[1, 2, 3]).cpu().numpy()
            
            scores.extend(img_scores)
            labels.extend(batch_labels.numpy())
            paths.extend(batch_paths)

    scores = np.array(scores)
    labels = np.array(labels)

    # --- Split test set in half: "val" for picking threshold, "holdout" for reporting ---
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(scores))
    half = len(idx) // 2
    val_idx, hold_idx = idx[:half], idx[half:]

    val_scores, val_labels = scores[val_idx], labels[val_idx]
    hold_scores, hold_labels = scores[hold_idx], labels[hold_idx]
    hold_paths = [paths[i] for i in hold_idx]

    # --- Find the threshold on the VAL split that achieves target_recall ---
    anomaly_scores = val_scores[val_labels == 1]
    candidate_thresholds = np.sort(anomaly_scores)
    best_thresh = candidate_thresholds[0]
    
    # Track metrics to choose threshold closest to target recall
    for t in candidate_thresholds:
        preds = (val_scores >= t).astype(int)
        r = recall_score(val_labels, preds, zero_division=0)
        if r >= target_recall:
            best_thresh = t
        else:
            break

    print(f"\nChosen threshold (targeting {target_recall*100:.0f}% recall on val split): {best_thresh:.6f}")

    # --- Report final metrics on the untouched HOLDOUT split ---
    hold_preds = (hold_scores >= best_thresh).astype(int)
    precision = precision_score(hold_labels, hold_preds, zero_division=0)
    recall = recall_score(hold_labels, hold_preds, zero_division=0)
    f1 = f1_score(hold_labels, hold_preds, zero_division=0)

    print(f"\n=== Holdout performance at chosen threshold ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")

    # --- Failure case analysis ---
    false_negatives = [(hold_paths[i], hold_scores[i]) for i in range(len(hold_labels))
                        if hold_labels[i] == 1 and hold_preds[i] == 0]
    false_positives = [(hold_paths[i], hold_scores[i]) for i in range(len(hold_labels))
                        if hold_labels[i] == 0 and hold_preds[i] == 1]

    print(f"\n--- False Negatives (missed defects): {len(false_negatives)} ---")
    for path, score in false_negatives[:4]:
        print(f"  {os.path.basename(path)}   score={score:.6f}")

    print(f"\n--- False Positives (normal flagged as defect): {len(false_positives)} ---")
    for path, score in false_positives[:4]:
        print(f"  {os.path.basename(path)}   score={score:.6f}")

    return best_thresh, precision, recall, f1


if __name__ == "__main__":
    threshold_analysis(category="bottle", target_recall=0.95)