"""
Baseline anomaly detector using PCA reconstruction error.
"""
import sys
import os

# Fix path resolution so imports work from any execution directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from dataloaders import get_dataloaders


def images_to_flat_array(loader):
    """Flatten each image [C,H,W] into a 1D vector for PCA."""
    all_imgs, all_labels = [], []
    for images, labels, paths in loader:
        flat = images.view(images.size(0), -1).numpy()  # [B, C*H*W]
        all_imgs.append(flat)
        all_labels.append(labels.numpy())
    return np.concatenate(all_imgs), np.concatenate(all_labels)


def run_pca_baseline(category="bottle", image_size=128, n_components=50):
    # Adjust root dir path relative to script location
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
    if not os.path.exists(root_dir):
        root_dir = "data"
        
    train_loader, test_loader = get_dataloaders(root_dir=root_dir, category=category, image_size=image_size, batch_size=32)

    X_train, _ = images_to_flat_array(train_loader)
    X_test, y_test = images_to_flat_array(test_loader)

    print(f"Fitting PCA with {n_components} components on {X_train.shape[0]} normal images...")
    pca = PCA(n_components=n_components)
    pca.fit(X_train)

    # Reconstruct test images and compute per-image MSE
    X_test_proj = pca.transform(X_test)
    X_test_recon = pca.inverse_transform(X_test_proj)

    errors = np.mean((X_test - X_test_recon) ** 2, axis=1)  # per-image scalar score

    auroc = roc_auc_score(y_test, errors)
    print(f"📊 PCA baseline AUROC ({category}, n_components={n_components}): {auroc:.4f}")
    return auroc


if __name__ == "__main__":
    run_pca_baseline(category="bottle", n_components=50)