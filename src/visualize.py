"""
Generates side-by-side panels: Original | Reconstruction | Anomaly Heatmap
for a handful of test images (mix of normal and defective).
"""
import sys
import os

# Ensure Python looks inside the src directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import torch
import numpy as np
import matplotlib.pyplot as plt
from dataloaders import get_dataloaders
from model import ConvAutoencoder


def visualize_samples(category="bottle", checkpoint_dir="checkpoints",
                      image_size=128, num_samples=4, out_path="heatmaps.png"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = os.path.join(checkpoint_dir, f"{category}_autoencoder.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)

    # Initialize the architecture matching your checked implementation
    model = ConvAutoencoder().to(device)
    
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()

    # Load a test batch
    _, test_loader = get_dataloaders(category=category, image_size=image_size, batch_size=8)
    images, labels, paths = next(iter(test_loader))

    idxs = list(range(min(num_samples, len(images))))

    with torch.no_grad():
        images_dev = images.to(device)
        # Unpack the tuple (reconstruction, latent) returned by your model
        reconstructions, _ = model(images_dev)
        reconstructions = reconstructions.cpu()

    fig, axes = plt.subplots(len(idxs), 3, figsize=(9, 3 * len(idxs)))
    if len(idxs) == 1:
        axes = axes.reshape(1, -1)

    for row, i in enumerate(idxs):
        orig = images[i]
        recon = reconstructions[i]
        
        # Inline absolute error map computation averaged over RGB channels
        error_map = torch.abs(orig - recon).numpy()
        error_map = np.mean(error_map, axis=0)

        # HWC formatting for plotting
        orig_np = orig.permute(1, 2, 0).numpy()
        recon_np = np.clip(recon.permute(1, 2, 0).numpy(), 0, 1)

        status = "Anomaly" if labels[i].item() == 1 else "Normal"

        axes[row, 0].imshow(orig_np)
        axes[row, 0].set_title(f"Original ({status})")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(recon_np)
        axes[row, 1].set_title("Reconstruction")
        axes[row, 1].axis("off")

        # Overlay heatmap over original image
        axes[row, 2].imshow(orig_np)
        axes[row, 2].imshow(error_map, cmap="jet", alpha=0.5)
        axes[row, 2].set_title("Anomaly Heatmap")
        axes[row, 2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved heatmap visualization to {out_path}")


if __name__ == "__main__":
    visualize_samples(category="bottle")