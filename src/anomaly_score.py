# src/anomaly_score.py
"""
Computes anomaly scores for test images using a trained autoencoder.

Two complementary metrics:
  - MSE:  raw pixel-wise squared error. Sensitive to any pixel difference,
          including lighting/noise — can cause false positives.
  - SSIM: Structural Similarity Index. Compares local structure/luminance/contrast
          in patches rather than raw pixels — more robust to minor lighting shifts,
          better at catching structural distortions (cracks, dents, missing parts).

We compute a PER-PIXEL error map (for heatmaps) AND a single scalar
per-image score (for image-level classification / AUROC).
"""
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim


def compute_mse_map(original, reconstruction):
    """
    original, reconstruction: torch tensors [C, H, W], values in [0,1]
    Returns: 2D numpy array [H, W] of per-pixel squared error, averaged over channels.
    """
    diff = (original - reconstruction) ** 2
    error_map = diff.mean(dim=0)  # average over RGB channels -> [H, W]
    return error_map.cpu().numpy()


def compute_ssim_map(original, reconstruction):
    """
    original, reconstruction: torch tensors [C, H, W], values in [0,1]
    Returns: 2D numpy array [H, W] of (1 - SSIM) per pixel (so higher = more anomalous,
    consistent with the MSE map's convention).
    """
    orig_np = original.permute(1, 2, 0).cpu().numpy()   # H,W,C
    recon_np = reconstruction.permute(1, 2, 0).cpu().numpy()

    # full=True returns the full per-pixel SSIM map, not just the scalar mean
    score, ssim_map = ssim(orig_np, recon_np, channel_axis=2, full=True, data_range=1.0)
    anomaly_map = 1 - ssim_map.mean(axis=2)  # average channels, invert so high=anomalous
    return anomaly_map


def compute_anomaly_scores(model, dataloader, device):
    """
    Runs the model over a dataloader and returns:
      - image_scores: list of scalar anomaly scores (one per image) using MSE
      - ssim_scores:  list of scalar anomaly scores (one per image) using SSIM
      - labels:       list of true labels (0=normal, 1=anomaly)
      - paths:        list of image file paths (for later visualization)
    """
    model.eval()
    image_scores, ssim_scores, labels_out, paths_out = [], [], [], []

    with torch.no_grad():
        for images, labels, paths in dataloader:
            images = images.to(device)
            reconstructions, _ = model(images)

            for i in range(images.size(0)):
                orig = images[i]
                recon = reconstructions[i]

                mse_map = compute_mse_map(orig, recon)
                ssim_map = compute_ssim_map(orig, recon)

                # Scalar score per image = mean error over all pixels.
                # (We could also use max, but mean is more stable/less noise-sensitive.)
                image_scores.append(mse_map.mean())
                ssim_scores.append(ssim_map.mean())
                labels_out.append(labels[i].item())
                paths_out.append(paths[i])

    return np.array(image_scores), np.array(ssim_scores), np.array(labels_out), paths_out