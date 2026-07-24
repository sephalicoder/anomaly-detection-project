# Industrial Anomaly Detection — Unsupervised Convolutional Autoencoder

An unsupervised anomaly detection pipeline for industrial defect inspection, built on the MVTec-AD benchmark. Instead of training a classifier on labeled defects (rare and expensive to collect), this project trains a convolutional autoencoder exclusively on **normal** parts, then flags anomalies via reconstruction error.

🔗 **Live demo:** https://huggingface.co/spaces/shifu01/industrial-anomaly

## Problem

Industrial defect detection suffers from severe class imbalance — thousands of normal parts, almost no labeled defective ones. Standard supervised classifiers fail under this constraint. This project reframes the problem as one-class anomaly detection: learn what "normal" looks like, and flag anything that doesn't reconstruct well.

## Approach

1. Train a convolutional autoencoder on `train/good` images only (no defects seen during training)
2. At inference, compute per-pixel reconstruction error (MSE) between input and output
3. Aggregate into an image-level anomaly score; overlay per-pixel error as a localization heatmap
4. Evaluate using image-level AUROC against ground-truth labels

## Results

| Category | Image-level AUROC | Notes |
|---|---|---|
| bottle | 0.9587 (50 epochs) | Simple, rigid, uniform background |
| screw | 0.7336 (10 epochs) | Small object, fine-grained defects — harder |

**Baseline comparison:** A PCA-based reconstruction baseline was implemented for the same task, to isolate the contribution of learned non-linear (CNN) features over a linear projection. [Fill in your actual PCA AUROC number here once you run `baseline_pca.py`.]

**Key finding:** Performance is not uniform across categories — objects with small, fine-grained defect regions (screw) are inherently harder for this architecture at 128×128 resolution than larger, more uniform objects (bottle), since the anomalous region occupies a smaller fraction of the reconstruction error signal.

## Architecture

- Convolutional encoder: 4 stride-2 conv blocks (BatchNorm + ReLU), downsampling 128→8
- Fully-connected bottleneck (128-dim latent space) — forces a global, compressed representation of "normal"
- Mirrored decoder using transposed convolutions, Sigmoid output activation
- ~5.6M trainable parameters
- Trained with Adam optimizer, MSE reconstruction loss

## Project structure