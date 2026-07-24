# src/model.py
"""
Convolutional Autoencoder for anomaly detection.

Architecture idea:
  Encoder: progressively downsample the image (128 -> 64 -> 32 -> 16 -> 8)
           while increasing channel depth, ending in a compressed latent vector.
  Decoder: mirror the encoder in reverse using transposed convolutions,
           upsampling back to the original 128x128 resolution.

The bottleneck (latent space) is the key design lever:
  - Too large -> model can "cheat" and memorize/reconstruct anomalies too (bad).
  - Too small -> model can't even reconstruct normal images well (underfitting).
"""
import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=128, image_size=128):
        super().__init__()
        self.image_size = image_size

        # ----- ENCODER -----
        # Each block: Conv2d (downsample by stride=2) -> BatchNorm -> ReLU
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),   # 128 -> 64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 64 -> 32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 32 -> 16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),# 16 -> 8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Flatten spatial map (256 channels x 8 x 8) into a latent vector
        self.flatten_dim = 256 * 8 * 8
        self.fc_enc = nn.Linear(self.flatten_dim, latent_dim)

        # ----- DECODER -----
        self.fc_dec = nn.Linear(latent_dim, self.flatten_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), # 8 -> 16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 16 -> 32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 32 -> 64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),    # 64 -> 128
            nn.Sigmoid(),  # squashes output to [0,1] to match input pixel range
        )

    def forward(self, x):
        # Encode
        z = self.encoder(x)                      # [B, 256, 8, 8]
        z_flat = z.view(z.size(0), -1)            # [B, 256*8*8]
        latent = self.fc_enc(z_flat)              # [B, latent_dim]

        # Decode
        z_dec = self.fc_dec(latent)               # [B, 256*8*8]
        z_dec = z_dec.view(-1, 256, 8, 8)          # reshape back to spatial map
        reconstruction = self.decoder(z_dec)       # [B, 3, 128, 128]

        return reconstruction, latent