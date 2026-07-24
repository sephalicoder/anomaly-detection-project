import sys
import os

# Explicitly tell Python to look inside the src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

import torch
from model import ConvAutoencoder

# Auto-detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Instantiate model
model = ConvAutoencoder().to(device)

# Fake batch of 4 random images
dummy_input = torch.randn(4, 3, 128, 128).to(device)

# Unpack the tuple returned by your forward pass
reconstruction, latent = model(dummy_input)

print("Input shape:        ", dummy_input.shape)
print("Latent shape:        ", latent.shape)
print("Reconstruction shape:", reconstruction.shape)

assert reconstruction.shape == dummy_input.shape, "Shape mismatch! Reconstruction must match input."
print("✅ Shapes match. Model architecture is wired correctly.")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total trainable parameters: {total_params:,}")