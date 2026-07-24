# src/train.py
"""
Training loop for the Convolutional Autoencoder.
Trains ONLY on normal (good) images, minimizing reconstruction error.
Includes checkpointing so a CPU run that gets interrupted can resume.
"""
import sys, os
sys.path.append("src")

import torch
import torch.nn as nn
from dataloaders import get_dataloaders
from model import ConvAutoencoder

def train(category="bottle", image_size=128, batch_size=16,
          epochs=50, lr=1e-3, latent_dim=128,
          checkpoint_dir="checkpoints"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"{category}_autoencoder.pt")

    train_loader, _ = get_dataloaders(category=category, image_size=image_size, batch_size=batch_size)

    model = ConvAutoencoder(latent_dim=latent_dim, image_size=image_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()  # pixel-wise reconstruction loss

    start_epoch = 0
    best_loss = float("inf")

    # --- Resume from checkpoint if one exists ---
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint["best_loss"]
        print(f"Resumed from epoch {start_epoch}, best_loss so far: {best_loss:.6f}")

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss = 0.0

        for images, labels, paths in train_loader:
            images = images.to(device)

            optimizer.zero_grad()
            reconstructions, _ = model(images)
            loss = criterion(reconstructions, images)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{epochs}]  Loss: {epoch_loss:.6f}")

        # --- Save checkpoint every epoch (so a crash never costs more than 1 epoch) ---
        is_best = epoch_loss < best_loss
        if is_best:
            best_loss = epoch_loss

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_loss": best_loss,
            "category": category,
            "image_size": image_size,
            "latent_dim": latent_dim,
        }, ckpt_path)

    print(f"\nTraining complete. Final model saved to {ckpt_path}")
    return model


if __name__ == "__main__":
    train(category="bottle", epochs=50, batch_size=16)