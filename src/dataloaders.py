# src/dataloaders.py
from torch.utils.data import DataLoader
from dataset import MVTecDataset


def get_dataloaders(root_dir="data", category="bottle", image_size=128, batch_size=16):
    train_ds = MVTecDataset(root_dir, category, split="train", image_size=image_size)
    test_ds = MVTecDataset(root_dir, category, split="test", image_size=image_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader