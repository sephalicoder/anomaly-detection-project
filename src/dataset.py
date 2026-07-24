# src/dataset.py
"""
Custom PyTorch Dataset for MVTec-AD.

Two modes:
  - "train": loads ONLY train/good images (normal samples), with augmentation
  - "test":  loads ALL test images across every defect subfolder + "good",
             and returns a label (0=normal, 1=anomaly) for evaluation
"""
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MVTecDataset(Dataset):
    def __init__(self, root_dir, category, split="train", image_size=128):
        """
        root_dir: path to 'data' folder
        category: e.g. 'bottle'
        split: 'train' or 'test'
        image_size: resize target (square)
        """
        self.split = split
        self.image_size = image_size
        self.samples = []  # list of (image_path, label) where label: 0=normal, 1=anomaly
        self.defect_types = []  # parallel list: defect folder name (or 'good')

        cat_path = os.path.join(root_dir, category)

        if split == "train":
            good_dir = os.path.join(cat_path, "train", "good")
            for fname in sorted(os.listdir(good_dir)):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((os.path.join(good_dir, fname), 0))
                    self.defect_types.append("good")

        elif split == "test":
            test_dir = os.path.join(cat_path, "test")
            for defect_type in sorted(os.listdir(test_dir)):
                defect_dir = os.path.join(test_dir, defect_type)
                if not os.path.isdir(defect_dir):
                    continue
                label = 0 if defect_type == "good" else 1
                for fname in sorted(os.listdir(defect_dir)):
                    if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                        self.samples.append((os.path.join(defect_dir, fname), label))
                        self.defect_types.append(defect_type)
        else:
            raise ValueError("split must be 'train' or 'test'")

        # --- Transforms ---
        if split == "train":
            # Augmentation ONLY on training data, and only mild/realistic ones.
            # We must NOT distort the object's shape/structure too much,
            # since the autoencoder needs to learn a faithful notion of "normal".
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),  # scales pixels to [0, 1]
            ])
        else:
            # NO augmentation at test time — we want to measure true performance
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label, img_path