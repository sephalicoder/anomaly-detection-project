import sys
import os

# Explicitly tell Python to look inside the src folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

import torchvision.transforms as transforms
from dataset import MVTecBottleDataset  

# Define basic transform to resize to 128x128 and convert to tensor
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

print("--- Testing Training Split ---")
train_dataset = MVTecBottleDataset(root_dir="data/bottle", split="train", transform=transform)
print(f"Loaded {len(train_dataset)} training images.")

train_img, train_label = train_dataset[0]
print(f"Train Image shape: {train_img.shape} (Expected: torch.Size([3, 128, 128]))")
print(f"Train Image label: {train_label} (Expected: 0 for normal)")

print("\n--- Testing Testing Split ---")
test_dataset = MVTecBottleDataset(root_dir="data/bottle", split="test", transform=transform)
print(f"Loaded {len(test_dataset)} testing images.")

labels_found = set()
for i in range(len(test_dataset)):
    _, lbl, _ = test_dataset[i]  # Unpacks image, label, and path
    labels_found.add(lbl)

print(f"Labels found in test split: {labels_found} (Expected: {{0, 1}})")