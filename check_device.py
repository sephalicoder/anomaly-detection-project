# check_device.py
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
else:
    print("No NVIDIA GPU detected — we'll train on CPU.")
    print("This is fine for this project if we keep images small (128x128) and batch sizes modest.")