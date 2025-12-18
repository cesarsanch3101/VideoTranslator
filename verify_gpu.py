# verify_gpu.py
import torch

is_available = torch.cuda.is_available()
print(f"CUDA disponible: {is_available}")

if is_available:
    print(f"Nombre de GPU: {torch.cuda.get_device_name(0)}")