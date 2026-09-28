import sys
import torch


print("=" * 60)
print("GPU / CUDA Environment")
print("=" * 60)
print("Python version :", sys.version.split()[0])
print("PyTorch version:", torch.__version__)
print("CUDA version   :", torch.version.cuda)
print("CUDA available :", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Stop the lab and check PyTorch installation.")

print("GPU            :", torch.cuda.get_device_name(0))
print("GPU count      :", torch.cuda.device_count())

x = torch.randn(100, 100, device="cuda")
y = x @ x
torch.cuda.synchronize()

print("Result device  :", y.device)
print("Result shape   :", tuple(y.shape))
print("CUDA test      : SUCCESS")
