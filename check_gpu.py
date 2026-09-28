# ============================================================
# check_gpu.py
# Week 4 - GPU / CUDA Check
# ============================================================

import sys
import torch


print("=" * 60)
print("GPU / CUDA Environment")
print("=" * 60)

print("Python version :", sys.version.split()[0])
print("PyTorch version:", torch.__version__)
print("CUDA version   :", torch.version.cuda)
print("CUDA available :", torch.cuda.is_available())

print()

if not torch.cuda.is_available():
    print("Device         : CPU")
    raise SystemExit(
        "CUDA is not available. Check the Jetson environment."
    )

print("Device         : CUDA")
print("GPU            :", torch.cuda.get_device_name(0))
print("GPU count      :", torch.cuda.device_count())


# ============================================================
# Simple CUDA / cuBLAS Test
# ============================================================

x = torch.randn(
    100,
    100,
    device="cuda",
)

y = x @ x

torch.cuda.synchronize()

print()
print("Matrix multiplication test")
print("Result device  :", y.device)
print("Result shape   :", y.shape)
print("CUDA test      : SUCCESS")

print("=" * 60)
