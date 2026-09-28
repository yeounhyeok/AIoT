# ============================================================
# latency_toy.py
# Week 4 - Toy Inference Latency Measurement
# ============================================================

import time
import numpy as np

import torch
import torch.nn as nn


# ============================================================
# 1. Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("Latency Measurement")
print("=" * 60)

print("Device :", device)

if device.type == "cuda":
    print("GPU    :", torch.cuda.get_device_name(0))

print()


# ============================================================
# 2. Toy CNN
# ============================================================

class ToyCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(
                16 * 14 * 14,
                10,
            ),
        )

    def forward(self, x):
        return self.model(x)


model = ToyCNN().to(device)
model.eval()


# ============================================================
# 3. Dummy Input
# ============================================================

# Single-image inference
x = torch.randn(
    1,
    1,
    28,
    28,
    device=device,
)


# ============================================================
# 4. Warm-up
# ============================================================

WARMUP = 50

with torch.no_grad():
    for _ in range(WARMUP):
        _ = model(x)

if device.type == "cuda":
    torch.cuda.synchronize()


# ============================================================
# 5. Latency Measurement
# ============================================================

REPEAT = 200
latencies = []

with torch.no_grad():
    for _ in range(REPEAT):

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        _ = model(x)

        if device.type == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()

        latency_ms = (
            end - start
        ) * 1000

        latencies.append(latency_ms)


# ============================================================
# 6. Results
# ============================================================

mean_latency = np.mean(latencies)
median_latency = np.median(latencies)
std_latency = np.std(latencies)

print("=" * 60)
print("Single-image Inference Latency")
print("=" * 60)

print(f"Mean   : {mean_latency:.4f} ms")
print(f"Median : {median_latency:.4f} ms")
print(f"Std    : {std_latency:.4f} ms")

print("=" * 60)
