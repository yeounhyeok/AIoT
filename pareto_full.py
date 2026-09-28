# ============================================================
# pareto_full.py
#
# Week 4 (확장) - Accuracy / Latency Pareto Analysis
#
# pareto_check.py 와 달리 demo 값을 쓰지 않는다.
# 네 모델 모두 실제로 학습한 가중치를 불러와서 측정한다.
#
#   TinyMLP        (Part A)
#   ResNet18
#   DenseNet121
#   EfficientNetB0
# ============================================================

import json
import os
import time

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import build_model, count_parameters


# ============================================================
# 1. Settings
# ============================================================

BATCH_SIZE = 128

LATENCY_WARMUP = 50
LATENCY_REPEAT = 200

MODEL_NAMES = [
    "TinyMLP",
    "EfficientNetB0",
    "ResNet18",
    "DenseNet121",
]


# ============================================================
# 2. Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("Device :", device)

if device.type == "cuda":
    print("GPU    :", torch.cuda.get_device_name(0))

print("=" * 60)


# ============================================================
# 3. MNIST Test Dataset
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5,),
        (0.5,),
    ),
])

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# ============================================================
# 4. Accuracy
# ============================================================

def measure_accuracy(model):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total * 100


# ============================================================
# 5. Latency (single-image)
# ============================================================

def measure_latency(model):
    model.eval()

    x = torch.randn(
        1,
        1,
        28,
        28,
        device=device,
    )

    with torch.no_grad():
        for _ in range(LATENCY_WARMUP):
            _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    latencies = []

    with torch.no_grad():
        for _ in range(LATENCY_REPEAT):

            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            _ = model(x)

            if device.type == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()

            latencies.append((end - start) * 1000)

    return (
        float(np.mean(latencies)),
        float(np.std(latencies)),
    )


# ============================================================
# 6. Measure All Models
# ============================================================

results = []

for name in MODEL_NAMES:

    weight_path = os.path.join("weights", f"{name}.pth")

    if not os.path.exists(weight_path):
        print(f"[skip] {name}: {weight_path} not found")
        continue

    model = build_model(name)

    state_dict = torch.load(
        weight_path,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    accuracy = measure_accuracy(model)

    latency_mean, latency_std = measure_latency(model)

    n_params = count_parameters(model)

    print()
    print("=" * 60)
    print(name)
    print("=" * 60)
    print(f"Params   : {n_params:,}")
    print(f"Accuracy : {accuracy:.2f} %")
    print(f"Latency  : {latency_mean:.4f} ms (std {latency_std:.4f})")

    results.append({
        "Model": name,
        "Accuracy": round(accuracy, 2),
        "Latency": round(latency_mean, 4),
        "LatencyStd": round(latency_std, 4),
        "Params": n_params,
        "Source": "Measured",
    })

    del model

    if device.type == "cuda":
        torch.cuda.empty_cache()


df = pd.DataFrame(results)


# ============================================================
# 7. Pareto Dominance
# ============================================================

def is_dominated(i, dataframe):

    acc_i = dataframe.loc[i, "Accuracy"]
    latency_i = dataframe.loc[i, "Latency"]

    for j in dataframe.index:

        if i == j:
            continue

        acc_j = dataframe.loc[j, "Accuracy"]
        latency_j = dataframe.loc[j, "Latency"]

        no_worse = (
            acc_j >= acc_i
            and
            latency_j <= latency_i
        )

        strictly_better = (
            acc_j > acc_i
            or
            latency_j < latency_i
        )

        if no_worse and strictly_better:
            return True

    return False


df["Pareto"] = True

for i in df.index:
    if is_dominated(i, df):
        df.loc[i, "Pareto"] = False


# ============================================================
# 8. Print
# ============================================================

print()
print("=" * 60)
print("Accuracy / Latency Results (all measured)")
print("=" * 60)

print(df.to_string(index=False))


# ============================================================
# 9. Plot
# ============================================================

pareto_df = df[df["Pareto"]].sort_values("Latency")

plt.figure(figsize=(8, 6))

plt.scatter(
    df["Latency"],
    df["Accuracy"],
    s=80,
    zorder=3,
)

for _, row in df.iterrows():

    label = (
        f"{row['Model']}\n"
        f"{row['Params'] / 1e6:.2f}M params"
    )

    plt.annotate(
        label,
        (row["Latency"], row["Accuracy"]),
        xytext=(6, -14),
        textcoords="offset points",
        fontsize=9,
    )

plt.plot(
    pareto_df["Latency"],
    pareto_df["Accuracy"],
    marker="o",
    label="Pareto Front",
    zorder=2,
)

plt.xlabel("Single-image Latency (ms)")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs. Latency (all models measured)")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig("pareto_full.png", dpi=200)

df.to_csv("pareto_full_results.csv", index=False)

print()
print("Saved: pareto_full_results.csv")
print("Saved: pareto_full.png")
