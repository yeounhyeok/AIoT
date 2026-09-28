# ============================================================
# pareto_check.py
#
# Week 4 - Accuracy / Latency Pareto Analysis
#
# Actual measurement:
#   TinyMLP
#
# Demo values:
#   SmallCNN / MediumCNN / LargeCNN / EfficientCNN
# ============================================================

import time

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless 환경(디스플레이 없음)에서 저장만 수행
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ============================================================
# 1. Settings
# ============================================================

BATCH_SIZE = 128

LATENCY_WARMUP = 50
LATENCY_REPEAT = 200


# ============================================================
# 2. Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("Device :", device)

if device.type == "cuda":
    print(
        "GPU    :",
        torch.cuda.get_device_name(0),
    )

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
# 4. TinyMLP
# ============================================================

class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 5. Accuracy
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

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return (
        correct
        / total
        * 100
    )


# ============================================================
# 6. Latency
# ============================================================

def measure_latency(model):
    model.eval()

    # Single-image input
    x = torch.randn(
        1,
        1,
        28,
        28,
        device=device,
    )

    # Warm-up
    with torch.no_grad():
        for _ in range(
            LATENCY_WARMUP
        ):
            _ = model(x)

    if device.type == "cuda":
        torch.cuda.synchronize()

    latencies = []

    with torch.no_grad():
        for _ in range(
            LATENCY_REPEAT
        ):

            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()

            _ = model(x)

            if device.type == "cuda":
                torch.cuda.synchronize()

            end = time.perf_counter()

            latencies.append(
                (end - start)
                * 1000
            )

    return np.mean(latencies)


# ============================================================
# 7. Load Actual Model
# ============================================================

model = TinyMLP()

state_dict = torch.load(
    "weights/TinyMLP.pth",
    map_location="cpu",
    weights_only=True,
)

model.load_state_dict(
    state_dict
)

model = model.to(device)
model.eval()


# ============================================================
# 8. Actual Measurement
# ============================================================

print()
print("=" * 60)
print("TinyMLP - Actual Measurement")
print("=" * 60)

tiny_accuracy = measure_accuracy(
    model
)

tiny_latency = measure_latency(
    model
)

print(
    f"Accuracy : "
    f"{tiny_accuracy:.2f} %"
)

print(
    f"Latency  : "
    f"{tiny_latency:.4f} ms"
)


results = [
    {
        "Model": "TinyMLP",
        "Accuracy": tiny_accuracy,
        "Latency": tiny_latency,
        "Source": "Measured",
    }
]


# ============================================================
# 9. Demo Results
#
# These values are NOT real measurements.
# They are used only to demonstrate Pareto analysis.
# ============================================================

fake_results = [
    {
        "Model": "EfficientCNN",
        "Accuracy": 96.8,
        "Latency": 0.42,
        "Source": "Demo",
    },
    {
        "Model": "SmallCNN",
        "Accuracy": 97.8,
        "Latency": 0.62,
        "Source": "Demo",
    },
    {
        "Model": "MediumCNN",
        "Accuracy": 97.2,
        "Latency": 0.95,
        "Source": "Demo",
    },
    {
        "Model": "LargeCNN",
        "Accuracy": 99.0,
        "Latency": 1.65,
        "Source": "Demo",
    },
]

results.extend(
    fake_results
)


# ============================================================
# 10. DataFrame
# ============================================================

df = pd.DataFrame(
    results
)


# ============================================================
# 11. Pareto Dominance
# ============================================================

def is_dominated(
    i,
    dataframe,
):

    acc_i = dataframe.loc[
        i,
        "Accuracy"
    ]

    latency_i = dataframe.loc[
        i,
        "Latency"
    ]

    for j in dataframe.index:

        if i == j:
            continue

        acc_j = dataframe.loc[
            j,
            "Accuracy"
        ]

        latency_j = dataframe.loc[
            j,
            "Latency"
        ]

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

        if (
            no_worse
            and
            strictly_better
        ):
            return True

    return False


df["Pareto"] = True

for i in df.index:

    if is_dominated(
        i,
        df,
    ):
        df.loc[
            i,
            "Pareto"
        ] = False


# ============================================================
# 12. Print Results
# ============================================================

print()
print("=" * 60)
print("Accuracy / Latency Results")
print("=" * 60)

print(
    df.to_string(
        index=False
    )
)


# ============================================================
# 13. Pareto Front
# ============================================================

pareto_df = df[
    df["Pareto"]
].copy()

pareto_df = pareto_df.sort_values(
    "Latency"
)


# ============================================================
# 14. Plot
# ============================================================

plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    df["Latency"],
    df["Accuracy"],
    s=80,
)

for _, row in df.iterrows():

    label = (
        f"{row['Model']}"
        f" ({row['Source']})"
    )

    plt.annotate(
        label,
        (
            row["Latency"],
            row["Accuracy"],
        ),
        xytext=(5, 5),
        textcoords="offset points",
    )

plt.plot(
    pareto_df["Latency"],
    pareto_df["Accuracy"],
    marker="o",
    label="Pareto Front",
)

plt.xlabel(
    "Single-image Latency (ms)"
)

plt.ylabel(
    "Accuracy (%)"
)

plt.title(
    "Accuracy vs. Latency"
)

plt.grid(
    alpha=0.3
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "pareto_accuracy_latency.png",
    dpi=200,
)

df.to_csv(
    "pareto_results.csv",
    index=False,
)

print()
print(
    "Saved: pareto_results.csv"
)

print(
    "Saved: pareto_accuracy_latency.png"
)
