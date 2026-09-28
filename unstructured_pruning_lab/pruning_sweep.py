import copy
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.utils.prune as prune

from model import SmallCNN


# ============================================================
# 1. Device
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type != "cuda":
    raise RuntimeError("CUDA is not available.")

print("Device:", device)
print("GPU   :", torch.cuda.get_device_name(0))


# ============================================================
# 2. Load Test Data
# ============================================================

x_test = torch.tensor(
    np.load("test_data.npy"),
    dtype=torch.float32,
)

y_test = torch.tensor(
    np.load("test_labels.npy"),
    dtype=torch.long,
)

print("X shape:", tuple(x_test.shape))
print("Y shape:", tuple(y_test.shape))


# ============================================================
# 3. Helper Functions
# ============================================================

def pruning_targets(model):
    return (
        (model.features[0], "weight"),
        (model.features[3], "weight"),
        (model.classifier[1], "weight"),
        (model.classifier[3], "weight"),
    )


def evaluate_accuracy(model):
    model.eval()
    correct = 0
    total = 0
    batch_size = 256

    with torch.no_grad():
        for start in range(0, len(x_test), batch_size):
            images = x_test[start:start + batch_size].to(device)
            labels = y_test[start:start + batch_size].to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total


def measure_sparsity(model):
    zero_count = 0
    weight_count = 0

    for module, _ in pruning_targets(model):
        zero_count += (module.weight == 0).sum().item()
        weight_count += module.weight.numel()

    return 100.0 * zero_count / weight_count


def measure_latency(model, warmup=100, repeat=500):
    model.eval()
    sample = x_test[:1].to(device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(sample)

    torch.cuda.synchronize()
    latencies = []

    with torch.no_grad():
        for _ in range(repeat):
            torch.cuda.synchronize()
            start = time.perf_counter()

            _ = model(sample)

            torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000.0)

    return {
        "mean": float(np.mean(latencies)),
        "median": float(np.median(latencies)),
        "std": float(np.std(latencies)),
    }


def count_parameters(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    nonzero = sum(torch.count_nonzero(parameter).item() for parameter in model.parameters())
    return total, nonzero


# ============================================================
# 4. Load Baseline
# ============================================================

baseline = SmallCNN().to(device)
baseline.load_state_dict(
    torch.load(
        "mnist_fp32.pth",
        map_location=device,
        weights_only=True,
    )
)
baseline.eval()

ratios = [0.0, 0.3, 0.5, 0.7, 0.9]
results = []


# ============================================================
# 5. Pruning Sweep
# ============================================================

for ratio in ratios:
    model = copy.deepcopy(baseline)
    targets = pruning_targets(model)

    if ratio > 0:
        prune.global_unstructured(
            targets,
            pruning_method=prune.L1Unstructured,
            amount=ratio,
        )

        for module, name in targets:
            prune.remove(module, name)

    actual_sparsity = measure_sparsity(model)
    accuracy = evaluate_accuracy(model)
    latency = measure_latency(model)
    parameter_count, nonzero_parameter_count = count_parameters(model)

    output_path = f"mnist_pruned_{int(ratio * 100):02d}.pth"
    torch.save(model.state_dict(), output_path)
    file_size_kb = os.path.getsize(output_path) / 1024.0

    results.append({
        "Pruning (%)": int(ratio * 100),
        "Actual Sparsity (%)": actual_sparsity,
        "Accuracy (%)": accuracy,
        "Mean Latency (ms)": latency["mean"],
        "Median Latency (ms)": latency["median"],
        "Std Latency (ms)": latency["std"],
        "Parameters": parameter_count,
        "Nonzero Parameters": nonzero_parameter_count,
        "File Size (KB)": file_size_kb,
    })


# ============================================================
# 6. Save and Print Results
# ============================================================

df = pd.DataFrame(results)
df.to_csv("unstructured_pruning_results.csv", index=False)

print()
print("=" * 110)
print("Unstructured Pruning Results")
print("=" * 110)
print(df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
print()
print("Saved: unstructured_pruning_results.csv")


# ============================================================
# 7. Visualization
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(
    df["Actual Sparsity (%)"],
    df["Accuracy (%)"],
    marker="o",
)
axes[0].set_xlabel("Actual Weight Sparsity (%)")
axes[0].set_ylabel("Accuracy (%)")
axes[0].set_title("Accuracy vs. Sparsity")
axes[0].grid(alpha=0.3)

axes[1].errorbar(
    df["Actual Sparsity (%)"],
    df["Mean Latency (ms)"],
    yerr=df["Std Latency (ms)"],
    marker="o",
    capsize=4,
)
axes[1].set_xlabel("Actual Weight Sparsity (%)")
axes[1].set_ylabel("Single-image Latency (ms)")
axes[1].set_title("Latency vs. Sparsity")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("unstructured_pruning_results.png", dpi=200)

print("Saved: unstructured_pruning_results.png")
