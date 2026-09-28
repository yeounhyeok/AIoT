import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
import gzip
import struct
from pathlib import Path

from model import SmallCNN


# ============================================================
# 1. Reproducibility and Device
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type != "cuda":
    raise RuntimeError("CUDA is not available. Check the previous lab setup.")

print("Device:", device)
print("GPU   :", torch.cuda.get_device_name(0))


# ============================================================
# 2. Required Files
# ============================================================

required_files = [
    "model.py",
    "mnist_fp32.pth",
    "test_data.npy",
    "test_labels.npy",
]

for path in required_files:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing required file: {path}. "
            "Complete the unstructured pruning lab first."
        )


# ============================================================
# 3. Training and Test Data
# ============================================================

def load_images(path):
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(n, 1, rows, cols)

    return data.astype(np.float32) / 255.0


def load_labels(path):
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return labels.astype(np.int64)


data_dir = Path("./data")

train_images_path = data_dir / "train-images-idx3-ubyte.gz"
train_labels_path = data_dir / "train-labels-idx1-ubyte.gz"

if not train_images_path.exists():
    raise FileNotFoundError(
        f"Missing MNIST file: {train_images_path}. "
        "Complete the previous lab data setup first."
    )

if not train_labels_path.exists():
    raise FileNotFoundError(
        f"Missing MNIST file: {train_labels_path}. "
        "Complete the previous lab data setup first."
    )

train_images = load_images(train_images_path)
train_labels = load_labels(train_labels_path)

train_dataset = TensorDataset(
    torch.tensor(train_images, dtype=torch.float32),
    torch.tensor(train_labels, dtype=torch.long),
)

train_loader = DataLoader(
    train_dataset,
    batch_size=256,
    shuffle=True,
    num_workers=0,
)

x_test = torch.tensor(
    np.load("test_data.npy"),
    dtype=torch.float32,
)

y_test = torch.tensor(
    np.load("test_labels.npy"),
    dtype=torch.long,
)

print("Train samples:", len(train_dataset))
print("Test X shape :", tuple(x_test.shape))
print("Test Y shape :", tuple(y_test.shape))


# ============================================================
# 4. Structured Model
# ============================================================

class PrunedCNN(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c2 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# ============================================================
# 5. Accuracy
# ============================================================

def evaluate(model):
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


# ============================================================
# 6. Latency
# ============================================================

def measure_latency(model, warmup=100, repeat=500):
    """Measure synchronized end-to-end single-image latency."""
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


# ============================================================
# 7. Parameter Count
# ============================================================

def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


# ============================================================
# 8. FLOPs
# ============================================================

def calculate_flops(model):
    """Count multiply and add as two operations."""

    flops = 0
    hooks = []

    def conv_hook(module, inputs, output):
        nonlocal flops

        batch_size = output.shape[0]
        output_channels = output.shape[1]
        output_height = output.shape[2]
        output_width = output.shape[3]
        kernel_height, kernel_width = module.kernel_size
        input_channels = module.in_channels // module.groups

        flops += (
            batch_size
            * output_channels
            * output_height
            * output_width
            * input_channels
            * kernel_height
            * kernel_width
            * 2
        )

    def linear_hook(module, inputs, output):
        nonlocal flops
        batch_size = output.shape[0] if output.dim() > 1 else 1
        flops += batch_size * module.in_features * module.out_features * 2

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))

    dummy = torch.randn(1, 1, 28, 28, device=device)

    model.eval()
    with torch.no_grad():
        _ = model(dummy)

    for hook in hooks:
        hook.remove()

    return int(flops)


# ============================================================
# 9. Create a Structured-Pruned Model
# ============================================================

def create_pruned_model(original, c1, c2):
    new_model = PrunedCNN(c1, c2).to(device)

    # Conv1: select output filters by L1 norm.
    conv1 = original.features[0]
    importance1 = conv1.weight.detach().abs().sum(dim=(1, 2, 3))
    keep1 = torch.argsort(importance1, descending=True)[:c1]
    keep1 = torch.sort(keep1).values

    with torch.no_grad():
        new_model.features[0].weight.copy_(conv1.weight[keep1])
        new_model.features[0].bias.copy_(conv1.bias[keep1])

    # Conv2: first retain the Conv1 input channels, then rank outputs.
    conv2 = original.features[3]
    reduced_input = conv2.weight.detach()[:, keep1, :, :]
    importance2 = reduced_input.abs().sum(dim=(1, 2, 3))
    keep2 = torch.argsort(importance2, descending=True)[:c2]
    keep2 = torch.sort(keep2).values

    with torch.no_grad():
        new_model.features[3].weight.copy_(
            conv2.weight[keep2][:, keep1, :, :]
        )
        new_model.features[3].bias.copy_(conv2.bias[keep2])

    # FC1: retain all 7x7 features belonging to each kept Conv2 channel.
    feature_indices = []

    for channel in keep2.tolist():
        start = channel * 7 * 7
        end = start + 7 * 7
        feature_indices.extend(range(start, end))

    feature_indices = torch.tensor(
        feature_indices,
        dtype=torch.long,
        device=device,
    )

    fc1 = original.classifier[1]

    with torch.no_grad():
        new_model.classifier[1].weight.copy_(
            fc1.weight[:, feature_indices]
        )
        new_model.classifier[1].bias.copy_(fc1.bias)

        # FC2 shape does not change.
        new_model.classifier[3].weight.copy_(
            original.classifier[3].weight
        )
        new_model.classifier[3].bias.copy_(
            original.classifier[3].bias
        )

    return new_model, keep1.tolist(), keep2.tolist()


# ============================================================
# 10. Fine-Tuning
# ============================================================

def finetune(model, epochs=3):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    history = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        average_loss = running_loss / len(train_dataset)
        accuracy = evaluate(model)
        history.append((average_loss, accuracy))

        print(
            f"  Epoch {epoch + 1}/{epochs}"
            f" | Loss: {average_loss:.4f}"
            f" | Accuracy: {accuracy:.2f}%"
        )

    return history


# ============================================================
# 11. Pareto Test
# ============================================================

def is_dominated(index, dataframe):
    accuracy_i = dataframe.loc[index, "Accuracy After FT (%)"]
    flops_i = dataframe.loc[index, "FLOPs"]

    for other in dataframe.index:
        if other == index:
            continue

        accuracy_j = dataframe.loc[other, "Accuracy After FT (%)"]
        flops_j = dataframe.loc[other, "FLOPs"]

        no_worse = accuracy_j >= accuracy_i and flops_j <= flops_i
        strictly_better = accuracy_j > accuracy_i or flops_j < flops_i

        if no_worse and strictly_better:
            return True

    return False


# ============================================================
# 12. Load Baseline
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


# ============================================================
# 13. Baseline Measurement
# ============================================================

baseline_accuracy = evaluate(baseline)
baseline_latency = measure_latency(baseline)
baseline_parameters = count_parameters(baseline)
baseline_flops = calculate_flops(baseline)

results = [{
    "Model": "Baseline",
    "C1": 16,
    "C2": 32,
    "Parameters": baseline_parameters,
    "FLOPs": baseline_flops,
    "Accuracy Before FT (%)": baseline_accuracy,
    "Accuracy After FT (%)": baseline_accuracy,
    "Mean Latency (ms)": baseline_latency["mean"],
    "Median Latency (ms)": baseline_latency["median"],
    "Std Latency (ms)": baseline_latency["std"],
}]


# ============================================================
# 14. Structured Pruning Sweep
# ============================================================

settings = [
    ("Structured 25%", 12, 24, 25),
    ("Structured 50%", 8, 16, 50),
    ("Structured 75%", 4, 8, 75),
]

for name, c1, c2, ratio in settings:
    print()
    print("=" * 60)
    print(name)
    print(f"Channels: Conv1 16 -> {c1}, Conv2 32 -> {c2}")
    print("=" * 60)

    model, keep1, keep2 = create_pruned_model(baseline, c1, c2)

    print("Kept Conv1 channels:", keep1)
    print("Kept Conv2 channels:", keep2)

    accuracy_before = evaluate(model)
    print(f"Accuracy before fine-tuning: {accuracy_before:.2f}%")

    finetune(model, epochs=3)

    accuracy_after = evaluate(model)
    latency = measure_latency(model)
    parameters = count_parameters(model)
    flops = calculate_flops(model)

    output_path = f"mnist_structured_{ratio}_ft.pth"
    torch.save(model.state_dict(), output_path)
    print("Saved:", output_path)

    results.append({
        "Model": name,
        "C1": c1,
        "C2": c2,
        "Parameters": parameters,
        "FLOPs": flops,
        "Accuracy Before FT (%)": accuracy_before,
        "Accuracy After FT (%)": accuracy_after,
        "Mean Latency (ms)": latency["mean"],
        "Median Latency (ms)": latency["median"],
        "Std Latency (ms)": latency["std"],
    })


# ============================================================
# 15. Results and Pareto Front
# ============================================================

df = pd.DataFrame(results)
df["Parameter Reduction (%)"] = (
    100.0 * (1.0 - df["Parameters"] / baseline_parameters)
)
df["FLOPs Reduction (%)"] = (
    100.0 * (1.0 - df["FLOPs"] / baseline_flops)
)

df["Pareto"] = True

for index in df.index:
    if is_dominated(index, df):
        df.loc[index, "Pareto"] = False

df.to_csv("structured_sweep_results.csv", index=False)

print()
print("=" * 120)
print("STRUCTURED PRUNING RESULTS")
print("=" * 120)
print(df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
print()
print("Saved: structured_sweep_results.csv")


# ============================================================
# 16. Visualization
# ============================================================

pareto_df = df[df["Pareto"]].sort_values("FLOPs")

plt.figure(figsize=(8, 6))

plt.scatter(
    df["FLOPs"] / 1_000_000,
    df["Accuracy After FT (%)"],
    s=80,
    label="Models",
)

for _, row in df.iterrows():
    plt.annotate(
        row["Model"],
        (row["FLOPs"] / 1_000_000, row["Accuracy After FT (%)"]),
        xytext=(5, 5),
        textcoords="offset points",
    )

plt.plot(
    pareto_df["FLOPs"] / 1_000_000,
    pareto_df["Accuracy After FT (%)"],
    marker="o",
    label="Pareto Front",
)

plt.xlabel("FLOPs (million, multiply + add = 2 operations)")
plt.ylabel("Accuracy After Fine-Tuning (%)")
plt.title("Structured Pruning: Accuracy vs. FLOPs")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("structured_accuracy_flops.png", dpi=200)

print("Saved: structured_accuracy_flops.png")
