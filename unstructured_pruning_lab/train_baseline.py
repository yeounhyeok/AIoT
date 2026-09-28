import gzip
import random
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset

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
    raise RuntimeError("CUDA is not available.")

print("Device:", device)
print("GPU   :", torch.cuda.get_device_name(0))


# ============================================================
# 2. Hyperparameters
# ============================================================

BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 1e-3


# ============================================================
# 3. MNIST Loader
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

train_images = load_images(data_dir / "train-images-idx3-ubyte.gz")
train_labels = load_labels(data_dir / "train-labels-idx1-ubyte.gz")

test_images = load_images(data_dir / "t10k-images-idx3-ubyte.gz")
test_labels = load_labels(data_dir / "t10k-labels-idx1-ubyte.gz")

print("Train:", train_images.shape)
print("Test :", test_images.shape)


train_dataset = TensorDataset(
    torch.tensor(train_images, dtype=torch.float32),
    torch.tensor(train_labels, dtype=torch.long),
)

test_dataset = TensorDataset(
    torch.tensor(test_images, dtype=torch.float32),
    torch.tensor(test_labels, dtype=torch.long),
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=0,
)


# ============================================================
# 4. Model, Loss, and Optimizer
# ============================================================

model = SmallCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# ============================================================
# 5. Training
# ============================================================

for epoch in range(EPOCHS):
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

        running_loss += loss.item()

    average_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {average_loss:.4f}")


# ============================================================
# 6. Baseline Accuracy
# ============================================================

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

accuracy = 100.0 * correct / total
print(f"Baseline Accuracy: {accuracy:.2f}%")


# ============================================================
# 7. Save Weights and Test Data
# ============================================================

torch.save(model.state_dict(), "mnist_fp32.pth")
np.save("test_data.npy", test_images)
np.save("test_labels.npy", test_labels)

print("Saved: mnist_fp32.pth")
print("Saved: test_data.npy", test_images.shape)
print("Saved: test_labels.npy", test_labels.shape)
