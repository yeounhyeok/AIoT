# ============================================================
# train_tinymlp.py
# Week 4 - Part A : TinyMLP MNIST Training
#
# Colab 셀 코드와 동일한 내용.
# google.colab 의존성(files.download)만 제거하고
# 가중치를 weights/TinyMLP.pth 로 바로 저장한다.
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ============================================================
# 1. Device and Hyperparameters
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 5

print("Device:", device)

if device.type == "cuda":
    print("GPU   :", torch.cuda.get_device_name(0))


# ============================================================
# 2. MNIST Dataset
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5,),
        (0.5,),
    ),
])

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# ============================================================
# 3. Model Definition
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


model = TinyMLP().to(device)

print()
print(model)


# ============================================================
# 4. Loss and Optimizer
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


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

    avg_loss = running_loss / len(train_loader)

    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
        f" | Loss = {avg_loss:.4f}"
    )


# ============================================================
# 6. Accuracy Check
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

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

accuracy = correct / total * 100

print()
print(f"Test Accuracy: {accuracy:.2f}%")


# ============================================================
# 7. Save Weights
# ============================================================

import os

os.makedirs("weights", exist_ok=True)

torch.save(
    model.state_dict(),
    "weights/TinyMLP.pth",
)

print()
print("Saved: weights/TinyMLP.pth")
