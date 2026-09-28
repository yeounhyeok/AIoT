# ============================================================
# train_models.py
# Week 4 (확장) - ResNet18 / DenseNet121 / EfficientNetB0 학습
#
# TinyMLP와 완전히 동일한 조건으로 학습한다.
#   전처리 : ToTensor + Normalize((0.5,), (0.5,))
#   최적화 : Adam, lr 1e-3
#   배치   : 128
#   에폭   : 5
#
# 조건을 맞춰야 Accuracy 차이를 "모델 구조 차이"로 해석할 수 있다.
# ============================================================

import argparse
import json
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import build_model, count_parameters


# ============================================================
# 1. Settings
# ============================================================

BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCHS = 5

WEIGHT_DIR = "weights"
LOG_PATH = os.environ.get("LOG_PATH", "train_log.json")


# ============================================================
# 2. Data
# ============================================================

def build_loaders():
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
        num_workers=4,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, test_loader


# ============================================================
# 3. Train / Eval
# ============================================================

def evaluate(model, loader, device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return correct / total * 100


def train_one(name, train_loader, test_loader, device):
    model = build_model(name).to(device)

    n_params = count_parameters(model)

    print()
    print("=" * 60)
    print(f"{name}  (params: {n_params:,})")
    print("=" * 60)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    t0 = time.perf_counter()

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

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
            f" | {time.perf_counter() - t0:7.1f}s"
        )

    train_time = time.perf_counter() - t0

    accuracy = evaluate(model, test_loader, device)

    os.makedirs(WEIGHT_DIR, exist_ok=True)

    weight_path = os.path.join(
        WEIGHT_DIR,
        f"{name}.pth",
    )

    torch.save(model.state_dict(), weight_path)

    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Train time   : {train_time:.1f}s")
    print(f"Saved        : {weight_path}")

    return {
        "model": name,
        "params": n_params,
        "accuracy": accuracy,
        "train_time_sec": round(train_time, 1),
        "epochs": EPOCHS,
    }


# ============================================================
# 4. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "ResNet18",
            "DenseNet121",
            "EfficientNetB0",
        ],
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if device.type == "cuda":
        print("GPU   :", torch.cuda.get_device_name(0))

    train_loader, test_loader = build_loaders()

    logs = []

    if os.path.exists(LOG_PATH):
        with open(LOG_PATH) as f:
            logs = json.load(f)

    for name in args.models:
        record = train_one(
            name,
            train_loader,
            test_loader,
            device,
        )

        logs = [
            r for r in logs
            if r["model"] != name
        ]

        logs.append(record)

        with open(LOG_PATH, "w") as f:
            json.dump(logs, f, indent=2)

    print()
    print("Saved:", LOG_PATH)


if __name__ == "__main__":
    main()
