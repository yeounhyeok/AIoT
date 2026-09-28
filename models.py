# ============================================================
# models.py
# Week 4 (확장) - MNIST용 모델 정의
#
# torchvision의 ResNet / DenseNet / EfficientNet은 원래
# ImageNet(3채널 224x224, 1000 class)용이다.
# MNIST(1채널 28x28, 10 class)에 쓰려면 두 곳만 고치면 된다.
#
#   1) 첫 conv : in_channels 3 -> 1
#   2) 분류기   : out_features 1000 -> 10
#
# 입력 해상도는 28x28 그대로 둔다.
# (마지막에 AdaptiveAvgPool이 있어서 해상도가 작아도 동작한다)
# ============================================================

import math

import torch
import torch.nn as nn

from torchvision import models as tvm


NUM_CLASSES = 10


# ============================================================
# 1. TinyMLP (Part A에서 쓰던 것과 완전히 동일)
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
# 2. ResNet-18
# ============================================================

def build_resnet18():
    model = tvm.resnet18(
        weights=None,
        num_classes=NUM_CLASSES,
    )

    # 28x28 입력에서 7x7 stride 2 conv + maxpool은 너무 공격적이라
    # 3x3 stride 1 conv로 바꾸고 maxpool을 제거하는 것이 CIFAR/MNIST 관례다.
    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )

    model.maxpool = nn.Identity()

    return model


# ============================================================
# 3. DenseNet-121
# ============================================================

def build_densenet121():
    model = tvm.densenet121(
        weights=None,
        num_classes=NUM_CLASSES,
    )

    model.features.conv0 = nn.Conv2d(
        in_channels=1,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )

    model.features.pool0 = nn.Identity()

    return model


# ============================================================
# 4. EfficientNet-B0
# ============================================================

def build_efficientnet_b0():
    model = tvm.efficientnet_b0(
        weights=None,
        num_classes=NUM_CLASSES,
    )

    first_conv = model.features[0][0]

    model.features[0][0] = nn.Conv2d(
        in_channels=1,
        out_channels=first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=False,
    )

    return model


# ============================================================
# 5. Registry
# ============================================================

MODEL_REGISTRY = {
    "TinyMLP": TinyMLP,
    "ResNet18": build_resnet18,
    "DenseNet121": build_densenet121,
    "EfficientNetB0": build_efficientnet_b0,
}


def build_model(name):
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model: {name}. "
            f"Available: {list(MODEL_REGISTRY)}"
        )

    return MODEL_REGISTRY[name]()


def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
    )
