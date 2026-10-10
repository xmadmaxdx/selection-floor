"""Small-image backbones with explicit contracts."""
from __future__ import annotations

import torch.nn as nn
import torchvision.models as tvm


class SimpleCNN(nn.Module):
    """Three-block convolutional network. The lightweight arm of the study."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(x).flatten(1))


class SimpleCNN1Ch(nn.Module):
    """SimpleCNN with a single input channel, for the 28x28 greyscale sets."""

    def __init__(self, num_classes: int = 10, in_channels: int = 1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(x).flatten(1))


def _resnet18_small(num_classes: int, in_channels: int = 3) -> nn.Module:
    model = tvm.resnet18(weights=None)
    model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_model(name: str, num_classes: int, in_channels: int = 3) -> nn.Module:
    """Build a backbone. Raises ValueError on an unknown name."""
    if num_classes < 2:
        raise ValueError("num_classes must be >= 2")
    if in_channels not in (1, 3):
        raise ValueError("in_channels must be 1 or 3")
    if name == "resnet18_small":
        return _resnet18_small(num_classes, in_channels)
    if name == "simplecnn":
        return SimpleCNN1Ch(num_classes=num_classes, in_channels=in_channels) if in_channels == 1 \
            else SimpleCNN(num_classes=num_classes)
    raise ValueError(f"unknown model: {name}")


def count_parameters(model: nn.Module) -> int:
    """Total trainable parameter count."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))