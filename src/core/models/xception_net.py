"""
Xception architecture matching RamadhanZome / Chollet-style checkpoints.

Checkpoint layout uses:
  block{1-3} entry residual blocks, middle_flow[0-7], block4 exit residual,
  sepconv1/2 + bn3/bn4, fc classifier.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class EntryExitBlock(nn.Module):
    """Entry/exit residual block with 2 separable convolutions + max-pool."""

    def __init__(self, in_channels: int, out_channels: int, start_with_relu: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        if start_with_relu:
            layers.append(nn.ReLU(inplace=False))
        else:
            layers.append(nn.Identity())
        layers.extend(
            [
                SeparableConv2d(in_channels, out_channels),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=False),
                SeparableConv2d(out_channels, out_channels),
                nn.BatchNorm2d(out_channels),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ]
        )
        self.layers = nn.Sequential(*layers)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x) + self.shortcut(x)


class MiddleBlock(nn.Module):
    """Middle-flow residual block with 3 separable convolutions."""

    def __init__(self, channels: int = 728) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.ReLU(inplace=False),
            SeparableConv2d(channels, channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False),
            SeparableConv2d(channels, channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False),
            SeparableConv2d(channels, channels),
            nn.BatchNorm2d(channels),
        )
        # Identity shortcut stored as 1x1 + BN to match checkpoint keys
        self.shortcut = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x) + self.shortcut(x)


class XceptionNet(nn.Module):
    """Full Xception for binary real/fake classification (299x299 input)."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        self.block1 = EntryExitBlock(64, 128, start_with_relu=False)
        self.block2 = EntryExitBlock(128, 256, start_with_relu=True)
        self.block3 = EntryExitBlock(256, 728, start_with_relu=True)

        self.middle_flow = nn.Sequential(*[MiddleBlock(728) for _ in range(8)])

        self.block4 = EntryExitBlock(728, 1024, start_with_relu=True)

        self.sepconv1 = SeparableConv2d(1024, 1536)
        self.bn3 = nn.BatchNorm2d(1536)
        self.sepconv2 = SeparableConv2d(1536, 2048)
        self.bn4 = nn.BatchNorm2d(2048)

        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.middle_flow(x)
        x = self.block4(x)
        x = self.relu(self.bn3(self.sepconv1(x)))
        x = self.relu(self.bn4(self.sepconv2(x)))
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return self.fc(x)
