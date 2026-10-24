"""Simplified public replacements for the named D2Q-MoE components.

These small, trainable examples are not the research algorithms. They use
ordinary convolutions, fixed branch averaging, and an independent class
projection. The complete research implementations are not distributed here.
"""

import torch
from torch import nn


class GTDMoE(nn.Module):
    """Public replacement: two ordinary convolution branches, equally averaged."""

    def __init__(self, channels):
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv2d(channels, channels, 1), nn.SiLU()),
            nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.SiLU()),
        ])

    def forward(self, depth_features):
        return torch.stack([branch(depth_features) for branch in self.branches]).mean(0)


class RADF(nn.Module):
    """Public replacement: concatenate RGB/depth features and project channels."""

    def __init__(self, rgb_channels, depth_channels):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(rgb_channels + depth_channels, rgb_channels, 1, bias=False),
            nn.BatchNorm2d(rgb_channels),
            nn.SiLU(),
        )

    def forward(self, rgb_features, depth_features):
        if rgb_features.shape[0] != depth_features.shape[0] or rgb_features.shape[-2:] != depth_features.shape[-2:]:
            raise ValueError("RGB and depth features must have matching batch and spatial dimensions")
        return self.projection(torch.cat([rgb_features, depth_features], dim=1))


class SRQM(nn.Module):
    """Public replacement: the same independent class projection for each query."""

    def __init__(self, num_classes):
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes must be positive")
        self.class_projection = nn.Linear(num_classes, num_classes)
        nn.init.eye_(self.class_projection.weight)
        nn.init.zeros_(self.class_projection.bias)

    def forward(self, logits):
        return self.class_projection(logits)
