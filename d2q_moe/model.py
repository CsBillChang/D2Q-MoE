"""Runnable public RGB-D example built on the current DEIM component interface.

The wrapper follows the local DEIM backbone/encoder/decoder injection contract.
Its added modules are simplified replacements, not the complete paper model.
"""

import torch
from torch import nn
import torch.nn.functional as F

from .modules import GTDMoE, RADF, SRQM


class D2QMoE(nn.Module):
    """Public RGB-D detector using deliberately simplified trainable components."""

    __inject__ = ["backbone", "encoder", "decoder"]
    __share__ = ["num_classes"]
    expects_multimodal_input = True

    def __init__(self, backbone, encoder, decoder, num_classes=80,
                 feat_channels=(384, 768, 1536), depth_channels=16):
        super().__init__()
        if not feat_channels or depth_channels < 1:
            raise ValueError("feat_channels and depth_channels must be positive")
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        self.num_classes = int(num_classes)
        self.depth_projections = nn.ModuleList([
            nn.Conv2d(1, depth_channels, 1) for _ in feat_channels
        ])
        self.depth_adapters = nn.ModuleList([GTDMoE(depth_channels) for _ in feat_channels])
        self.fusion_layers = nn.ModuleList([RADF(int(channels), depth_channels) for channels in feat_channels])
        self.query_modulation = SRQM(self.num_classes)

    def _project_class_outputs(self, output):
        if isinstance(output, dict):
            return {
                key: self.query_modulation(value)
                if key in {"pred_logits", "teacher_logits"} and isinstance(value, torch.Tensor)
                and value.shape[-1] == self.num_classes
                else self._project_class_outputs(value)
                for key, value in output.items()
            }
        if isinstance(output, list):
            return [self._project_class_outputs(value) for value in output]
        if isinstance(output, tuple):
            return tuple(self._project_class_outputs(value) for value in output)
        return output

    def forward(self, samples, targets=None):
        if not isinstance(samples, dict) or "rgb" not in samples or "npy" not in samples:
            raise TypeError("The public RGB-D example expects a dictionary with 'rgb' and 'npy' tensors")
        rgb, depth = samples["rgb"], samples["npy"]
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("rgb must have shape [batch, 3, height, width]")
        if depth.ndim == 3:
            depth = depth.unsqueeze(1)
        if depth.ndim != 4 or depth.shape[1] != 1 or depth.shape[0] != rgb.shape[0]:
            raise ValueError("npy must have shape [batch, 1, height, width]")
        if depth.shape[-2:] != rgb.shape[-2:]:
            raise ValueError("RGB and depth input tensors must be spatially aligned")
        depth = depth.to(device=rgb.device, dtype=rgb.dtype)
        rgb_features = self.backbone(rgb)
        if len(rgb_features) != len(self.depth_projections):
            raise ValueError("feat_channels must match the number of backbone output levels")
        fused_features = []
        for rgb_level, project_depth, adapt_depth, fuse in zip(
            rgb_features, self.depth_projections, self.depth_adapters, self.fusion_layers
        ):
            resized_depth = F.interpolate(depth, size=rgb_level.shape[-2:], mode="bilinear", align_corners=False)
            depth_level = adapt_depth(project_depth(resized_depth))
            fused_features.append(fuse(rgb_level, depth_level))
        outputs = self.decoder(self.encoder(fused_features), targets)
        return self._project_class_outputs(outputs)

    def deploy(self):
        self.eval()
        for module in self.modules():
            if hasattr(module, "convert_to_deploy"):
                module.convert_to_deploy()
        return self


# Register after defining the class so both ``import engine`` and
# ``from d2q_moe import D2QMoE`` work without a circular-import dependency.
from engine.core import register

register()(D2QMoE)
