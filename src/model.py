"""model.py — Stage 2: transfer-learning model + embedding extractor.

Configure a pretrained ResNet18 backbone for transfer learning (freeze the backbone,
replace the final layer with a fresh 2-class head). The same backbone is reused as a
512-dim feature extractor for embedding drift. See notebook "Model Development".
"""
from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def build_model(freeze: bool | None = None) -> nn.Module:
    # TODO 2: load torchvision resnet18 with ImageNet weights; if freeze, set
    #         requires_grad=False on backbone params; replace net.fc with a
    #         nn.Linear(in_features, config.NUM_CLASSES) trainable head.
    from torchvision import models
    freeze = config.FREEZE_BACKBONE if freeze is None else freeze
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    net = models.resnet18(weights=weights)
    if freeze:
        for p in net.parameters():
            p.requires_grad = False
    net.fc = nn.Linear(net.fc.in_features, config.NUM_CLASSES)   # trainable head
    return net
    # raise NotImplementedError("Build the ResNet18 transfer-learning model")


def trainable_parameters(net: nn.Module):
    return [p for p in net.parameters() if p.requires_grad]


class EmbeddingExtractor(nn.Module):
    """Expose the 512-dim penultimate features (drop the fc layer)."""
    def __init__(self, net: nn.Module):
        super().__init__()
        # TODO 4 (embedding drift): keep all layers except the final fc.
        self.features = nn.Sequential(*list(net.children())[:-1])
        # raise NotImplementedError("Wrap the backbone to output pre-fc embeddings")
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        return torch.flatten(z, 1)

def save_model(net: nn.Module, path: Path | None = None) -> None:
    path = path or config.MODEL_PATH
    torch.save(net.state_dict(), path)


def load_model(path: Path | None = None, freeze: bool = True) -> nn.Module:
    path = path or config.MODEL_PATH
    net = build_model(freeze=freeze)
    net.load_state_dict(torch.load(path, map_location=config.DEVICE))
    net.eval()
    return net
