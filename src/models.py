"""The audited networks: a two-stream EfficientNet-B0 and ResNet18 regressors."""

import warnings
from pathlib import Path

import torch
from torch import nn
from torchvision import models


def _efficientnet_b0(pretrained: bool) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    try:
        return models.efficientnet_b0(weights=weights)
    except (OSError, RuntimeError) as error:  # weights could not be downloaded
        warnings.warn(f"ImageNet weights unavailable, using random init: {error}", stacklevel=2)
        return models.efficientnet_b0(weights=None)


def _resnet18(pretrained: bool) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    try:
        return models.resnet18(weights=weights)
    except (OSError, RuntimeError) as error:  # weights could not be downloaded
        warnings.warn(f"ImageNet weights unavailable, using random init: {error}", stacklevel=2)
        return models.resnet18(weights=None)


class TwoStreamNet(nn.Module):
    """Anatomy stream and HH high-frequency stream, fused by an MLP into one patch score."""

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        self.stream_anatomy = _efficientnet_b0(pretrained)
        self.stream_anatomy.features[0][0] = nn.Conv2d(
            1, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.stream_anatomy.classifier = nn.Identity()
        self.stream_noise = _efficientnet_b0(pretrained)
        self.stream_noise.features[0][0] = nn.Conv2d(
            1, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.stream_noise.classifier = nn.Identity()
        self.fusion_fc = nn.Sequential(
            nn.Linear(2560, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, 1)
        )

    def forward(self, x_anatomy: torch.Tensor, x_noise: torch.Tensor) -> torch.Tensor:
        """Return scores of shape [N, 1] for anatomy and HH inputs of shape [N, 1, H, W]."""
        features = torch.cat((self.stream_anatomy(x_anatomy), self.stream_noise(x_noise)), dim=1)
        return self.fusion_fc(features)


class ResNetRegressor(nn.Module):
    """Single-channel ResNet18 with one regression output (patch or global model)."""

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        self.net = _resnet18(pretrained)
        self.net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.net.fc = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return scores of shape [N, 1] for inputs of shape [N, 1, H, W]."""
        return self.net(x)


def is_two_stream(model_kind: str) -> bool:
    """True for the two-stream model kinds."""
    return model_kind.startswith("twostream")


def build_model(model_kind: str, pretrained: bool = False) -> nn.Module:
    """Create an untrained network for ``model_kind``."""
    return TwoStreamNet(pretrained) if is_two_stream(model_kind) else ResNetRegressor(pretrained)


def load_model(model_kind: str, checkpoint: Path, device: torch.device) -> nn.Module:
    """Load a ``state_dict`` checkpoint into a fresh network in evaluation mode.

    ResNet checkpoints saved without the ``net.`` key prefix are accepted too.
    """
    state = torch.load(checkpoint, map_location=device)
    if not is_two_stream(model_kind) and not any(key.startswith("net.") for key in state):
        state = {f"net.{key}": value for key, value in state.items()}
    model = build_model(model_kind, pretrained=False).to(device)
    model.load_state_dict(state)
    return model.eval()


def forward_model(model_kind: str, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run a model on a channel stack ``x`` of shape [N, C, H, W]; returns shape [N]."""
    if is_two_stream(model_kind):
        return model(x[:, 0:1], x[:, 1:2]).squeeze(1)
    return model(x).squeeze(1)
