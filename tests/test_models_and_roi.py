"""Tests for the networks, model inputs and the smart ROI."""

import numpy as np
import torch

from models import build_model, forward_model
from preprocessing import MODEL_KINDS, hh_band, prepare_input
from roi import smart_roi_mask


def disk_phantom() -> np.ndarray:
    """Bright disk (the body) on a dark background, with a small bright speck in the air."""
    rows, cols = np.mgrid[0:512, 0:512]
    image = np.zeros((512, 512), dtype=np.float32)
    image[(rows - 256) ** 2 + (cols - 256) ** 2 < 180**2] = 0.6
    image[(rows - 300) ** 2 + (cols - 256) ** 2 < 20**2] = 0.1  # dark hole inside the body
    image[10:13, 10:13] = 0.9  # speck in the air
    return image


def test_roi_keeps_body_fills_holes_and_drops_specks() -> None:
    """ROI keeps the body, fills holes and drops specks."""
    roi = smart_roi_mask(disk_phantom())
    assert roi[256, 256]
    assert roi[300, 256]  # inner hole is filled
    assert not roi[11, 11]  # speck removed
    assert not roi[0, 511]


def test_hh_band_is_scaled_to_unit_range() -> None:
    """HH band is scaled to unit range."""
    hh = hh_band(np.random.default_rng(0).random((64, 64)).astype(np.float32))
    assert hh.shape == (64, 64)
    assert hh.min() >= 0.0 and hh.max() <= 1.0 + 1e-6


def test_input_channels_per_model_kind() -> None:
    """Input channels per model kind."""
    image = disk_phantom()
    channels = {kind: prepare_input(kind, image).shape[0] for kind in MODEL_KINDS}
    assert channels == {
        "twostream_roi": 2,
        "twostream_noroi": 2,
        "patch_resnet18": 1,
        "global_resnet18": 1,
    }
    assert prepare_input("twostream_roi", image)[0, 11, 11] == 0.0  # air is masked
    assert prepare_input("twostream_noroi", image)[0, 11, 11] > 0.0


def test_forward_returns_one_score_per_patch() -> None:
    """Forward returns one score per patch."""
    torch.manual_seed(0)
    patches = torch.rand(3, 2, 128, 128)
    for kind in ("twostream_roi", "patch_resnet18"):
        model = build_model(kind, pretrained=False).eval()
        x = patches if kind.startswith("twostream") else patches[:, :1]
        with torch.no_grad():
            assert forward_model(kind, model, x).shape == (3,)
