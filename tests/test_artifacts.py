"""Tests for local artifact injection and PSNR calibration."""

import numpy as np
import pytest

from artifacts import calibrate_amount, inject, patch_psnr

TOP, LEFT, PATCH, FEATHER = 192, 192, 128, 8


@pytest.fixture
def phantom() -> np.ndarray:
    """Smooth synthetic slice with fine texture, values in [0, 1]."""
    rows, cols = np.mgrid[0:512, 0:512] / 512.0
    texture = 0.05 * np.sin(40 * np.pi * rows) * np.sin(40 * np.pi * cols)
    return np.clip(0.5 + 0.2 * np.sin(np.pi * rows) + texture, 0, 1).astype(np.float32)


@pytest.mark.parametrize("artifact", ["noise", "blur", "streak", "shading"])
def test_injection_stays_inside_target_patch(phantom: np.ndarray, artifact: str) -> None:
    """Injection stays inside target patch."""
    corrupted = inject(phantom, artifact, 0.3, TOP, LEFT, PATCH, FEATHER, rng_seed=1)
    changed = np.argwhere(np.abs(corrupted - phantom) > 0)
    assert changed.size > 0
    assert changed[:, 0].min() >= TOP and changed[:, 0].max() < TOP + PATCH
    assert changed[:, 1].min() >= LEFT and changed[:, 1].max() < LEFT + PATCH


def test_injection_is_deterministic_for_a_seed(phantom: np.ndarray) -> None:
    """Injection is deterministic for a seed."""
    first = inject(phantom, "streak", 0.2, TOP, LEFT, PATCH, FEATHER, rng_seed=7)
    second = inject(phantom, "streak", 0.2, TOP, LEFT, PATCH, FEATHER, rng_seed=7)
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("artifact", ["noise", "streak", "shading"])
@pytest.mark.parametrize("target", [35.0, 30.0, 25.0])
def test_calibration_reaches_target_psnr(phantom: np.ndarray, artifact: str, target: float) -> None:
    """Calibration reaches the target PSNR."""
    amount, achieved = calibrate_amount(phantom, artifact, target, TOP, LEFT, PATCH, FEATHER, 3)
    assert amount > 0
    assert abs(achieved - target) < 0.5
    corrupted = inject(phantom, artifact, amount, TOP, LEFT, PATCH, FEATHER, 3)
    assert patch_psnr(phantom, corrupted, TOP, LEFT, PATCH) == pytest.approx(achieved)


def test_unknown_artifact_raises(phantom: np.ndarray) -> None:
    """Unknown artifact raises."""
    with pytest.raises(ValueError):
        inject(phantom, "ring", 0.1, TOP, LEFT, PATCH, FEATHER, rng_seed=0)
