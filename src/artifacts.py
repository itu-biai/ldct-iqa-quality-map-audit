"""Local artifact injection at a matched patch PSNR.

Each artifact is applied to the whole image and then blended into the target patch
with a feathered mask: ``x' = (1 - M) * x + M * A(x)``, clipped to [0, 1].
"""

import numpy as np
from scipy.ndimage import gaussian_filter

# Search interval of the severity parameter ``a`` for each artifact.
AMOUNT_RANGE: dict[str, tuple[float, float]] = {
    "noise": (0.0, 0.5),  # standard deviation of Gaussian noise
    "blur": (0.1, 10.0),  # Gaussian kernel width in pixels (capped at 10 px)
    "streak": (0.0, 1.0),  # streak amplitude
    "shading": (0.0, 1.0),  # shading amplitude
}


def feather_mask(
    shape: tuple[int, int], top: int, left: int, patch_size: int, feather: int
) -> np.ndarray:
    """Mask equal to 1 inside the patch, with a raised-cosine taper of ``feather`` px."""
    ramp = np.ones(patch_size, dtype=np.float32)
    taper = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, feather, dtype=np.float32))
    ramp[:feather], ramp[-feather:] = taper, taper[::-1]
    mask = np.zeros(shape, dtype=np.float32)
    mask[top : top + patch_size, left : left + patch_size] = np.outer(ramp, ramp)
    return mask


def streak_pattern(
    shape: tuple[int, int],
    top: int,
    left: int,
    patch_size: int,
    rng: np.random.Generator,
    count: int = 5,
    width: float = 1.0,
) -> np.ndarray:
    """Sum of ``count`` Gaussian line profiles of random angle and sign through the patch."""
    rows, cols = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float32)
    pattern = np.zeros(shape, dtype=np.float32)
    for _ in range(count):
        angle = rng.uniform(0.0, np.pi)
        center_row = top + rng.uniform(0, patch_size)
        center_col = left + rng.uniform(0, patch_size)
        distance = (rows - center_row) * np.cos(angle) - (cols - center_col) * np.sin(angle)
        pattern += rng.choice([-1.0, 1.0]) * np.exp(-(distance**2) / (2 * width**2))
    return pattern


def shading_pattern(
    shape: tuple[int, int], top: int, left: int, patch_size: int, radius: float = 48.0
) -> np.ndarray:
    """Isotropic Gaussian bump centered on the patch (sigma = ``radius``)."""
    rows, cols = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float32)
    center_row, center_col = top + patch_size / 2, left + patch_size / 2
    return np.exp(-((rows - center_row) ** 2 + (cols - center_col) ** 2) / (2 * radius**2))


def inject(
    image: np.ndarray,
    artifact: str,
    amount: float,
    top: int,
    left: int,
    patch_size: int,
    feather: int,
    rng_seed: int,
) -> np.ndarray:
    """Inject one artifact of severity ``amount`` into the patch at (``top``, ``left``)."""
    rng = np.random.default_rng(rng_seed)
    mask = feather_mask(image.shape, top, left, patch_size, feather)
    if artifact == "noise":
        corrupted = image + amount * rng.standard_normal(image.shape).astype(np.float32)
    elif artifact == "blur":
        corrupted = gaussian_filter(image, sigma=amount)
    elif artifact == "streak":
        corrupted = image + amount * streak_pattern(image.shape, top, left, patch_size, rng)
    elif artifact == "shading":
        corrupted = image - amount * shading_pattern(image.shape, top, left, patch_size)
    else:
        raise ValueError(f"unknown artifact: {artifact}")
    return np.clip(mask * corrupted + (1.0 - mask) * image, 0.0, 1.0)


def patch_psnr(
    clean: np.ndarray, corrupted: np.ndarray, top: int, left: int, patch_size: int
) -> float:
    """PSNR (dB) of the target patch, with MAX = 1 for images in [0, 1]."""
    rows, cols = slice(top, top + patch_size), slice(left, left + patch_size)
    error = corrupted[rows, cols] - clean[rows, cols]
    return float(10 * np.log10(1.0 / max(float(np.mean(error**2)), 1e-12)))


def calibrate_amount(
    image: np.ndarray,
    artifact: str,
    target_psnr: float,
    top: int,
    left: int,
    patch_size: int,
    feather: int,
    rng_seed: int,
    steps: int = 15,
) -> tuple[float, float]:
    """Bisection on the severity until the patch PSNR equals ``target_psnr``.

    Returns the severity and the PSNR it achieves. When the target is out of reach
    (for example blur at 25 dB with the 10 px cap), the closest severity is returned.
    """
    low, high = AMOUNT_RANGE[artifact]

    def psnr_at(amount: float) -> float:
        corrupted = inject(image, artifact, amount, top, left, patch_size, feather, rng_seed)
        return patch_psnr(image, corrupted, top, left, patch_size)

    for _ in range(steps):
        middle = (low + high) / 2
        low, high = (middle, high) if psnr_at(middle) > target_psnr else (low, middle)
    amount = (low + high) / 2
    return amount, psnr_at(amount)
