"""Patch grid helpers shared by training, scoring and the audit."""

import numpy as np

PatchGrid = list[tuple[int, int]]


def patch_grid(image_size: int, patch_size: int, stride: int) -> PatchGrid:
    """Top-left corners of all patches, row by row (49 patches for 512/128/64)."""
    starts = range(0, image_size - patch_size + 1, stride)
    return [(top, left) for top in starts for left in starts]


def patch_coverage(mask: np.ndarray, grid: PatchGrid, patch_size: int) -> np.ndarray:
    """Fraction of each patch that lies inside ``mask``."""
    return np.array(
        [mask[top : top + patch_size, left : left + patch_size].mean() for top, left in grid]
    )


def patch_overlaps(grid: PatchGrid, top: int, left: int, patch_size: int) -> np.ndarray:
    """Number of pixels each grid patch shares with the patch at (``top``, ``left``)."""
    rows = np.array([row for row, _ in grid])
    cols = np.array([col for _, col in grid])
    overlap_rows = np.clip(np.minimum(rows, top) + patch_size - np.maximum(rows, top), 0, None)
    overlap_cols = np.clip(np.minimum(cols, left) + patch_size - np.maximum(cols, left), 0, None)
    return overlap_rows * overlap_cols


def grid_index(grid: PatchGrid, top: int, left: int) -> int:
    """Index of the grid patch whose top-left corner is (``top``, ``left``)."""
    return grid.index((top, left))


def scores_to_map(
    scores: np.ndarray,
    grid: PatchGrid,
    patch_size: int,
    image_size: int,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    """Spread patch values to a pixel map by averaging overlapping patches.

    Pixels covered by no valid patch are NaN.
    """
    valid = np.ones(len(grid), dtype=bool) if valid is None else valid
    total = np.zeros((image_size, image_size), dtype=np.float64)
    count = np.zeros((image_size, image_size), dtype=np.float64)
    for (top, left), value, keep in zip(grid, scores, valid, strict=True):
        if keep:
            total[top : top + patch_size, left : left + patch_size] += value
            count[top : top + patch_size, left : left + patch_size] += 1
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)
