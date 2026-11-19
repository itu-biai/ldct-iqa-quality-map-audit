"""Tests for the patch grid helpers."""

import numpy as np

from patches import grid_index, patch_coverage, patch_grid, patch_overlaps, scores_to_map


def test_grid_has_49_patches_for_paper_setting() -> None:
    """Grid has 49 patches for paper setting."""
    grid = patch_grid(512, 128, 64)
    assert len(grid) == 49
    assert grid[0] == (0, 0)
    assert grid[-1] == (384, 384)


def test_overlap_is_full_for_same_patch_and_zero_for_distant_patch() -> None:
    """Overlap is full for same patch and zero for distant patch."""
    grid = patch_grid(512, 128, 64)
    overlaps = patch_overlaps(grid, 128, 128, 128)
    assert overlaps[grid_index(grid, 128, 128)] == 128 * 128
    assert overlaps[grid_index(grid, 128, 192)] == 128 * 64
    assert overlaps[grid_index(grid, 384, 384)] == 0


def test_coverage_of_half_mask() -> None:
    """Coverage of half mask."""
    grid = patch_grid(512, 128, 64)
    mask = np.zeros((512, 512))
    mask[:, :256] = 1
    coverage = patch_coverage(mask, grid, 128)
    assert coverage[grid_index(grid, 0, 0)] == 1.0
    assert coverage[grid_index(grid, 0, 192)] == 0.5
    assert coverage[grid_index(grid, 0, 384)] == 0.0


def test_scores_to_map_averages_overlaps_and_marks_invalid_as_nan() -> None:
    """Scores to map averages overlaps and marks invalid as NaN."""
    grid = patch_grid(256, 128, 64)
    scores = np.arange(len(grid), dtype=float)
    valid = np.ones(len(grid), dtype=bool)
    valid[-1] = False
    pixel_map = scores_to_map(scores, grid, 128, 256, valid)
    assert pixel_map[0, 0] == 0.0
    assert pixel_map[100, 100] == np.mean(scores[[0, 1, 3, 4]])
    assert np.isnan(pixel_map[255, 255])
