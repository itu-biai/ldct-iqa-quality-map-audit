"""Figures of the paper: Fig. 2 (normalized drop heatmap) and Fig. 3 (example maps)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

from tables import HEATMAP_SOURCES

HEATMAP_LABELS: dict[str, str] = {
    "local_noise_reference": "Local Noise Ref.",
    "twostream_fused_roi": "Two-Stream Fused",
    "twostream_anatomy_roi": "Two-Stream Anatomy",
    "twostream_hf_roi": "Two-Stream HF",
    "patch_resnet18": "Patch ResNet18",
}
EXAMPLE_LABELS: dict[str, str] = {
    "local_noise_reference": "Ref.",
    "twostream_fused_roi": "Fused",
    "twostream_hf_roi": "HF",
    "patch_resnet18": "ResNet18",
}


def plot_drop_heatmap(
    values: pd.DataFrame,
    artifacts: tuple[str, ...],
    strengths: tuple[str, ...],
    psnr_labels: tuple[str, ...],
    path: Path,
) -> None:
    """Heatmap of the mean normalized drop (and seed std) per artifact, source and PSNR.

    Blue is a quality drop, red an increase; every panel has its own symmetric scale.
    """
    plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.titlesize": 8})
    figure, axes = plt.subplots(1, len(artifacts), figsize=(7.2, 2.4), constrained_layout=True)
    for index, (axis, artifact) in enumerate(zip(axes, artifacts, strict=True)):
        part = values[values["artifact"] == artifact]
        means = part.pivot(index="source", columns="strength", values="mean")
        stds = part.pivot(index="source", columns="strength", values="std")
        means = means.reindex(index=list(HEATMAP_SOURCES), columns=list(strengths))
        stds = stds.reindex(index=list(HEATMAP_SOURCES), columns=list(strengths))
        grid = means.to_numpy(dtype=float)
        limit = max(float(np.nanmax(np.abs(grid))) if np.isfinite(grid).any() else 0.0, 1e-6)
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        image = axis.imshow(grid, cmap="RdBu", norm=norm, aspect="auto")
        for row in range(grid.shape[0]):
            for col in range(grid.shape[1]):
                value, spread = grid[row, col], stds.iloc[row, col]
                text = "n/a" if np.isnan(value) else f"{value:.2f}"
                if np.isfinite(value) and spread > 0:
                    text += f"\n±{spread:.2f}"
                color = "white" if np.isfinite(value) and abs(value) > 0.6 * limit else "black"
                axis.text(col, row, text, ha="center", va="center", fontsize=6, color=color)
        axis.set_title(artifact.capitalize())
        axis.set_xticks(range(len(psnr_labels)))
        axis.set_xticklabels(psnr_labels)
        axis.set_xlabel("patch PSNR (dB)")
        axis.set_yticks(range(len(HEATMAP_SOURCES)) if index == 0 else [])
        if index == 0:
            axis.set_yticklabels([HEATMAP_LABELS[source] for source in HEATMAP_SOURCES])
        axis.tick_params(length=0)
        colorbar = figure.colorbar(image, ax=axis, fraction=0.07, pad=0.03)
        colorbar.ax.tick_params(labelsize=5, length=2)
        colorbar.set_label(r"$\Delta_p/\sigma_{\mathrm{clean}}$", fontsize=6)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def color_limits(panels: dict[tuple[str, str], np.ndarray], sources: list[str]) -> dict[str, float]:
    """Symmetric color limit per source: 99.5th percentile of |value| over all artifacts."""
    limits = {}
    for source in sources:
        values = [panel[~np.isnan(panel)] for (_, key), panel in panels.items() if key == source]
        limits[source] = max(float(np.percentile(np.abs(np.concatenate(values)), 99.5)), 1e-6)
    return limits


def plot_example_maps(
    perturbed: dict[str, np.ndarray],
    panels: dict[tuple[str, str], np.ndarray],
    targets: dict[str, tuple[int, int]],
    sources: list[str],
    patch_size: int,
    path: Path,
    width_in: float = 3.4,
) -> None:
    """One row per artifact: the perturbed slice and each source's drop map overlaid.

    ``panels[(artifact, source)]`` holds the pixel map of ``delta_p / sigma_clean``.
    """
    artifacts = list(perturbed)
    limits = color_limits(panels, sources)
    columns: list[tuple[str, str | None]] = [("Slice", None)]
    columns += [(EXAMPLE_LABELS.get(source, source), source) for source in sources]
    plt.rcParams.update({"font.family": "serif", "font.size": 6.5})
    height_in = width_in / len(columns) * len(artifacts) + 0.45
    figure, axes = plt.subplots(
        len(artifacts), len(columns), figsize=(width_in, height_in), squeeze=False
    )
    for row, artifact in enumerate(artifacts):
        top, left = targets[artifact]
        for col, (title, source) in enumerate(columns):
            axis = axes[row, col]
            axis.imshow(perturbed[artifact], cmap="gray", vmin=0, vmax=1)
            if source is None:
                box_color = "#c82828"
            else:
                norm = TwoSlopeNorm(vcenter=0, vmin=-limits[source], vmax=limits[source])
                axis.imshow(panels[(artifact, source)], cmap="RdBu", norm=norm, alpha=0.85)
                box_color = "black"
            box = Rectangle((left, top), patch_size, patch_size, fill=False, linewidth=0.6)
            box.set_edgecolor(box_color)
            axis.add_patch(box)
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            if row == 0:
                axis.set_title(title, pad=1.5)
            if col == 0:
                axis.set_ylabel(artifact.capitalize(), labelpad=1.5)
    figure.subplots_adjust(left=0.06, right=0.99, top=0.95, bottom=0.12, wspace=0.05, hspace=0.05)
    for col, (_, source) in enumerate(columns):
        if source is None:
            continue
        box = axes[-1, col].get_position()
        cax = figure.add_axes([box.x0 + 0.05 * box.width, 0.05, 0.9 * box.width, 0.015])
        mappable = plt.cm.ScalarMappable(
            cmap="RdBu", norm=TwoSlopeNorm(vcenter=0, vmin=-limits[source], vmax=limits[source])
        )
        colorbar = figure.colorbar(mappable, cax=cax, orientation="horizontal")
        colorbar.ax.tick_params(labelsize=5, length=1.5, pad=0.8)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=400)
    plt.close(figure)
