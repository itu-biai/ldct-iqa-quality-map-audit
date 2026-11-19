"""Paper figures.

- Fig. 2 (normalized drop heatmap) from a heatmap values CSV; no images needed.
- Fig. 3 (example maps) with ``--example-maps``; needs the data and the seed-42 models.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


import numpy as np
import pandas as pd

from artifacts import inject
from audit import build_map_sources, load_trained_models
from cli import audit_image_paths, base_parser, config_from_args, get_device
from config import CONFIG, Config
from data import read_slice
from figures import plot_drop_heatmap, plot_example_maps
from patches import grid_index, patch_coverage, patch_grid, scores_to_map
from roi import smart_roi_mask

EXAMPLE_SOURCES = [
    "local_noise_reference",
    "twostream_fused_roi",
    "twostream_hf_roi",
    "patch_resnet18",
]


def make_example_maps(config: Config, manifest: pd.DataFrame, seed: int, strength: str) -> None:
    """Fig. 3: the first slice and target patch that carry all four artifacts."""
    device = get_device()
    sources = build_map_sources(config, load_trained_models(config, seed, device), device, True)
    selected = manifest[(manifest["region"] == "anatomy") & (manifest["strength"] == strength)]
    candidates = [
        key
        for key, group in selected.groupby(["image_id", "position"])
        if set(config.artifacts) <= set(group["artifact"])
    ]
    image_id, position = candidates[0]
    rows = selected[(selected["image_id"] == image_id) & (selected["position"] == position)]
    image = read_slice(audit_image_paths(config)[str(image_id)])
    roi = smart_roi_mask(image)
    grid = patch_grid(config.image_size, config.patch_size, config.patch_stride)
    valid = patch_coverage(roi, grid, config.patch_size) >= config.roi_coverage_min
    clean = {name: sources[name].patch_scores(image, roi) for name in EXAMPLE_SOURCES}

    perturbed, panels, targets = {}, {}, {}
    for item in rows.itertuples():
        corrupted = inject(
            image,
            item.artifact,
            item.amount,
            item.top,
            item.left,
            config.patch_size,
            config.feather_px,
            item.rng_seed,
        )
        perturbed[item.artifact] = corrupted
        targets[item.artifact] = (item.top, item.left)
        target = grid_index(grid, item.top, item.left)
        for name in EXAMPLE_SOURCES:
            drop = clean[name] - sources[name].patch_scores(corrupted, roi)
            normalized = drop / (float(np.std(clean[name][valid])) + 1e-8)
            panels[(item.artifact, name)] = scores_to_map(
                normalized, grid, config.patch_size, config.image_size, valid
            )
            print(f"{item.artifact:8s} {name:24s} target drop/sigma = {normalized[target]:+.2f}")

    path = config.output_dir / "figures" / "fig3_example_maps.png"
    ordered = {artifact: perturbed[artifact] for artifact in config.artifacts}
    plot_example_maps(ordered, panels, targets, EXAMPLE_SOURCES, config.patch_size, path)
    print(f"slice {image_id}, target patch {position}, seed {seed}, strength {strength} -> {path}")


def main() -> None:
    """Draw the requested figures into ``<output-dir>/figures``."""
    parser = base_parser(__doc__)
    parser.add_argument(
        "--heatmap-values", type=Path, default=Path("results/paper_5seed/heatmap_values_5seed.csv")
    )
    parser.add_argument("--example-maps", action="store_true")
    parser.add_argument("--manifest", type=Path, default=CONFIG.manifest_path)
    parser.add_argument("--example-seed", type=int, default=42)
    parser.add_argument("--example-strength", default="medium")
    args = parser.parse_args()
    config = config_from_args(args)

    heatmap_path = config.output_dir / "figures" / "fig2_drop_heatmap.png"
    plot_drop_heatmap(
        pd.read_csv(args.heatmap_values),
        config.artifacts,
        config.strength_names,
        tuple(f"{psnr:g}" for psnr in config.target_psnr_db),
        heatmap_path,
    )
    print(f"Fig. 2 -> {heatmap_path}")
    if args.example_maps:
        manifest = pd.read_csv(args.manifest)
        make_example_maps(config, manifest, args.example_seed, args.example_strength)


if __name__ == "__main__":
    main()
