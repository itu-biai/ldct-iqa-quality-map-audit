"""The localized injection protocol: manifest, per-instance AUCs and global deltas."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from artifacts import calibrate_amount, inject
from config import Config
from data import read_slice
from map_sources import GradCamMap, LocalNoiseMap, MapSource, ModelPatchScorer
from metrics import global_mos_metrics, instance_aucs
from models import load_model
from patches import patch_coverage, patch_grid, patch_overlaps
from preprocessing import MODEL_KINDS
from roi import smart_roi_mask

REFERENCE_SOURCE = "local_noise_reference"
GRADCAM_SOURCE = "global_resnet18_gradcam"

# Image-level score used for each model's global SROCC.
GLOBAL_SOURCE_OF_MODEL: dict[str, str] = {
    "twostream_roi": "twostream_fused_roi",
    "twostream_noroi": "twostream_fused_noroi",
    "patch_resnet18": "patch_resnet18",
    "global_resnet18": "global_resnet18",
}

GLOBAL_DELTA_COLUMNS = [
    "source",
    "seed",
    "image_id",
    "region",
    "artifact",
    "strength",
    "global_delta",
]


def pick_positions(
    coverage: np.ndarray,
    count: int,
    rng: np.random.Generator,
    min_coverage: float,
    max_coverage: float,
) -> list[int]:
    """Draw ``count`` distinct grid indices whose ROI coverage lies in the given range."""
    candidates = np.flatnonzero((coverage >= min_coverage) & (coverage <= max_coverage))
    if len(candidates) < count:
        return []
    return sorted(int(index) for index in rng.choice(candidates, size=count, replace=False))


def build_manifest(config: Config, test_table: pd.DataFrame) -> pd.DataFrame:
    """List every injection: slice, target patch, artifact, strength and calibrated severity.

    For each test slice with MOS >= ``config.min_mos_for_audit``: three target patches
    with at least 90% ROI coverage get all four artifacts at 35, 30 and 25 dB, and one
    patch fully in air (if any) gets noise at 30 dB.
    """
    grid = patch_grid(config.image_size, config.patch_size, config.patch_stride)
    selected = test_table[test_table["mos"] >= config.min_mos_for_audit].reset_index(drop=True)
    rows = []
    for image_index, row in enumerate(
        tqdm(selected.itertuples(), total=len(selected), desc="manifest")
    ):
        image = read_slice(row.path)
        coverage = patch_coverage(smart_roi_mask(image), grid, config.patch_size)
        rng = np.random.default_rng(config.perturbation_seed + image_index)
        regions = {
            "anatomy": pick_positions(
                coverage, config.positions_per_image, rng, config.roi_coverage_min, 1.0
            ),
            "background": pick_positions(coverage, 1, rng, 0.0, 0.0),
        }
        for region, positions in regions.items():
            if region == "anatomy":
                artifacts = config.artifacts
                strengths = list(zip(config.strength_names, config.target_psnr_db, strict=True))
            else:
                artifacts, strengths = ("noise",), [("medium", config.background_psnr_db)]
            for position in positions:
                top, left = grid[position]
                rng_seed = config.perturbation_seed * 100_000 + image_index * 100 + position
                for strength, target in strengths:
                    for artifact in artifacts:
                        amount, achieved = calibrate_amount(
                            image,
                            artifact,
                            target,
                            top,
                            left,
                            config.patch_size,
                            config.feather_px,
                            rng_seed,
                            steps=config.calibration_steps,
                        )
                        rows.append(
                            {
                                "image_id": row.image_id,
                                "mos": row.mos,
                                "region": region,
                                "position": position,
                                "top": top,
                                "left": left,
                                "artifact": artifact,
                                "strength": strength,
                                "target_psnr": target,
                                "achieved_psnr": achieved,
                                "amount": amount,
                                "rng_seed": rng_seed,
                            }
                        )
    return pd.DataFrame(rows)


def load_trained_models(
    config: Config, seed: int, device: torch.device
) -> dict[str, torch.nn.Module]:
    """Load the four trained models of one seed."""
    return {
        kind: load_model(kind, config.checkpoint_path(kind, seed), device) for kind in MODEL_KINDS
    }


def build_global_scorers(
    config: Config, models: dict[str, torch.nn.Module], device: torch.device
) -> dict[str, ModelPatchScorer]:
    """Scorers that give each model's image-level prediction."""
    grid = patch_grid(config.image_size, config.patch_size, config.patch_stride)
    return {
        source: ModelPatchScorer(
            source,
            kind,
            models[kind],
            grid,
            config.patch_size,
            device,
            background_mean_threshold=config.background_mean_threshold,
        )
        for kind, source in GLOBAL_SOURCE_OF_MODEL.items()
    }


def build_map_sources(
    config: Config,
    models: dict[str, torch.nn.Module],
    device: torch.device,
    include_reference: bool,
) -> dict[str, MapSource]:
    """The audited map sources of one seed (the training-free reference is optional)."""
    grid = patch_grid(config.image_size, config.patch_size, config.patch_stride)

    def scorer(name: str, kind: str, variant: str = "fused") -> ModelPatchScorer:
        return ModelPatchScorer(
            name,
            kind,
            models[kind],
            grid,
            config.patch_size,
            device,
            variant=variant,
            background_mean_threshold=config.background_mean_threshold,
        )

    sources: dict[str, MapSource] = {}
    if include_reference:
        sources[REFERENCE_SOURCE] = LocalNoiseMap(grid, config.patch_size, config.local_noise_sigma)
    sources["twostream_fused_roi"] = scorer("twostream_fused_roi", "twostream_roi")
    sources["twostream_anatomy_roi"] = scorer("twostream_anatomy_roi", "twostream_roi", "anatomy")
    sources["twostream_hf_roi"] = scorer("twostream_hf_roi", "twostream_roi", "hf")
    sources["twostream_fused_noroi"] = scorer("twostream_fused_noroi", "twostream_noroi")
    sources["patch_resnet18"] = scorer("patch_resnet18", "patch_resnet18")
    sources[GRADCAM_SOURCE] = GradCamMap(models["global_resnet18"], grid, config.patch_size, device)
    return sources


def global_metrics_for_seed(
    config: Config, test_table: pd.DataFrame, seed: int, device: torch.device
) -> pd.DataFrame:
    """PLCC, SROCC and KROCC of the four models of one seed on the full test set."""
    scorers = build_global_scorers(config, load_trained_models(config, seed, device), device)
    predictions: dict[str, list[float]] = {kind: [] for kind in GLOBAL_SOURCE_OF_MODEL}
    for row in tqdm(test_table.itertuples(), total=len(test_table), desc=f"test seed {seed}"):
        image = read_slice(row.path)
        roi = smart_roi_mask(image)
        for kind, source in GLOBAL_SOURCE_OF_MODEL.items():
            predictions[kind].append(scorers[source].global_score(image, roi))
    mos = test_table["mos"].to_numpy()
    rows = [
        {"model": kind, "seed": seed, **global_mos_metrics(np.array(values), mos)}
        for kind, values in predictions.items()
    ]
    return pd.DataFrame(rows)


def run_audit_for_seed(
    config: Config,
    manifest: pd.DataFrame,
    image_paths: dict[str, Path],
    seed: int,
    device: torch.device,
    include_reference: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every manifest injection through every map source of one seed.

    Returns the per-instance table (Loc/Dir AUC, drops) and the global score change of
    the two-stream models for injections into background air.
    """
    models = load_trained_models(config, seed, device)
    sources = build_map_sources(config, models, device, include_reference)
    global_scorers = {
        name: scorer
        for name, scorer in build_global_scorers(config, models, device).items()
        if name in ("twostream_fused_roi", "twostream_fused_noroi")
    }
    grid = patch_grid(config.image_size, config.patch_size, config.patch_stride)
    instance_rows, global_rows = [], []

    for image_id, group in tqdm(manifest.groupby("image_id"), desc=f"audit seed {seed}"):
        image = read_slice(image_paths[str(image_id)])
        roi = smart_roi_mask(image)
        valid = patch_coverage(roi, grid, config.patch_size) >= config.roi_coverage_min
        clean = {name: source.patch_scores(image, roi) for name, source in sources.items()}
        clean_global = {name: s.global_score(image, roi) for name, s in global_scorers.items()}
        for item in group.itertuples():
            perturbed = inject(
                image,
                item.artifact,
                item.amount,
                item.top,
                item.left,
                config.patch_size,
                config.feather_px,
                item.rng_seed,
            )
            no_overlap = patch_overlaps(grid, item.top, item.left, config.patch_size) == 0
            negatives = np.flatnonzero(no_overlap & valid)
            for name, source in sources.items():
                scores = source.patch_scores(perturbed, roi)
                delta = clean[name] - scores if source.signed else scores - clean[name]
                # Dir is also stored for Grad-CAM, but it is not reported (no quality direction).
                aucs = instance_aucs(
                    delta, item.position, negatives, source.signed or name == GRADCAM_SOURCE
                )
                instance_rows.append(
                    {
                        "source": name,
                        "seed": seed,
                        "image_id": image_id,
                        "region": item.region,
                        "position": item.position,
                        "artifact": item.artifact,
                        "strength": item.strength,
                        "achieved_psnr": item.achieved_psnr,
                        "clean_score_std": float(np.std(clean[name][valid])),
                        **aucs,
                    }
                )
            if item.region == "background":
                for name, scorer in global_scorers.items():
                    global_rows.append(
                        {
                            "source": name,
                            "seed": seed,
                            "image_id": image_id,
                            "region": item.region,
                            "artifact": item.artifact,
                            "strength": item.strength,
                            "global_delta": clean_global[name]
                            - scorer.global_score(perturbed, roi),
                        }
                    )
    return pd.DataFrame(instance_rows), pd.DataFrame(global_rows, columns=GLOBAL_DELTA_COLUMNS)
