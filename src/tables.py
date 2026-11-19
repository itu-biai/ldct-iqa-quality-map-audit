"""Summary tables of the paper (Table 1, background analysis, Fig. 2 values)."""

import numpy as np
import pandas as pd

from audit import GRADCAM_SOURCE, REFERENCE_SOURCE

TABLE1_ROWS: dict[str, str] = {
    REFERENCE_SOURCE: "Local noise map (ref)",
    "twostream_fused_roi": "Two-stream, fused, ROI",
    "twostream_anatomy_roi": "Anatomy stream",
    "twostream_hf_roi": "HF stream",
    "twostream_fused_noroi": "Two-stream, fused, no ROI",
    "patch_resnet18": "Patch ResNet18",
    GRADCAM_SOURCE: "Global ResNet18 + Grad-CAM",
}

# Which model's global SROCC is shown on each Table 1 row.
TABLE1_GLOBAL_MODEL: dict[str, str] = {
    "twostream_fused_roi": "twostream_roi",
    "twostream_fused_noroi": "twostream_noroi",
    "patch_resnet18": "patch_resnet18",
    GRADCAM_SOURCE: "global_resnet18",
}

HEATMAP_SOURCES: tuple[str, ...] = (
    REFERENCE_SOURCE,
    "twostream_fused_roi",
    "twostream_anatomy_roi",
    "twostream_hf_roi",
    "patch_resnet18",
)


def per_seed_auc(instances: pd.DataFrame) -> pd.DataFrame:
    """Mean Loc and Dir AUC per (source, seed, artifact) over anatomy targets."""
    anatomy = instances[instances["region"] == "anatomy"]
    columns = ["source", "seed", "artifact"]
    return anatomy.groupby(columns)[["loc_auc", "dir_auc"]].mean().reset_index()


def consistent_direction(per_seed: pd.DataFrame) -> set[tuple[str, str]]:
    """(source, artifact) pairs whose Dir AUC is on the same side of 0.5 in every seed."""
    consistent = set()
    for (source, artifact), group in per_seed.groupby(["source", "artifact"]):
        below = int((group["dir_auc"] < 0.5).sum())
        if below in (0, len(group)):
            consistent.add((source, artifact))
    return consistent


def table1_marks(
    per_seed: pd.DataFrame, statistics: pd.DataFrame, alpha: float
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """Cells marked with a dagger (Dir) and a double dagger (Loc).

    Dagger: Dir differs from 0.5 (Holm p < alpha) and has the same side in all seeds.
    Double dagger: Loc differs from the local noise reference (Holm p < alpha).
    """
    significant = statistics[statistics["p_holm"] < alpha]
    family_a = significant[significant["family"] == "A"]
    family_b = significant[significant["family"] == "B"]
    dagger = set(zip(family_a["source"], family_a["artifact"], strict=True))
    dagger &= consistent_direction(per_seed)
    double_dagger = set(zip(family_b["source"], family_b["artifact"], strict=True))
    return dagger, double_dagger


def table1(
    per_seed: pd.DataFrame,
    statistics: pd.DataFrame,
    global_metrics: pd.DataFrame,
    artifacts: tuple[str, ...],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Table 1: mean and seed std of Loc/Dir AUC plus global SROCC, with test marks."""
    dagger, double_dagger = table1_marks(per_seed, statistics, alpha)
    means = per_seed.groupby(["source", "artifact"])[["loc_auc", "dir_auc"]].mean()
    stds = per_seed.groupby(["source", "artifact"])[["loc_auc", "dir_auc"]].std()
    srocc = global_metrics.groupby("model")["srocc"].agg(["mean", "std"])
    rows = []
    for source, label in TABLE1_ROWS.items():
        row: dict[str, object] = {"source": source, "map_source": label}
        model = TABLE1_GLOBAL_MODEL.get(source)
        row["srocc_mean"] = srocc.loc[model, "mean"] if model else np.nan
        row["srocc_std"] = srocc.loc[model, "std"] if model else np.nan
        for artifact in artifacts:
            for metric, short in (("loc_auc", "loc"), ("dir_auc", "dir")):
                mean = means.loc[(source, artifact), metric]
                if metric == "dir_auc" and source == GRADCAM_SOURCE:
                    mean = np.nan  # an attribution has no quality direction
                row[f"{artifact}_{short}_mean"] = mean
                row[f"{artifact}_{short}_std"] = stds.loc[(source, artifact), metric]
            row[f"{artifact}_dir_mark"] = "†" if (source, artifact) in dagger else ""
            row[f"{artifact}_loc_mark"] = "‡" if (source, artifact) in double_dagger else ""
        rows.append(row)
    return pd.DataFrame(rows)


def table1_markdown(table: pd.DataFrame, artifacts: tuple[str, ...]) -> str:
    """Render :func:`table1` as a Markdown table (mean ± seed std)."""

    def cell(mean: float, std: float, mark: str) -> str:
        if np.isnan(mean):
            return "n/a"
        text = f"{mean:.4f}" + (f" ± {std:.4f}" if np.isfinite(std) else "")
        return text + mark

    header = ["Map source", "Global SROCC"]
    header += [f"{a.capitalize()} {m}" for a in artifacts for m in ("Loc", "Dir")]
    lines = ["| " + " | ".join(header) + " |", "|---" * len(header) + "|"]
    for row in table.itertuples(index=False):
        values = row._asdict()
        cells = [values["map_source"], cell(values["srocc_mean"], values["srocc_std"], "")]
        cells = [c if c != "n/a" else "-" for c in cells]
        for artifact in artifacts:
            cells.append(
                cell(
                    values[f"{artifact}_loc_mean"],
                    values[f"{artifact}_loc_std"],
                    values[f"{artifact}_loc_mark"],
                )
            )
            cells.append(
                cell(
                    values[f"{artifact}_dir_mean"],
                    values[f"{artifact}_dir_std"],
                    values[f"{artifact}_dir_mark"],
                )
            )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _noise_medium_response(frame: pd.DataFrame) -> pd.Series:
    selected = frame[(frame["artifact"] == "noise") & (frame["strength"] == "medium")]
    return selected["delta_in"].abs() / selected["clean_score_std"]


def background_table(
    instances: pd.DataFrame, global_deltas: pd.DataFrame, global_metrics: pd.DataFrame
) -> pd.DataFrame:
    """Per-seed background analysis of the two-stream model with and without the mask.

    ``anatomy_rho`` and ``background_rho`` are the mean normalized response
    ``|delta_p| / sigma_clean`` to 30 dB noise in anatomy and in air;
    ``background_global_delta`` is the drop of the image-level score for noise in air.
    """
    variants = {
        "twostream_fused_roi": ("Smart ROI", "twostream_roi"),
        "twostream_fused_noroi": ("No mask", "twostream_noroi"),
    }
    records = []
    for source, (label, model) in variants.items():
        for seed in sorted(instances["seed"].unique()):
            selected = instances[(instances["source"] == source) & (instances["seed"] == seed)]
            anatomy = selected[selected["region"] == "anatomy"]
            background = selected[selected["region"] == "background"]
            deltas = global_deltas[
                (global_deltas["source"] == source)
                & (global_deltas["seed"] == seed)
                & (global_deltas["region"] == "background")
            ]
            metrics = global_metrics[
                (global_metrics["model"] == model) & (global_metrics["seed"] == seed)
            ].iloc[0]
            anatomy_response = _noise_medium_response(anatomy)
            background_response = _noise_medium_response(background)
            records.append(
                {
                    "mask": label,
                    "seed": seed,
                    "plcc": metrics["plcc"],
                    "srocc": metrics["srocc"],
                    "noise_dir_anatomy": anatomy[anatomy["artifact"] == "noise"]["dir_auc"].mean(),
                    "anatomy_rho": anatomy_response.mean(),
                    "background_rho": background_response.mean(),
                    "background_global_delta": deltas["global_delta"].mean(),
                    "n_anatomy": len(anatomy_response),
                    "n_background": len(background_response),
                }
            )
    return pd.DataFrame(records)


def drop_heatmap_values(instances: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
    """Normalized drop of the target patch per artifact, source and strength (Fig. 2)."""
    anatomy = instances[
        (instances["region"] == "anatomy") & instances["source"].isin(HEATMAP_SOURCES)
    ].copy()
    anatomy["z"] = anatomy["delta_in"] / (anatomy["clean_score_std"] + eps)
    columns = ["artifact", "source", "seed", "strength"]
    per_seed = anatomy.groupby(columns)["z"].mean().reset_index()
    summary = (
        per_seed.groupby(["artifact", "source", "strength"])["z"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary["std"] = summary["std"].fillna(0.0)
    return summary
