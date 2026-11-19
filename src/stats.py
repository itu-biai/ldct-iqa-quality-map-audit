"""Hypothesis tests, multiple-comparison correction and bootstrap intervals.

The unit of analysis is the slice: instance AUCs are first averaged per slice over
target positions, strengths and seeds (anatomy targets only).
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

from audit import GRADCAM_SOURCE, REFERENCE_SOURCE


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (same order as the input)."""
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(running_max, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running_max
    return adjusted.tolist()


def rank_biserial(differences: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation (effect size of the Wilcoxon test)."""
    nonzero = differences[differences != 0]
    if len(nonzero) == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero))
    positive, negative = ranks[nonzero > 0].sum(), ranks[nonzero < 0].sum()
    return float((positive - negative) / (positive + negative))


def safe_wilcoxon(differences: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value; 1.0 when every difference is zero."""
    return float(wilcoxon(differences).pvalue) if np.any(differences != 0) else 1.0


def per_slice(instances: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mean of ``metric`` per (source, artifact, slice) over anatomy targets."""
    anatomy = instances[instances["region"] == "anatomy"]
    return anatomy.groupby(["source", "artifact", "image_id"])[metric].mean().reset_index()


def family_direction(instances: pd.DataFrame) -> pd.DataFrame:
    """Family A: is Dir AUC different from 0.5? (signed sources x artifacts, Holm)."""
    signed = instances[instances["source"] != GRADCAM_SOURCE]
    rows = []
    for (source, artifact), group in (
        per_slice(signed, "dir_auc").dropna().groupby(["source", "artifact"])
    ):
        differences = group["dir_auc"].to_numpy() - 0.5
        rows.append(
            {
                "family": "A",
                "source": source,
                "artifact": artifact,
                "mean_auc": float(group["dir_auc"].mean()),
                "p_raw": safe_wilcoxon(differences),
                "effect_size_r": rank_biserial(differences),
            }
        )
    frame = pd.DataFrame(rows)
    frame["p_holm"] = holm_adjust(frame["p_raw"].tolist())
    return frame


def family_localization(instances: pd.DataFrame) -> pd.DataFrame:
    """Family B: does Loc AUC differ from the local noise reference? (Holm)."""
    wide = per_slice(instances, "loc_auc").pivot_table(
        index=["artifact", "image_id"], columns="source", values="loc_auc"
    )
    rows = []
    for artifact, group in wide.groupby(level="artifact"):
        for source in [name for name in group.columns if name != REFERENCE_SOURCE]:
            differences = (group[source] - group[REFERENCE_SOURCE]).dropna().to_numpy()
            rows.append(
                {
                    "family": "B",
                    "source": source,
                    "artifact": artifact,
                    "median_difference": float(np.median(differences)),
                    "p_raw": safe_wilcoxon(differences),
                    "effect_size_r": rank_biserial(differences),
                }
            )
    frame = pd.DataFrame(rows)
    frame["p_holm"] = holm_adjust(frame["p_raw"].tolist())
    return frame


def bootstrap_ci(
    instances: pd.DataFrame, resamples: int, seed: int, level: float = 0.95
) -> pd.DataFrame:
    """Percentile bootstrap interval of mean Loc and Dir AUC, resampling slices."""
    rng = np.random.default_rng(seed)
    anatomy = instances[instances["region"] == "anatomy"]
    tail = 100 * (1 - level) / 2
    rows = []
    for (source, artifact), group in anatomy.groupby(["source", "artifact"]):
        for metric in ("loc_auc", "dir_auc"):
            values = group.groupby("image_id")[metric].mean().dropna().to_numpy()
            if len(values) == 0:
                continue
            means = values[rng.integers(0, len(values), (resamples, len(values)))].mean(axis=1)
            rows.append(
                {
                    "source": source,
                    "artifact": artifact,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "ci_low": float(np.percentile(means, tail)),
                    "ci_high": float(np.percentile(means, 100 - tail)),
                    "n_images": len(values),
                }
            )
    return pd.DataFrame(rows)
