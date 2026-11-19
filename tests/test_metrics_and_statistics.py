"""Tests for the AUC metrics and the statistical helpers."""

import numpy as np
import pandas as pd
import pytest

from metrics import auc_against, global_mos_metrics, instance_aucs
from stats import bootstrap_ci, holm_adjust, rank_biserial, safe_wilcoxon


def test_auc_is_one_when_target_is_largest_and_half_for_ties() -> None:
    """AUC is 1 when the target is largest and 0.5 for ties."""
    values = np.array([5.0, 1.0, 2.0, 3.0])
    assert auc_against(values, 0, np.array([1, 2, 3])) == 1.0
    assert auc_against(np.ones(4), 0, np.array([1, 2, 3])) == 0.5


def test_dir_below_half_when_target_quality_rises() -> None:
    """Dir below half when target quality rises."""
    delta = np.array([-0.4, 0.0, 0.0, 0.0])  # the target patch got better
    aucs = instance_aucs(delta, 0, np.array([1, 2, 3]), signed=True)
    assert aucs["loc_auc"] == 1.0
    assert aucs["dir_auc"] == 0.0


def test_unsigned_source_has_no_direction() -> None:
    """Unsigned source has no direction."""
    aucs = instance_aucs(np.array([1.0, 0.0]), 0, np.array([1]), signed=False)
    assert np.isnan(aucs["dir_auc"])


def test_holm_matches_hand_computation() -> None:
    """Holm matches hand computation."""
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])


def test_rank_biserial_extremes_and_wilcoxon_on_zeros() -> None:
    """Rank-biserial extremes and the Wilcoxon test on zeros."""
    assert rank_biserial(np.array([1.0, 2.0, 3.0])) == 1.0
    assert rank_biserial(np.array([-1.0, -2.0])) == -1.0
    assert safe_wilcoxon(np.zeros(5)) == 1.0


def test_global_metrics_perfect_ranking() -> None:
    """Global metrics perfect ranking."""
    mos = np.array([1.0, 2.0, 3.0, 4.0])
    metrics = global_mos_metrics(mos * 2, mos)
    assert metrics["srocc"] == pytest.approx(1.0)
    assert metrics["krocc"] == pytest.approx(1.0)


def test_bootstrap_interval_contains_mean() -> None:
    """Bootstrap interval contains mean."""
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "source": "s",
            "artifact": "noise",
            "region": "anatomy",
            "image_id": np.repeat(np.arange(30), 3),
            "loc_auc": rng.uniform(0.8, 1.0, 90),
            "dir_auc": rng.uniform(0.3, 0.7, 90),
        }
    )
    intervals = bootstrap_ci(frame, resamples=500, seed=0)
    for row in intervals.itertuples():
        assert row.ci_low <= row.mean <= row.ci_high
        assert row.n_images == 30
