"""Check that the shipped paper results rebuild Table 1 as printed in the paper."""

from pathlib import Path

import pandas as pd
import pytest

from config import CONFIG
from tables import table1

RESULTS = Path(__file__).resolve().parents[1] / "results" / "paper_5seed"

# (source, artifact) -> (Loc mean, Dir mean, Dir dagger, Loc double dagger) from the paper.
PAPER_CELLS = {
    ("twostream_fused_roi", "noise"): (0.9942, 0.9791, True, True),
    ("twostream_fused_roi", "blur"): (0.9899, 0.4085, False, True),
    ("twostream_fused_roi", "shading"): (0.9975, 0.4430, False, False),
    ("twostream_hf_roi", "blur"): (0.9595, 0.2277, True, True),
    ("twostream_fused_noroi", "blur"): (0.9920, 0.5530, False, True),
    ("patch_resnet18", "blur"): (1.0000, 0.0000, True, False),
    ("patch_resnet18", "shading"): (1.0000, 0.5394, False, True),
    ("local_noise_reference", "shading"): (0.9950, 0.7574, True, False),
}
PAPER_SROCC = {"twostream_fused_roi": 0.9270, "twostream_fused_noroi": 0.9353}


@pytest.fixture(scope="module")
def paper_table() -> pd.DataFrame:
    """Table 1 rebuilt from the shipped summary files."""
    return table1(
        pd.read_csv(RESULTS / "per_seed_auc_5seed.csv"),
        pd.read_csv(RESULTS / "statistics_5seed.csv"),
        pd.read_csv(RESULTS / "global_metrics_5seed.csv"),
        CONFIG.artifacts,
        CONFIG.alpha,
    ).set_index("source")


@pytest.mark.parametrize(("key", "expected"), PAPER_CELLS.items())
def test_table1_cells_match_paper(paper_table: pd.DataFrame, key: tuple, expected: tuple) -> None:
    """Table 1 cells match the paper."""
    source, artifact = key
    loc, direction, dagger, double_dagger = expected
    row = paper_table.loc[source]
    assert round(row[f"{artifact}_loc_mean"], 4) == loc
    assert round(row[f"{artifact}_dir_mean"], 4) == direction
    assert (row[f"{artifact}_dir_mark"] == "†") is dagger
    assert (row[f"{artifact}_loc_mark"] == "‡") is double_dagger


@pytest.mark.parametrize(("source", "srocc"), PAPER_SROCC.items())
def test_global_srocc_matches_paper(paper_table: pd.DataFrame, source: str, srocc: float) -> None:
    """Global SROCC matches the paper."""
    assert round(paper_table.loc[source, "srocc_mean"], 4) == srocc
