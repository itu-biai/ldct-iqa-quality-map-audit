"""Localization (Loc) and Direction (Dir) AUC, and global MOS agreement."""

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr


def auc_against(values: np.ndarray, positive: int, negatives: np.ndarray) -> float:
    """Mann-Whitney AUC of one target value against a set of other patches (ties count 0.5)."""
    target, others = values[positive], values[negatives]
    return float(np.mean((target > others) + 0.5 * (target == others)))


def instance_aucs(
    delta: np.ndarray, positive: int, negatives: np.ndarray, signed: bool
) -> dict[str, float]:
    """Loc and Dir AUC of one injection.

    ``delta`` is the quality drop of every patch, ``s_p(x) - s_p(x')``. Loc uses the
    absolute drop; Dir uses the signed drop and is NaN for unsigned sources.
    """
    return {
        "loc_auc": auc_against(np.abs(delta), positive, negatives),
        "dir_auc": auc_against(delta, positive, negatives) if signed else float("nan"),
        "delta_in": float(delta[positive]),
        "delta_out": float(np.mean(delta[negatives])),
    }


def global_mos_metrics(prediction: np.ndarray, mos: np.ndarray) -> dict[str, float]:
    """PLCC, SROCC and KROCC between image-level predictions and MOS."""
    plcc = float(pearsonr(prediction, mos)[0])
    srocc = float(spearmanr(prediction, mos)[0])
    krocc = float(kendalltau(prediction, mos)[0])
    return {"plcc": plcc, "srocc": srocc, "krocc": krocc, "overall": plcc + srocc + krocc}
