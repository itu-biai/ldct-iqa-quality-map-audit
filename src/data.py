"""Loading the LDCT-IQA 2023 slices and their mean opinion scores (MOS)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from sklearn.model_selection import train_test_split


def find_image_dir(data_root: Path, mos_file: str) -> Path:
    """Return the folder that holds the images listed in ``mos_file``.

    The public release stores training and test slices in different sub-folders
    (for example ``image`` and ``test images``), so the folder is found by looking
    for the first listed file name.
    """
    with (data_root / mos_file).open() as handle:
        first_name = next(iter(json.load(handle)))
    candidates = [data_root] + sorted(path for path in data_root.iterdir() if path.is_dir())
    for folder in candidates:
        if (folder / first_name).exists():
            return folder
    raise FileNotFoundError(f"{first_name} (from {mos_file}) was not found under {data_root}")


def load_mos_table(data_root: Path, mos_file: str) -> pd.DataFrame:
    """Read a MOS json file into a table with ``image_id``, ``path`` and ``mos`` columns.

    Rows keep the order of the json file.
    """
    with (data_root / mos_file).open() as handle:
        scores: dict[str, float] = json.load(handle)
    image_dir = find_image_dir(data_root, mos_file)
    return pd.DataFrame(
        {
            "image_id": [Path(name).stem for name in scores],
            "path": [image_dir / name for name in scores],
            "mos": [float(value) for value in scores.values()],
        }
    )


def read_slice(path: Path | str) -> np.ndarray:
    """Read one slice and min-max normalize it to [0, 1] (per image)."""
    image = tifffile.imread(path).astype(np.float32)
    low, high = float(image.min()), float(image.max())
    if high > low:
        return (image - low) / (high - low)
    return np.zeros_like(image)


def split_train_val(
    table: pd.DataFrame, val_fraction: float, split_seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random slice-level split of the training set into training and validation parts.

    The split is identical for every training seed. Patient identifiers are not
    distributed with the dataset, so a patient-level split was not possible.
    """
    train, val = train_test_split(table, test_size=val_fraction, random_state=split_seed)
    return train.reset_index(drop=True), val.reset_index(drop=True)
