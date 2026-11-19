"""Small helpers shared by the command-line scripts."""

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch

from config import CONFIG, Config
from data import load_mos_table


def base_parser(description: str) -> argparse.ArgumentParser:
    """Parser with the options every script accepts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-root", type=Path, default=CONFIG.data_root)
    parser.add_argument("--output-dir", type=Path, default=CONFIG.output_dir)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(CONFIG.seeds))
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    """Apply command-line paths and seeds to the default configuration."""
    return replace(
        CONFIG, data_root=args.data_root, output_dir=args.output_dir, seeds=tuple(args.seeds)
    )


def get_device() -> torch.device:
    """GPU if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def audit_image_paths(config: Config) -> dict[str, Path]:
    """Map every test slice id to its file path."""
    table = load_mos_table(config.data_root, config.test_mos_file)
    return dict(zip(table["image_id"].astype(str), table["path"], strict=True))


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    """Read a CSV file, or return None when it does not exist."""
    return pd.read_csv(path) if path.exists() else None
