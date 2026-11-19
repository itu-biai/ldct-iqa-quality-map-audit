"""Global agreement (PLCC, SROCC, KROCC) of every trained model with the test MOS."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from audit import global_metrics_for_seed
from cli import base_parser, config_from_args, get_device
from data import load_mos_table


def main() -> None:
    """Write ``<output-dir>/global_metrics.csv``."""
    config = config_from_args(base_parser(__doc__).parse_args())
    device = get_device()
    test_table = load_mos_table(config.data_root, config.test_mos_file)
    frames = [global_metrics_for_seed(config, test_table, seed, device) for seed in config.seeds]
    metrics = pd.concat(frames, ignore_index=True)
    path = config.output_dir / "global_metrics.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(path, index=False)
    print(metrics.groupby("model")[["plcc", "srocc", "krocc"]].agg(["mean", "std"]).round(4))


if __name__ == "__main__":
    main()
