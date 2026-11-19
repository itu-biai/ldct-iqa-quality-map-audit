"""Statistics and tables from the audit outputs.

Default mode reads ``instances.csv``, ``global_deltas.csv`` and ``global_metrics.csv``
from the output folder. With ``--from-summary`` it rebuilds Table 1 from the shipped
summary files of the paper (``results/paper_5seed``) without any images or models.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


import pandas as pd

from cli import base_parser, config_from_args
from config import Config
from stats import bootstrap_ci, family_direction, family_localization
from tables import (
    background_table,
    drop_heatmap_values,
    per_seed_auc,
    table1,
    table1_markdown,
)


def analyze_instances(config: Config) -> None:
    """Compute every summary file from the per-instance audit table."""
    out = config.output_dir
    instances = pd.read_csv(out / "instances.csv")
    global_deltas = pd.read_csv(out / "global_deltas.csv")
    global_metrics = pd.read_csv(out / "global_metrics.csv")

    statistics = pd.concat(
        [family_direction(instances), family_localization(instances)], ignore_index=True
    )
    per_seed = per_seed_auc(instances)
    table = table1(per_seed, statistics, global_metrics, config.artifacts, config.alpha)

    statistics.to_csv(out / "statistics.csv", index=False)
    per_seed.to_csv(out / "per_seed_auc.csv", index=False)
    table.to_csv(out / "table1.csv", index=False)
    (out / "table1.md").write_text(table1_markdown(table, config.artifacts) + "\n")
    bootstrap = bootstrap_ci(instances, config.bootstrap_resamples, config.bootstrap_seed)
    bootstrap.to_csv(out / "bootstrap_ci.csv", index=False)
    background = background_table(instances, global_deltas, global_metrics)
    background.to_csv(out / "background_per_seed.csv", index=False)
    drop_heatmap_values(instances).to_csv(out / "heatmap_values.csv", index=False)

    print(table1_markdown(table, config.artifacts))
    print("\nBackground analysis (mean over seeds):")
    print(background.groupby("mask", sort=False).mean(numeric_only=True).round(4))


def table1_from_summary(config: Config, folder: Path) -> None:
    """Rebuild Table 1 from the shipped per-seed AUCs, test results and global metrics."""
    per_seed = pd.read_csv(folder / "per_seed_auc_5seed.csv")
    statistics = pd.read_csv(folder / "statistics_5seed.csv")
    global_metrics = pd.read_csv(folder / "global_metrics_5seed.csv")
    table = table1(per_seed, statistics, global_metrics, config.artifacts, config.alpha)
    print(table1_markdown(table, config.artifacts))


def main() -> None:
    """Run the requested analysis."""
    parser = base_parser(__doc__)
    parser.add_argument("--from-summary", type=Path, default=None)
    args = parser.parse_args()
    config = config_from_args(args)
    if args.from_summary is not None:
        table1_from_summary(config, args.from_summary)
    else:
        analyze_instances(config)


if __name__ == "__main__":
    main()
