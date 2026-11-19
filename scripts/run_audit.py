"""Run the localized injection audit for every seed.

One file per seed is written to ``<output-dir>/parts``; finished seeds are skipped.
The training-free reference map does not depend on the seed and is computed once.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse

import pandas as pd

from audit import run_audit_for_seed
from cli import audit_image_paths, base_parser, config_from_args, get_device
from config import CONFIG


def main() -> None:
    """Write ``instances.csv`` and ``global_deltas.csv`` to the output folder."""
    parser = base_parser(__doc__)
    parser.add_argument("--manifest", type=Path, default=CONFIG.manifest_path)
    args: argparse.Namespace = parser.parse_args()
    config = config_from_args(args)
    device = get_device()
    manifest = pd.read_csv(args.manifest)
    image_paths = audit_image_paths(config)
    parts = config.output_dir / "parts"
    parts.mkdir(parents=True, exist_ok=True)

    for index, seed in enumerate(config.seeds):
        instance_path = parts / f"instances_seed{seed}.csv"
        delta_path = parts / f"global_deltas_seed{seed}.csv"
        if instance_path.exists() and delta_path.exists():
            print(f"seed {seed}: already done")
            continue
        instances, deltas = run_audit_for_seed(
            config, manifest, image_paths, seed, device, include_reference=index == 0
        )
        deltas.to_csv(delta_path, index=False)
        instances.to_csv(instance_path, index=False)  # written last: marks the seed as done

    instances = pd.concat([pd.read_csv(parts / f"instances_seed{s}.csv") for s in config.seeds])
    deltas = pd.concat([pd.read_csv(parts / f"global_deltas_seed{s}.csv") for s in config.seeds])
    collapsed = instances.groupby(["source", "seed"])["clean_score_std"].max()
    if (collapsed <= 0).any():
        raise RuntimeError(f"constant-output map sources:\n{collapsed[collapsed <= 0]}")
    instances.to_csv(config.output_dir / "instances.csv", index=False)
    deltas.to_csv(config.output_dir / "global_deltas.csv", index=False)
    print(instances.groupby(["source", "seed"]).size().unstack())


if __name__ == "__main__":
    main()
