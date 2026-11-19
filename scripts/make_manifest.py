"""Build the injection manifest (target patches and calibrated severities).

The manifest used in the paper is shipped as ``manifests/benchmark_manifest.csv``.
This script documents how it was made and writes a new one to the output folder.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audit import build_manifest
from cli import base_parser, config_from_args
from data import load_mos_table


def main() -> None:
    """Write ``<output-dir>/benchmark_manifest.csv``."""
    parser = base_parser(__doc__)
    config = config_from_args(parser.parse_args())
    test_table = load_mos_table(config.data_root, config.test_mos_file)
    manifest = build_manifest(config, test_table)
    path = config.output_dir / "benchmark_manifest.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(path, index=False)
    print(f"Wrote {len(manifest)} injections for {manifest['image_id'].nunique()} slices to {path}")


if __name__ == "__main__":
    main()
