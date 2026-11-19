"""Train the four audited models for every seed.

Existing checkpoints are skipped, so an interrupted run can be restarted. A run whose
predictions collapse to a constant is retried with gradient clipping and then with a
lower learning rate (``Config.training_attempts``).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from cli import base_parser, config_from_args, get_device
from data import load_mos_table, split_train_val
from preprocessing import MODEL_KINDS
from training import InputCache, Trainer


def main() -> None:
    """Train and write ``<output-dir>/training_log.csv``."""
    parser = base_parser(__doc__)
    parser.add_argument("--models", nargs="+", default=list(MODEL_KINDS), choices=MODEL_KINDS)
    args = parser.parse_args()
    config = config_from_args(args)
    device = get_device()

    table = load_mos_table(config.data_root, config.train_mos_file)
    train_table, val_table = split_train_val(table, config.val_fraction, config.split_seed)
    print(f"train {len(train_table)} | validation {len(val_table)} | device {device}")
    trainer = Trainer(config, train_table, val_table, device, InputCache())

    log_path = config.output_dir / "training_log.csv"
    logs = [pd.read_csv(log_path)] if log_path.exists() else []
    for seed in config.seeds:
        for model_kind in args.models:
            path = config.checkpoint_path(model_kind, seed)
            if path.exists():
                print(f"skip {model_kind} seed {seed}: checkpoint exists")
                continue
            for attempt, (clip_norm, lr_scale) in enumerate(config.training_attempts, start=1):
                result = trainer.train(
                    model_kind, seed, path, clip_norm, config.learning_rate * lr_scale
                )
                logs.append(pd.DataFrame(result.log).assign(attempt=attempt, clip=clip_norm))
                log_path.parent.mkdir(parents=True, exist_ok=True)
                pd.concat(logs).to_csv(log_path, index=False)
                if not result.collapsed:
                    break
                path.unlink(missing_ok=True)
                print(f"{model_kind} seed {seed}: predictions collapsed, retrying")
            else:
                raise RuntimeError(f"{model_kind} seed {seed} collapsed in every attempt")


if __name__ == "__main__":
    main()
