"""Central configuration for every experiment in the repository.

All hyperparameters live here as a frozen dataclass. Scripts import ``CONFIG`` and
override only paths or seeds from the command line with ``dataclasses.replace``.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """Settings used for the ISBI 2027 paper."""

    # Paths
    data_root: Path = Path("data/ldct_iqa_2023")
    train_mos_file: str = "train.json"
    test_mos_file: str = "test.json"
    output_dir: Path = Path("outputs")
    manifest_path: Path = Path("manifests/benchmark_manifest.csv")

    # Patch grid
    image_size: int = 512
    patch_size: int = 128
    patch_stride: int = 64
    roi_coverage_min: float = 0.9
    background_mean_threshold: float = 0.05

    # Artifact injection
    artifacts: tuple[str, ...] = ("noise", "blur", "streak", "shading")
    strength_names: tuple[str, ...] = ("low", "medium", "high")
    target_psnr_db: tuple[float, ...] = (35.0, 30.0, 25.0)
    background_psnr_db: float = 30.0
    feather_px: int = 8
    positions_per_image: int = 3
    min_mos_for_audit: float = 2.0
    perturbation_seed: int = 0
    calibration_steps: int = 15
    local_noise_sigma: float = 1.5

    # Training
    seeds: tuple[int, ...] = (42, 123, 456, 789, 1011)
    num_epochs: int = 25
    batch_size: int = 32
    global_batch_size: int = 16
    learning_rate: float = 1e-4
    lr_factor: float = 0.5
    lr_patience: int = 3
    val_fraction: float = 0.2
    split_seed: int = 42
    crop_attempts: int = 15
    use_pretrained_backbone: bool = True
    # (gradient clipping norm, learning-rate scale); 0 disables clipping. A run whose
    # predictions collapse to a constant is retried with the next setting.
    training_attempts: tuple[tuple[float, float], ...] = ((0.0, 1.0), (1.0, 1.0), (1.0, 0.5))
    collapse_min_val_srocc: float = 0.5
    collapse_min_pred_std: float = 0.05

    # Statistics
    alpha: float = 0.05
    bootstrap_resamples: int = 2000
    bootstrap_seed: int = 0

    @property
    def checkpoint_dir(self) -> Path:
        """Directory holding one sub-folder per (model, seed) run."""
        return self.output_dir / "checkpoints"

    def checkpoint_path(self, model_kind: str, seed: int) -> Path:
        """Best-on-validation checkpoint of one trained model."""
        return self.checkpoint_dir / f"{model_kind}_seed{seed}" / "best_model.pth"


CONFIG = Config()
