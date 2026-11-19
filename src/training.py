"""Training loop for the audited models.

Patch models are trained on random 128 x 128 crops and validated on the mean score of
the grid patches of each validation slice; the global ResNet18 sees whole slices.
The checkpoint with the lowest validation loss is kept.

Batches are drawn with a NumPy generator instead of a ``DataLoader`` so that the
order of slices and crop positions depends only on the seed, as in the paper runs.
"""

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch import nn, optim
from tqdm import tqdm

from config import Config
from data import read_slice
from models import build_model, forward_model
from patches import PatchGrid, patch_grid
from preprocessing import prepare_input


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch and make cuDNN deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class InputCache:
    """Prepared channel stacks of each slice, computed once per model kind."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], np.ndarray] = {}

    def get(self, model_kind: str, path: Path | str) -> np.ndarray:
        """Return the channel stack of ``path`` for ``model_kind``."""
        key = (model_kind, str(path))
        if key not in self._cache:
            self._cache[key] = prepare_input(model_kind, read_slice(path))
        return self._cache[key]


def random_crop(
    stack: np.ndarray,
    rng: np.random.Generator,
    size: int,
    attempts: int,
    background_threshold: float,
) -> np.ndarray:
    """Random crop; retried up to ``attempts`` times while the first channel is mostly air.

    With ``attempts == 1`` no background filtering is applied.
    """
    height, width = stack.shape[-2:]
    patch = stack
    for _ in range(attempts):
        top = int(rng.integers(0, height - size + 1))
        left = int(rng.integers(0, width - size + 1))
        patch = stack[..., top : top + size, left : left + size]
        if attempts == 1 or patch[0].mean() > background_threshold:
            return patch
    return patch


def grid_patches(
    stack: np.ndarray,
    grid: PatchGrid,
    size: int,
    background_threshold: float | None,
) -> np.ndarray | None:
    """All grid patches of a slice, optionally without background patches."""
    pieces = [stack[..., top : top + size, left : left + size] for top, left in grid]
    if background_threshold is not None:
        pieces = [piece for piece in pieces if piece[0].mean() > background_threshold]
    return np.stack(pieces) if pieces else None


@dataclass
class TrainingResult:
    """Summary of one training run."""

    best_val_loss: float
    best_val_srocc: float
    collapsed: bool
    log: list[dict] = field(default_factory=list)


class Trainer:
    """Train one model kind with one seed on fixed training and validation tables."""

    def __init__(
        self,
        config: Config,
        train_table: pd.DataFrame,
        val_table: pd.DataFrame,
        device: torch.device,
        cache: InputCache | None = None,
    ) -> None:
        self.config, self.device = config, device
        self.train_table, self.val_table = train_table, val_table
        self.cache = cache or InputCache()
        self.grid = patch_grid(config.image_size, config.patch_size, config.patch_stride)

    def _filters_background(self, model_kind: str) -> bool:
        # The model without a mask is trained and validated on every patch, air included.
        return model_kind != "twostream_noroi"

    @torch.no_grad()
    def validate(
        self, model_kind: str, model: nn.Module, criterion: nn.Module
    ) -> tuple[float, float, float]:
        """Return validation loss, validation SROCC and the std of the predictions."""
        model.eval()
        threshold = (
            self.config.background_mean_threshold if self._filters_background(model_kind) else None
        )
        losses, predictions, targets = [], [], []
        for row in self.val_table.itertuples():
            stack = self.cache.get(model_kind, row.path)
            if model_kind == "global_resnet18":
                x = torch.from_numpy(stack)[None].to(self.device)
                score = forward_model(model_kind, model, x)[0]
            else:
                patches = grid_patches(stack, self.grid, self.config.patch_size, threshold)
                if patches is None:
                    continue
                score = forward_model(model_kind, model, torch.from_numpy(patches).to(self.device))
                score = score.mean()
            predictions.append(float(score))
            targets.append(row.mos)
            target = torch.tensor(row.mos, device=self.device)
            losses.append(float(criterion(score, target)))
        if model_kind == "global_resnet18":
            # The global model is validated in batches of 4 slices.
            pred_tensor = torch.tensor(predictions)
            target_tensor = torch.tensor(targets, dtype=torch.float32)
            losses = [
                float(criterion(pred_tensor[i : i + 4], target_tensor[i : i + 4]))
                for i in range(0, len(pred_tensor), 4)
            ]
        srocc = float(spearmanr(predictions, targets)[0]) if len(predictions) > 2 else float("nan")
        return float(np.mean(losses)), srocc, float(np.std(predictions))

    def _training_batch(
        self, model_kind: str, indices: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        paths = self.train_table["path"].to_numpy()[indices]
        if model_kind == "global_resnet18":
            return np.stack([self.cache.get(model_kind, path) for path in paths])
        attempts = self.config.crop_attempts if self._filters_background(model_kind) else 1
        return np.stack(
            [
                random_crop(
                    self.cache.get(model_kind, path),
                    rng,
                    self.config.patch_size,
                    attempts,
                    self.config.background_mean_threshold,
                )
                for path in paths
            ]
        )

    def train(
        self,
        model_kind: str,
        seed: int,
        save_path: Path,
        clip_norm: float = 0.0,
        learning_rate: float | None = None,
    ) -> TrainingResult:
        """Train and save the best-on-validation ``state_dict`` to ``save_path``."""
        config = self.config
        set_seed(seed)
        rng = np.random.default_rng(seed)
        model = build_model(model_kind, config.use_pretrained_backbone).to(self.device)
        criterion = nn.SmoothL1Loss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate or config.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=config.lr_factor, patience=config.lr_patience
        )
        batch_size = (
            config.global_batch_size if model_kind == "global_resnet18" else config.batch_size
        )
        mos = self.train_table["mos"].to_numpy()
        best_loss, best_srocc, log = float("inf"), float("nan"), []
        save_path.parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(config.num_epochs):
            model.train()
            order = rng.permutation(len(self.train_table))
            running_loss, steps = 0.0, 0
            batches = range(0, len(order), batch_size)
            for start in tqdm(
                batches, desc=f"{model_kind} seed {seed} epoch {epoch + 1}", leave=False
            ):
                indices = order[start : start + batch_size]
                x = torch.from_numpy(self._training_batch(model_kind, indices, rng)).to(self.device)
                y = torch.tensor(mos[indices], dtype=torch.float32, device=self.device)
                optimizer.zero_grad()
                loss = criterion(forward_model(model_kind, model, x), y)
                loss.backward()
                if clip_norm > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                optimizer.step()
                running_loss += float(loss.detach())
                steps += 1

            val_loss, val_srocc, val_std = self.validate(model_kind, model, criterion)
            scheduler.step(val_loss)
            saved = bool(np.isfinite(val_loss)) and val_loss < best_loss
            if saved:
                best_loss, best_srocc = val_loss, val_srocc
                torch.save(model.state_dict(), save_path)
            log.append(
                {
                    "model": model_kind,
                    "seed": seed,
                    "epoch": epoch + 1,
                    "lr": optimizer.param_groups[0]["lr"],
                    "train_loss": running_loss / steps,
                    "val_loss": val_loss,
                    "val_srocc": val_srocc,
                    "val_pred_std": val_std,
                    "saved": saved,
                }
            )
            print(
                f"[{model_kind} seed {seed}] epoch {epoch + 1:02d}/{config.num_epochs} "
                f"train {running_loss / steps:.4f} val {val_loss:.4f} "
                f"val SROCC {val_srocc:.4f}{' (saved)' if saved else ''}"
            )

        final_std = log[-1]["val_pred_std"] if log else 0.0
        collapsed = (
            not np.isfinite(best_srocc)
            or best_srocc < config.collapse_min_val_srocc
            or final_std < config.collapse_min_pred_std
        )
        return TrainingResult(best_loss, best_srocc, collapsed, log)
