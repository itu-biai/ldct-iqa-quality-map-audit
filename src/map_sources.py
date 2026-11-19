"""Map sources: anything that assigns one score per grid patch of a slice.

Higher scores mean better quality for the signed sources. The Grad-CAM source returns
an attribution, which has no quality direction, so it is reported for Loc only.
"""

from typing import Protocol

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch import nn

from models import forward_model, is_two_stream
from patches import PatchGrid
from preprocessing import prepare_input_with_roi


class MapSource(Protocol):
    """Interface used by the audit."""

    name: str
    signed: bool

    def patch_scores(self, image: np.ndarray, roi: np.ndarray) -> np.ndarray:
        """Return one score per grid patch."""
        ...


class ModelPatchScorer:
    """Patch scores (and the global score) of a trained network.

    ``variant`` applies only to two-stream models: ``fused`` uses both streams,
    ``anatomy`` zeroes the HH input and ``hf`` zeroes the anatomy input.
    """

    signed = True

    def __init__(
        self,
        name: str,
        model_kind: str,
        model: nn.Module,
        grid: PatchGrid,
        patch_size: int,
        device: torch.device,
        variant: str = "fused",
        background_mean_threshold: float = 0.05,
    ) -> None:
        self.name, self.model_kind, self.variant = name, model_kind, variant
        self.model = model.eval().to(device)
        self.grid, self.patch_size, self.device = grid, patch_size, device
        self.background_mean_threshold = background_mean_threshold

    def _patches(self, image: np.ndarray, roi: np.ndarray) -> torch.Tensor:
        stack = torch.from_numpy(prepare_input_with_roi(self.model_kind, image, roi))
        size = self.patch_size
        patches = [stack[..., top : top + size, left : left + size] for top, left in self.grid]
        return torch.stack(patches).to(self.device)

    @torch.no_grad()
    def patch_scores(self, image: np.ndarray, roi: np.ndarray) -> np.ndarray:
        """Patch scores clamped to the MOS range [1, 5] (global ResNet: unclamped)."""
        patches = self._patches(image, roi)
        if is_two_stream(self.model_kind):
            if self.variant == "anatomy":
                patches[:, 1] = 0
            elif self.variant == "hf":
                patches[:, 0] = 0
        scores = forward_model(self.model_kind, self.model, patches)
        if self.model_kind != "global_resnet18":
            scores = torch.clamp(scores, 1.0, 5.0)
        return scores.float().cpu().numpy()

    @torch.no_grad()
    def global_score(self, image: np.ndarray, roi: np.ndarray) -> float:
        """Image-level MOS prediction, as used for global SROCC.

        - global ResNet18: the clamped whole-image output;
        - patch ResNet18: mean of clamped scores of non-background patches;
        - two-stream with ROI: HH-std weighted mean over non-background patches;
        - two-stream without ROI: HH-std weighted mean over all patches.
        """
        if self.model_kind == "global_resnet18":
            x = torch.from_numpy(image.astype(np.float32))[None, None].to(self.device)
            return float(torch.clamp(forward_model(self.model_kind, self.model, x)[0], 1.0, 5.0))
        patches = self._patches(image, roi)
        scores = torch.clamp(forward_model(self.model_kind, self.model, patches), 1.0, 5.0)
        scores_np = scores.cpu().numpy()
        valid = (patches[:, 0].mean(dim=(1, 2)) > self.background_mean_threshold).cpu().numpy()
        if self.model_kind == "twostream_noroi":
            valid = np.ones_like(valid)
        if not valid.any():
            return 0.0
        if is_two_stream(self.model_kind):
            weights = (patches[:, 1].flatten(1).std(dim=1) + 1e-6).cpu().numpy()
            return float(np.average(scores_np[valid], weights=weights[valid]))
        return float(scores_np[valid].mean())


class LocalNoiseMap:
    """Training-free reference: minus the std of the Gaussian high-pass residual per patch."""

    name, signed = "local_noise_reference", True

    def __init__(self, grid: PatchGrid, patch_size: int, sigma: float = 1.5) -> None:
        self.grid, self.patch_size, self.sigma = grid, patch_size, sigma

    def patch_scores(self, image: np.ndarray, roi: np.ndarray) -> np.ndarray:
        """Patch scores from ROI pixels only; patches with 16 or fewer ROI pixels get 0."""
        residual = image - gaussian_filter(image, sigma=self.sigma)
        size = self.patch_size
        scores = []
        for top, left in self.grid:
            inside = roi[top : top + size, left : left + size] > 0
            values = residual[top : top + size, left : left + size][inside]
            scores.append(-values.std() if values.size > 16 else 0.0)
        return np.array(scores, dtype=np.float32)


class GradCamMap:
    """Grad-CAM of the global ResNet18 (``layer4``), averaged inside each patch."""

    name, signed = "global_resnet18_gradcam", False

    def __init__(
        self, model: nn.Module, grid: PatchGrid, patch_size: int, device: torch.device
    ) -> None:
        self.model = model.eval().to(device)
        self.grid, self.patch_size, self.device = grid, patch_size, device
        self.activations: torch.Tensor | None = None
        self.model.net.layer4.register_forward_hook(self._save_activations)

    def _save_activations(self, _module: nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
        self.activations = output

    def patch_scores(self, image: np.ndarray, roi: np.ndarray) -> np.ndarray:
        """Mean ReLU Grad-CAM attribution inside each patch (bilinear upsampling)."""
        with torch.enable_grad():
            x = torch.from_numpy(image.astype(np.float32))[None, None].to(self.device)
            score = self.model(x).sum()
            gradients = torch.autograd.grad(score, self.activations)[0]
            weights = gradients.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * self.activations).sum(1, keepdim=True))
            cam = torch.nn.functional.interpolate(cam, size=image.shape, mode="bilinear")
            cam_np = cam[0, 0].detach().cpu().numpy()
        size = self.patch_size
        return np.array(
            [cam_np[top : top + size, left : left + size].mean() for top, left in self.grid]
        )
