"""Model inputs: the anatomy channel and the Haar wavelet HH high-frequency channel."""

import cv2
import numpy as np
import pywt

from roi import apply_roi

MODEL_KINDS: tuple[str, ...] = (
    "twostream_roi",
    "twostream_noroi",
    "patch_resnet18",
    "global_resnet18",
)


def hh_band(image: np.ndarray) -> np.ndarray:
    """Absolute HH wavelet band, upsampled to image size and clipped at mean + 3 std.

    The result is scaled to [0, 1].
    """
    _, (_, _, hh) = pywt.dwt2(image, "haar")
    hh = cv2.resize(hh, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    hh = np.abs(hh)
    clip_max = float(np.mean(hh) + 3 * np.std(hh))
    hh = np.clip(hh, 0, clip_max)
    return (hh / clip_max if clip_max > 0 else hh).astype(np.float32)


def two_stream_stack(anatomy: np.ndarray) -> np.ndarray:
    """Stack the anatomy channel with its HH channel: shape [2, H, W]."""
    anatomy = anatomy.astype(np.float32)
    return np.stack([anatomy, hh_band(anatomy)])


def prepare_input(model_kind: str, image: np.ndarray) -> np.ndarray:
    """Channel stack a model sees for a full slice (training and validation).

    - ``twostream_roi``: [image * ROI, HH(image * ROI)]
    - ``twostream_noroi``: [image, HH(image)]
    - ``patch_resnet18``: [image * ROI]
    - ``global_resnet18``: [image]
    """
    image = image.astype(np.float32)
    if model_kind == "twostream_roi":
        return two_stream_stack(apply_roi(image))
    if model_kind == "twostream_noroi":
        return two_stream_stack(image)
    if model_kind == "patch_resnet18":
        return apply_roi(image)[None]
    if model_kind == "global_resnet18":
        return image[None]
    raise ValueError(f"unknown model kind: {model_kind}")


def prepare_input_with_roi(model_kind: str, image: np.ndarray, roi: np.ndarray) -> np.ndarray:
    """Same as :func:`prepare_input` but with a precomputed ROI of the clean slice.

    The audit computes the ROI once on the clean slice and reuses it for every
    perturbed copy, so the mask does not change with the injected artifact.
    """
    image = image.astype(np.float32)
    if model_kind == "twostream_roi":
        return two_stream_stack(image * roi)
    if model_kind == "patch_resnet18":
        return (image * roi).astype(np.float32)[None]
    return prepare_input(model_kind, image)
