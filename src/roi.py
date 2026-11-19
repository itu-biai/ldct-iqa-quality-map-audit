"""Smart ROI: a body mask that separates tissue from the surrounding air."""

import cv2
import numpy as np


def smart_roi_mask(image: np.ndarray) -> np.ndarray:
    """Return a boolean body mask for a slice normalized to [0, 1].

    Steps: Otsu threshold, two passes of a 7 x 7 morphological opening, the largest
    connected component, and filling of its outer contour (which fills inner holes).
    When no foreground component is found, the whole image is kept.
    """
    image_8bit = (image * 255).astype(np.uint8)
    _, binary = cv2.threshold(image_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8), iterations=2)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return np.ones_like(image, dtype=bool)
    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component = (labels == largest_label).astype(np.uint8)
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.ones_like(image, dtype=bool)
    outer = max(contours, key=cv2.contourArea)
    filled = np.zeros_like(component, dtype=np.float32)
    cv2.drawContours(filled, [outer], -1, 1, thickness=cv2.FILLED)
    return filled.astype(bool)


def apply_roi(image: np.ndarray) -> np.ndarray:
    """Zero every pixel outside the smart ROI of ``image``."""
    return (image * smart_roi_mask(image)).astype(np.float32)
