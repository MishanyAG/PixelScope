"""Foreground segmentation and pronounced blur of moving video objects.

A single stateful background model is shared across display modes so switching
between foreground extraction and blur does not discard the learned background.
"""

from __future__ import annotations

import cv2
import numpy as np


class VideoProcessor:
    def __init__(self, blur_size: int = 101) -> None:
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=240, varThreshold=20, detectShadows=False
        )
        self.blur_size = blur_size

    def foreground_mask(self, rgb: np.ndarray) -> np.ndarray:
        """Return a cleaned 0/255 moving-foreground mask."""
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        mask = self._subtractor.apply(bgr)
        # Remove isolated pixels without erasing thin moving boundaries.
        mask = cv2.medianBlur(mask, 5)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8)
        )
        return mask

    def process(self, rgb: np.ndarray, operation: str) -> np.ndarray:
        mask = self.foreground_mask(rgb)

        if operation == "mask":
            return cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        if operation == "background":
            # Background subtraction should reveal the original-color subject
            # against black rather than only the binary diagnostic mask.
            foreground = np.zeros_like(rgb)
            foreground[mask > 0] = rgb[mask > 0]
            return foreground

        if operation == "moving_blur":
            k = max(15, min(121, int(self.blur_size) | 1))
            blurred = cv2.GaussianBlur(rgb, (k, k), sigmaX=0)
            # Extend the moving region slightly and soften the compositing seam.
            expanded = cv2.dilate(
                mask, np.ones((11, 11), np.uint8), iterations=1
            )
            alpha = cv2.GaussianBlur(expanded, (9, 9), sigmaX=0)
            weight = (alpha.astype(np.float32) / 255.0)[:, :, None]
            combined = (
                rgb.astype(np.float32) * (1.0 - weight)
                + blurred.astype(np.float32) * weight
            )
            return np.clip(combined, 0, 255).astype(np.uint8)

        raise ValueError(f"Unknown video operation: {operation}")
