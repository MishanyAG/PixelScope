"""Frame-based motion segmentation and moving-object blur."""

from __future__ import annotations

import cv2
import numpy as np


class VideoProcessor:
    def __init__(self) -> None:
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=24, detectShadows=False
        )

    def process(self, rgb: np.ndarray, operation: str) -> np.ndarray:
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        mask = self._subtractor.apply(frame)
        mask = cv2.medianBlur(mask, 5)

        if operation == "background":
            return cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        if operation == "moving_blur":
            blurred = cv2.GaussianBlur(rgb, (31, 31), 0)
            result = rgb.copy()
            result[mask > 0] = blurred[mask > 0]
            return result

        raise ValueError(f"Unknown video operation: {operation}")
