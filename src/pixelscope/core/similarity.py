"""Image similarity search using difference hashes (dHash)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .processing import load_rgb


@dataclass(frozen=True, slots=True)
class SimilarPair:
    first: Path
    second: Path
    distance: int


def difference_hash(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    return (resized[:, 1:] > resized[:, :-1]).flatten()


def most_similar(paths: list[Path]) -> SimilarPair:
    if not (3 <= len(paths) <= 10):
        raise ValueError("Choose between 3 and 10 images.")

    hashes = [difference_hash(load_rgb(str(p))) for p in paths]
    best: SimilarPair | None = None

    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            distance = int(np.count_nonzero(hashes[i] != hashes[j]))
            if best is None or distance < best.distance:
                best = SimilarPair(paths[i], paths[j], distance)

    assert best is not None
    return best
