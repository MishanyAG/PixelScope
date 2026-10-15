"""Dialogs for histogram comparison and image pair inspection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QComboBox, QDialog, QHBoxLayout, QLabel, QVBoxLayout,
)

from ..core.processing import load_rgb
from ..core.similarity import SimilarPair
from .image_viewer import ImageViewer


class HistogramDialog(QDialog):
    def __init__(self, source: np.ndarray, result: np.ndarray, parent=None) -> None:
        super().__init__(parent)
        self._source = source.copy()
        self._result = result.copy()
        self.setWindowTitle("PixelScope · Histogram comparison")
        self.resize(950, 600)

        layout = QVBoxLayout(self)
        info = QLabel("Pixel intensity distribution (0–255) · Original vs processed")
        info.setObjectName("muted")
        layout.addWidget(info)

        self._channel = QComboBox()
        self._channel.addItems(["RGB", "Red", "Green", "Blue", "Luminance"])
        self._channel.currentIndexChanged.connect(self._draw)
        layout.addWidget(self._channel)

        self._figure = Figure(figsize=(10, 5))
        self._figure.patch.set_facecolor("#12141a")
        self._canvas = FigureCanvasQTAgg(self._figure)
        layout.addWidget(self._canvas, 1)
        self._draw()

    def _draw(self) -> None:
        self._figure.clear()
        axes = self._figure.subplots(1, 2)
        colors = ("#fa6a78", "#6cdb91", "#6f9ffc")
        selection = self._channel.currentText()
        for axis, rgb, title in zip(
            axes, (self._source, self._result), ("Original", "Processed")
        ):
            if selection == "Luminance":
                values = (
                    0.299 * rgb[:, :, 0]
                    + 0.587 * rgb[:, :, 1]
                    + 0.114 * rgb[:, :, 2]
                )
                histogram, _ = np.histogram(values, bins=256, range=(0, 256))
                axis.plot(histogram, color="#e8ebf2", lw=1)
            else:
                selected = range(3) if selection == "RGB" else (("Red", "Green", "Blue").index(selection),)
                for channel in selected:
                    histogram = np.bincount(
                        rgb[:, :, channel].ravel(), minlength=256
                    )
                    axis.plot(histogram, color=colors[channel], lw=1)
            axis.set_title(title, color="#e8ebf2")
            axis.set_xlim(0, 255)
            axis.tick_params(colors="#a7afbc")
            axis.set_facecolor("#1b1e27")
            for spine in axis.spines.values():
                spine.set_color("#353a46")
            axis.grid(alpha=0.12)

        self._figure.tight_layout()
        self._canvas.draw_idle()


class SimilarityDialog(QDialog):
    def __init__(self, pair: SimilarPair, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("PixelScope · Most similar images")
        self.resize(1060, 620)

        layout = QVBoxLayout(self)
        intro = QLabel(
            f"Closest pair · Difference-hash distance: {pair.distance} / 64 (lower is more similar)"
        )
        intro.setObjectName("muted")
        layout.addWidget(intro)

        row = QHBoxLayout()
        for path in (pair.first, pair.second):
            column = QVBoxLayout()
            viewer = ImageViewer("Match")
            viewer.set_array(load_rgb(str(path)))
            label = QLabel(Path(path).name)
            label.setObjectName("metaValue")
            column.addWidget(viewer, 1)
            column.addWidget(label)
            row.addLayout(column, 1)
        layout.addLayout(row, 1)
