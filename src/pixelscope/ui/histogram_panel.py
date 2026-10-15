"""Embedded, side-by-side histograms for source and edited image."""
from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class HistogramPanel(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._original: np.ndarray | None = None
        self._processed: np.ndarray | None = None
        self.setObjectName("histogramPanel")
        self.setMinimumHeight(220)
        self.setMaximumHeight(300)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(5)
        header = QHBoxLayout()
        title = QLabel("IMAGE HISTOGRAMS")
        title.setObjectName("metaValue")
        header.addWidget(title)
        header.addStretch()
        self.channel = QComboBox()
        self.channel.addItems(["RGB", "Red", "Green", "Blue", "Luminance"])
        self.channel.currentIndexChanged.connect(self._draw)
        self.channel.setMinimumWidth(135)
        header.addWidget(self.channel)
        layout.addLayout(header)

        self.figure = Figure(figsize=(10, 3), dpi=90, facecolor="#12141a")
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas, 1)
        self.axes = self.figure.subplots(1, 2)
        self._draw()

    def set_images(self, original: np.ndarray, processed: np.ndarray) -> None:
        self._original = original
        self._processed = processed
        self._draw()

    def _draw(self, *_args) -> None:
        palette = ("#ff697c", "#68d996", "#74a4ff")
        channel = self.channel.currentText()
        for ax, title, image in zip(
            self.axes, ("Original", "Processed"), (self._original, self._processed)
        ):
            ax.clear()
            ax.set_facecolor("#161a23")
            ax.set_title(title, color="#e7eaf0", fontsize=10, pad=8)
            ax.set_xlim(0, 255)
            ax.tick_params(axis="both", colors="#939bab", labelsize=8)
            ax.grid(alpha=0.12)
            for spine in ax.spines.values():
                spine.set_color("#343a46")
            if image is None:
                continue
            if channel == "Luminance":
                gray = (
                    0.299 * image[:, :, 0] +
                    0.587 * image[:, :, 1] +
                    0.114 * image[:, :, 2]
                ).astype(np.uint8)
                y = np.bincount(gray.ravel(), minlength=256)
                ax.plot(np.arange(256), y, color="#f0f0f3", linewidth=1)
            else:
                indices = range(3) if channel == "RGB" else [
                    {"Red": 0, "Green": 1, "Blue": 2}[channel]
                ]
                for i in indices:
                    y = np.bincount(image[:, :, i].ravel(), minlength=256)
                    ax.plot(np.arange(256), y, color=palette[i], linewidth=1)
        self.figure.subplots_adjust(
            left=0.055, right=0.985, bottom=0.17, top=0.83, wspace=0.13
        )
        self.canvas.draw_idle()
