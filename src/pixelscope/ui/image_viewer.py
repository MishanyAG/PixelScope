"""Image viewer with aspect-ratio preservation and pixel selection."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QMouseEvent, QPixmap, QResizeEvent
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout


class _Canvas(QLabel):
    pixel_clicked = Signal(int, int)

    def __init__(self) -> None:
        super().__init__()
        self.image_size: tuple[int, int] | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        pixmap = self.pixmap()
        if (
            event.button() == Qt.MouseButton.LeftButton
            and pixmap is not None
            and not pixmap.isNull()
            and self.image_size is not None
        ):
            px = event.position().x()
            py = event.position().y()
            left = (self.width() - pixmap.width()) / 2
            top = (self.height() - pixmap.height()) / 2
            if left <= px < left + pixmap.width() and top <= py < top + pixmap.height():
                width, height = self.image_size
                x = min(width - 1, int((px - left) * width / pixmap.width()))
                y = min(height - 1, int((py - top) * height / pixmap.height()))
                self.pixel_clicked.emit(x, y)
        super().mousePressEvent(event)


class ImageViewer(QFrame):
    pixel_clicked = Signal(int, int)

    def __init__(self, title: str = "Image", parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("imageViewer")
        self.setMinimumSize(300, 250)

        self._pixmap = QPixmap()
        self._caption = QLabel(title.upper())
        self._caption.setObjectName("muted")

        self._canvas = _Canvas()
        self._canvas.pixel_clicked.connect(self.pixel_clicked.emit)
        self._canvas.setText("Drop an image here\nor click “Open image”")
        self._canvas.setObjectName("viewerHint")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)
        layout.addWidget(self._caption)
        layout.addWidget(self._canvas, 1)

    def set_array(self, rgb: np.ndarray) -> None:
        frame = np.ascontiguousarray(rgb, dtype=np.uint8)
        if frame.ndim == 2:
            height, width = frame.shape
            image = QImage(
                frame.data, width, height, frame.strides[0],
                QImage.Format.Format_Grayscale8,
            ).copy()
        elif frame.ndim == 3 and frame.shape[2] == 3:
            height, width, _ = frame.shape
            image = QImage(
                frame.data, width, height, frame.strides[0],
                QImage.Format.Format_RGB888,
            ).copy()
        else:
            raise ValueError("Expected a grayscale or RGB uint8 image.")

        self._pixmap = QPixmap.fromImage(image)
        self._canvas.image_size = (width, height)
        self._canvas.setText("")
        self._refresh()

    def clear_image(self) -> None:
        self._pixmap = QPixmap()
        self._canvas.clear()
        self._canvas.image_size = None
        self._canvas.setText("Drop an image here\nor click “Open image”")

    def set_caption(self, title: str) -> None:
        self._caption.setText(title.upper())

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._refresh()

    def _refresh(self) -> None:
        if self._pixmap.isNull():
            return
        width = max(1, self._canvas.width() - 6)
        height = max(1, self._canvas.height() - 6)
        scaled = self._pixmap.scaled(
            width, height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._canvas.setPixmap(scaled)
