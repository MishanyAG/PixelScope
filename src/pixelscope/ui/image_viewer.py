from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QResizeEvent
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout


class ImageViewer(QFrame):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("imageViewer")
        self.setMinimumSize(480, 320)

        self._source_pixmap = QPixmap()

        self._label = QLabel("Drop an image here\nor click “Open image”")
        self._label.setObjectName("viewerHint")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.addWidget(self._label)

    def set_image(self, path: str | Path) -> None:
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            raise ValueError("Qt could not decode this image.")

        self._source_pixmap = pixmap
        self._label.setObjectName("")
        self._label.setText("")
        self._refresh_pixmap()

    def clear_image(self) -> None:
        self._source_pixmap = QPixmap()
        self._label.clear()
        self._label.setObjectName("viewerHint")
        self._label.setText("Drop an image here\nor click “Open image”")
        self._label.style().unpolish(self._label)
        self._label.style().polish(self._label)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._refresh_pixmap()

    def _refresh_pixmap(self) -> None:
        if self._source_pixmap.isNull():
            return

        viewport = self._label.size()
        target_width = max(1, viewport.width() - 8)
        target_height = max(1, viewport.height() - 8)

        scaled = self._source_pixmap.scaled(
            target_width,
            target_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._label.setPixmap(scaled)
