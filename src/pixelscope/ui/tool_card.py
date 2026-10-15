"""Explicit, accessible ON/OFF switches and optional expandable tool settings."""
from __future__ import annotations

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QCheckBox, QFrame, QHBoxLayout, QLabel, QPushButton,
    QSlider, QVBoxLayout, QWidget,
)


class VisibleCheckBox(QCheckBox):
    """Draw an unmistakable square toggle on dark backgrounds.

    Native Qt checkbox indicators can become almost invisible with Fusion and
    dark stylesheets. We keep a real, keyboard-accessible QCheckBox while
    painting a high-contrast outline and checkmark explicitly.
    """

    def __init__(self, title: str, parent=None) -> None:
        super().__init__(title, parent)
        self.setObjectName("effectCheckbox")
        self.setMinimumHeight(32)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Click the square or tool name to enable/disable this effect")

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        checked = self.isChecked()
        hovering = self.underMouse()

        box_size = 21
        x = 3.0
        y = (self.height() - box_size) / 2.0
        rect = QRectF(x, y, box_size, box_size)

        border = QColor("#9dc2ec" if (hovering or checked) else "#8299b6")
        background = QColor("#b0d6ff" if checked else "#101722")
        painter.setPen(QPen(border, 2))
        painter.setBrush(background)
        painter.drawRoundedRect(rect, 4, 4)

        if checked:
            painter.setPen(QPen(
                QColor("#102037"), 2.8,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            ))
            painter.drawLine(int(x + 5), int(y + 11), int(x + 9), int(y + 15))
            painter.drawLine(int(x + 9), int(y + 15), int(x + 17), int(y + 6))

        painter.setFont(self.font())
        painter.setPen(QColor("#f3f6ff" if self.isEnabled() else "#778196"))
        painter.drawText(
            int(x + box_size + 11), 0,
            max(1, self.width() - int(x + box_size + 14)),
            self.height(),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            self.text(),
        )

        if self.hasFocus():
            painter.setPen(QPen(QColor("#82b5fa"), 1, Qt.PenStyle.DotLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(QRectF(0.5, 0.5, self.width() - 1, self.height() - 1), 4, 4)
        painter.end()


class ToolCard(QFrame):
    def __init__(
        self, title: str, enabled: bool, settings: list[tuple],
        on_toggle, on_parameter, on_expand, expanded: bool = False, parent=None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("toolCard")
        self._has_settings = bool(settings)
        self._expanded = expanded if self._has_settings else False
        self._on_expand = on_expand

        body = QVBoxLayout(self)
        body.setContentsMargins(12, 10, 12, 10)
        body.setSpacing(8)

        header = QHBoxLayout()
        header.setSpacing(7)
        self.enabled_box = VisibleCheckBox(title)
        self.enabled_box.setChecked(enabled)
        self.enabled_box.toggled.connect(on_toggle)
        header.addWidget(self.enabled_box, 1)

        # An expand arrow is only useful for tools that have actual settings.
        self.details_button: QPushButton | None = None
        if self._has_settings:
            self.details_button = QPushButton("▾" if expanded else "▸")
            self.details_button.setFixedSize(30, 28)
            self.details_button.setToolTip("Show or hide tool settings")
            self.details_button.setCursor(Qt.CursorShape.PointingHandCursor)
            self.details_button.clicked.connect(self._toggle_details)
            header.addWidget(self.details_button)
        body.addLayout(header)

        self.details = QWidget()
        settings_layout = QVBoxLayout(self.details)
        settings_layout.setContentsMargins(1, 0, 1, 0)
        settings_layout.setSpacing(5)

        self.sliders: dict[str, QSlider] = {}
        for label, low, high, value, key in settings:
            caption = QLabel(f"{label}: {value}")
            caption.setObjectName("muted")
            settings_layout.addWidget(caption)
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(low, high)
            slider.setValue(value)

            def changed(next_value: int, key=key, label=label, caption=caption) -> None:
                caption.setText(f"{label}: {next_value}")
                on_parameter(key, next_value)

            slider.valueChanged.connect(changed)
            settings_layout.addWidget(slider)
            self.sliders[key] = slider

        if self._has_settings:
            body.addWidget(self.details)
            self.details.setVisible(self._expanded)
        else:
            self.details.hide()

    def _toggle_details(self) -> None:
        if not self._has_settings or self.details_button is None:
            return
        self._expanded = not self._expanded
        self.details.setVisible(self._expanded)
        self.details_button.setText("▾" if self._expanded else "▸")
        self._on_expand(self._expanded)
