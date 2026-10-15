"""Compact, expandable tool controls with independent enable switches."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QFrame, QHBoxLayout, QLabel, QPushButton,
    QSlider, QVBoxLayout, QWidget,
)


class ToolCard(QFrame):
    def __init__(
        self, title: str, enabled: bool, settings: list[tuple],
        on_toggle, on_parameter, on_expand, expanded: bool = False, parent=None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("toolCard")
        self._expanded = expanded
        self._on_expand = on_expand

        body = QVBoxLayout(self)
        body.setContentsMargins(12, 10, 12, 10)
        body.setSpacing(8)

        header = QHBoxLayout()
        header.setSpacing(7)
        self.enabled_box = QCheckBox(title)
        self.enabled_box.setChecked(enabled)
        self.enabled_box.setToolTip("Enable or disable this effect")
        self.enabled_box.toggled.connect(on_toggle)
        header.addWidget(self.enabled_box, 1)

        self.details_button = QPushButton("▾" if expanded else "▸")
        self.details_button.setFixedSize(30, 28)
        self.details_button.setToolTip("Show or hide tool settings")
        self.details_button.clicked.connect(self._toggle_details)
        header.addWidget(self.details_button)
        body.addLayout(header)

        self.details = QWidget()
        settings_layout = QVBoxLayout(self.details)
        settings_layout.setContentsMargins(1, 0, 1, 0)
        settings_layout.setSpacing(5)
        if not settings:
            hint = QLabel("No additional settings")
            hint.setObjectName("muted")
            settings_layout.addWidget(hint)

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

        body.addWidget(self.details)
        self.details.setVisible(expanded)

    def _toggle_details(self) -> None:
        self._expanded = not self._expanded
        self.details.setVisible(self._expanded)
        self.details_button.setText("▾" if self._expanded else "▸")
        self._on_expand(self._expanded)
