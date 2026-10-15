"""Main PixelScope desktop workspace.

Edits accumulate through a replayable history; section reset affects only
operations belonging to the current editor tab.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QDragEnterEvent, QDropEvent, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..core.edit_pipeline import EditPipeline
from ..core.image_document import ImageDocument
from ..core.processing import load_rgb, pixel_colors, run_tool
from ..core.similarity import most_similar
from ..core.video import VideoProcessor
from .histogram_panel import HistogramPanel
from .image_viewer import ImageViewer
from .tool_card import ToolCard


_IMAGE_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
_SECTIONS = (
    "Image", "Color", "Adjustments", "Histogram",
    "Morphology", "Filters", "Features", "Compare", "Video",
)
_MORPHOLOGY = (
    ("Erosion", "erode"),
    ("Dilation", "dilate"),
    ("Opening", "open"),
    ("Closing", "close"),
    ("Morphological gradient", "gradient"),
)
_FILTERS = (
    ("Invert colors", "invert"),
    ("Sharpen", "sharpen"),
    ("Motion blur", "motion"),
    ("Emboss", "emboss"),
    ("Median filter", "median"),
    ("Canny edges", "canny"),
    ("Roberts edges", "roberts"),
)
_FEATURES = (
    ("Harris corners", "harris"),
    ("SIFT keypoints", "sift"),
    ("FAST keypoints", "fast"),
)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self._document: ImageDocument | None = None
        self._source: np.ndarray | None = None
        self._result: np.ndarray | None = None
        self._selected_pixel: tuple[int, int] | None = None
        self._selected_origin = "Original"
        self._section = "Image"
        self._operation = "original"
        self._pipeline: EditPipeline | None = None
        self._navigation: dict[str, QPushButton] = {}

        self._brightness = 0
        self._contrast = 100
        self._saturation = 100
        self._gamma = 1.0
        self._morph_k = 5
        self._filter_k = 5
        self._low = 70
        self._high = 150
        self._sensitivity = 20
        self._morph_operation = ""
        self._filter_operation = ""
        self._feature_operation = ""
        # Individual tool controls do not apply an effect until enabled.
        self._tool_settings: dict[str, dict[str, int]] = {}
        self._expanded_tools: set[str] = set()
        self._tool_cards: dict[str, ToolCard] = {}

        self._video_path: str | None = None
        self._capture: cv2.VideoCapture | None = None
        self._video_processor: VideoProcessor | None = None
        self._video_operation = "background"
        self._video_blur = 101
        self._video_timer = QTimer(self)
        self._video_timer.timeout.connect(self._advance_video)

        self.setWindowTitle("PixelScope")
        self.resize(1530, 910)
        self.setMinimumSize(1120, 670)
        self.setAcceptDrops(True)

        self._create_actions()
        self._build_ui()
        self._render_controls()
        self._update_navigation()
        self.statusBar().showMessage("Ready · Open an image to get started")

    def _create_actions(self) -> None:
        # Native keyboard shortcuts remain available, but the redundant File
        # and View menus are deliberately absent from the visible interface.
        self._open_action = QAction("Open image", self)
        self._open_action.setShortcut(QKeySequence.StandardKey.Open)
        self._open_action.triggered.connect(self.open_image)
        self.addAction(self._open_action)

        self._save_action = QAction("Save image", self)
        self._save_action.setShortcut(QKeySequence.StandardKey.Save)
        self._save_action.triggered.connect(self.save_result)
        self._save_action.setEnabled(False)
        self.addAction(self._save_action)

        self._histogram_action = QAction("View histograms", self)
        self._histogram_action.triggered.connect(self._show_histograms)
        self._histogram_action.setEnabled(False)
        self.addAction(self._histogram_action)

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QHBoxLayout(root)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._build_sidebar())
        layout.addWidget(self._build_workspace(), 1)
        self.setCentralWidget(root)

    def _build_sidebar(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("sidebar")
        frame.setFixedWidth(204)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(15, 22, 15, 18)
        layout.setSpacing(5)

        brand = QLabel("PIXELSCOPE")
        brand.setObjectName("brand")
        layout.addWidget(brand)

        subtitle = QLabel("Image processing lab")
        subtitle.setObjectName("brandSubtitle")
        layout.addWidget(subtitle)
        layout.addSpacing(22)

        for section in _SECTIONS:
            button = QPushButton(section)
            button.setProperty("nav", True)
            button.setProperty("active", section == self._section)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.clicked.connect(lambda checked=False, name=section: self._switch_section(name))
            layout.addWidget(button)
            self._navigation[section] = button

        layout.addStretch()
        version = QLabel("v0.5 · independent tools")
        version.setObjectName("muted")
        layout.addWidget(version)
        return frame

    def _build_workspace(self) -> QWidget:
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._build_topbar())

        work = QWidget()
        row = QHBoxLayout(work)
        row.setContentsMargins(18, 20, 0, 20)
        row.setSpacing(12)

        # Each mode has an independent center workspace. The editor's current
        # image never becomes a placeholder or video frame in the other modes.
        self._mode_stack = QStackedWidget()
        self._mode_stack.addWidget(self._build_editor_workspace())
        self._mode_stack.addWidget(self._build_compare_workspace())
        self._mode_stack.addWidget(self._build_video_workspace())
        row.addWidget(self._mode_stack, 1)
        row.addWidget(self._build_inspector())

        layout.addWidget(work, 1)
        return root

    def _build_editor_workspace(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        pictures = QHBoxLayout()
        pictures.setSpacing(12)
        self._source_view = ImageViewer("Original image")
        self._source_view.pixel_clicked.connect(
            lambda x, y: self._on_pixel_clicked(x, y, processed=False)
        )
        pictures.addWidget(self._source_view, 1)

        self._result_view = ImageViewer("Processed")
        self._result_view.pixel_clicked.connect(
            lambda x, y: self._on_pixel_clicked(x, y, processed=True)
        )
        pictures.addWidget(self._result_view, 1)
        layout.addLayout(pictures, 1)

        self._histogram_panel = HistogramPanel()
        self._histogram_panel.setVisible(False)
        layout.addWidget(self._histogram_panel, 0)
        return page

    def _build_compare_workspace(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self._compare_info = QLabel(
            "Choose 3–10 images to find the closest visual pair. "
            "This workspace is independent of the image editor."
        )
        self._compare_info.setObjectName("muted")
        self._compare_info.setWordWrap(True)
        layout.addWidget(self._compare_info)

        images = QHBoxLayout()
        images.setSpacing(12)
        self._compare_left = ImageViewer("Closest match · A")
        self._compare_left.set_placeholder("No images compared yet")
        self._compare_right = ImageViewer("Closest match · B")
        self._compare_right.set_placeholder("No images compared yet")
        images.addWidget(self._compare_left, 1)
        images.addWidget(self._compare_right, 1)
        layout.addLayout(images, 1)
        return page

    def _build_video_workspace(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        self._video_info = QLabel(
            "Background subtraction learns the static scene over several frames. "
            "For clearer results, use a stationary camera."
        )
        self._video_info.setObjectName("muted")
        self._video_info.setWordWrap(True)
        layout.addWidget(self._video_info)

        images = QHBoxLayout()
        images.setSpacing(12)
        self._video_source_view = ImageViewer("Source video")
        self._video_source_view.set_placeholder("Load a video to begin")
        self._video_result_view = ImageViewer("Motion processing")
        self._video_result_view.set_placeholder("Processed video will appear here")
        images.addWidget(self._video_source_view, 1)
        images.addWidget(self._video_result_view, 1)
        layout.addLayout(images, 1)
        return page

    def _build_topbar(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("topbar")
        frame.setFixedHeight(76)

        layout = QHBoxLayout(frame)
        layout.setContentsMargins(23, 12, 23, 12)
        self._workspace_title = QLabel("Image workspace")
        self._workspace_title.setObjectName("sectionTitle")
        layout.addWidget(self._workspace_title)
        layout.addStretch()

        self._save_button = QPushButton("Save image")
        self._save_button.setEnabled(False)
        self._save_button.clicked.connect(self.save_result)
        layout.addWidget(self._save_button)

        self._open_button = QPushButton("Open image")
        self._open_button.setObjectName("primaryButton")
        self._open_button.clicked.connect(self.open_image)
        layout.addWidget(self._open_button)
        return frame

    def _build_inspector(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setObjectName("inspector")
        scroll.setFixedWidth(300)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        panel = QWidget()
        self._controls = QVBoxLayout(panel)
        self._controls.setContentsMargins(20, 12, 20, 20)
        self._controls.setSpacing(11)
        scroll.setWidget(panel)
        return scroll

    def _switch_section(self, section: str) -> None:
        if section == self._section:
            return
        if section not in ("Image", "Compare", "Video") and self._source is None:
            return
        if self._section == "Video" and section != "Video":
            self._stop_video()

        if self._pipeline is not None:
            self._pipeline.leave_tool()

        self._section = section
        self._histogram_panel.setVisible(section == "Histogram")
        self._source_view.set_pixel_sampling_enabled(section == "Color")
        self._result_view.set_pixel_sampling_enabled(section == "Color")
        if section == "Histogram" and self._source is not None and self._result is not None:
            self._histogram_panel.set_images(self._source, self._result)
        self._update_navigation()
        self._workspace_title.setText(f"{section} workspace")
        self._render_controls()

    def _update_navigation(self) -> None:
        standalone = self._section in ("Compare", "Video")
        has_image = self._source is not None
        for name, button in self._navigation.items():
            button.setEnabled(name in ("Image", "Compare", "Video") or has_image)
            button.setProperty("active", name == self._section)
            button.style().unpolish(button)
            button.style().polish(button)

        self._mode_stack.setCurrentIndex(
            {"Compare": 1, "Video": 2}.get(self._section, 0)
        )
        self._open_button.setVisible(not standalone)
        self._save_button.setVisible(not standalone)
        self._save_button.setEnabled(has_image)
        self._open_action.setEnabled(not standalone)
        self._save_action.setEnabled(has_image and not standalone)
        self._histogram_action.setEnabled(has_image and not standalone)

    def _clear_controls(self) -> None:
        self._tool_cards = {}
        while self._controls.count():
            item = self._controls.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _text(self, value: str, object_name: str = "muted") -> QLabel:
        label = QLabel(value)
        label.setObjectName(object_name)
        label.setWordWrap(True)
        self._controls.addWidget(label)
        return label

    def _button(self, title: str, callback, primary: bool = False) -> QPushButton:
        button = QPushButton(title)
        if primary:
            button.setObjectName("primaryButton")
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.clicked.connect(callback)
        self._controls.addWidget(button)
        return button

    def _selector(self, options: tuple[tuple[str, str], ...], callback, selected=None) -> QComboBox:
        combo = QComboBox()
        for label, value in options:
            combo.addItem(label, value)
        if selected is not None:
            index = combo.findData(selected)
            if index >= 0:
                combo.setCurrentIndex(index)
        combo.currentIndexChanged.connect(lambda _: callback(combo.currentData()))
        self._controls.addWidget(combo)
        return combo

    def _slider(self, title: str, minimum: int, maximum: int, value: int, callback) -> QSlider:
        label = self._text(f"{title}: {value}", "metaValue")
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(minimum, maximum)
        slider.setValue(value)

        def on_change(next_value: int) -> None:
            label.setText(f"{title}: {next_value}")
            callback(next_value)

        slider.valueChanged.connect(on_change)
        self._controls.addWidget(slider)
        return slider

    def _render_controls(self) -> None:
        self._clear_controls()
        self._text(self._section, "sectionTitle")
        if self._section not in ("Compare", "Video"):
            self._text("Tools build on the previous result. The source file stays unchanged.")

        if self._section == "Image":
            self._panel_image()
        elif self._section == "Color":
            self._panel_color()
        elif self._section == "Adjustments":
            self._panel_adjustments()
        elif self._section == "Histogram":
            self._panel_histogram()
        elif self._section == "Morphology":
            self._panel_morphology()
        elif self._section == "Filters":
            self._panel_filters()
        elif self._section == "Features":
            self._panel_features()
        elif self._section == "Compare":
            self._panel_compare()
        elif self._section == "Video":
            self._panel_video()

        self._controls.addStretch()

    def _panel_image(self) -> None:
        document = self._document
        if document is None:
            self._text("No image loaded. Click Open image or drag a file into the window.")
            return

        self._text(document.path.name, "metaValue")
        for label, value in (
            ("Resolution", document.resolution_text),
            ("Disk size", document.file_size_text),
            ("Format", document.file_format),
            ("Color mode", document.color_mode),
            ("Bit depth", f"{document.color_depth} bit"),
            ("Channels", document.channels_text),
        ):
            self._text(f"{label}:  {value}")

        self._button("Reset all edits…", self._confirm_reset_all)
        self._button("Close image", self._clear_image)

    def _panel_color(self) -> None:
        self._text("Click either image to inspect a pixel. Sampling works only in Color.")
        self._button("Convert image to grayscale", lambda: self._process("grayscale"))
        self._button("Reset Color changes", lambda: self._reset_current_section("Color"))

        if self._source is None or self._selected_pixel is None:
            self._text("Pixel not selected.")
            return

        x, y = self._selected_pixel
        sample = self._result if self._selected_origin == "Processed" else self._source
        color = tuple(int(n) for n in sample[y, x])
        values = pixel_colors(color)
        self._text(f"{self._selected_origin} pixel · x={x}, y={y}", "metaValue")
        swatch = QLabel("                         ")
        swatch.setFixedHeight(34)
        swatch.setStyleSheet(
            f"background-color: rgb({color[0]}, {color[1]}, {color[2]});"
            "border: 1px solid #454b57; border-radius: 8px;"
        )
        self._controls.addWidget(swatch)

        for model, value in values.items():
            self._text(model, "metaValue")
            self._text(value)

        self._text("CMYK is a simple device-independent approximation; LAB uses sRGB / D65.")

    def _panel_adjustments(self) -> None:
        self._text("Live brightness, contrast, and saturation adjustments.")
        self._slider("Brightness", -100, 100, self._brightness, self._change_brightness)
        self._slider("Contrast %", 0, 200, self._contrast, self._change_contrast)
        self._slider("Saturation %", 0, 200, self._saturation, self._change_saturation)
        self._button("Reset Adjustments", lambda: self._reset_current_section("Adjustments"))
        self._text("Grayscale correction", "metaValue")
        self._button("Linear contrast stretch", lambda: self._process("linear"))
        self._slider("Gamma × 100", 10, 300, round(self._gamma * 100), self._change_gamma)
        self._text("Gamma correction updates immediately when you move its slider.")

    def _panel_histogram(self) -> None:
        self._text("Compare pixel intensity distributions before and after processing.")
        self._text("Both histograms appear below their corresponding images. Switch between RGB, R, G, B and luminance in the chart header.")

    def _panel_morphology(self) -> None:
        self._text(
            "Turn effects on and off independently. Open a tool's settings "
            "without applying it. Multiple enabled effects combine in order."
        )
        for title, operation in _MORPHOLOGY:
            self._tool_card("Morphology", operation, title)
        self._button("Reset Morphology", lambda: self._reset_current_section("Morphology"))

    def _panel_filters(self) -> None:
        self._text(
            "Each filter has an independent ON/OFF switch. "
            "Expand its settings before or after enabling."
        )
        for title, operation in _FILTERS:
            self._tool_card("Filters", operation, title)
        self._button("Reset Filters", lambda: self._reset_current_section("Filters"))

    def _panel_features(self) -> None:
        self._text(
            "Enable detectors individually; disable one to remove only "
            "its contribution. SIFT uses its built-in feature settings."
        )
        for title, operation in _FEATURES:
            self._tool_card("Features", operation, title)
        self._button("Reset Features", lambda: self._reset_current_section("Features"))

    def _tool_card(self, section: str, operation: str, title: str) -> None:
        if self._pipeline is None:
            return
        defaults: dict[str, int] = {}
        settings: list[tuple] = []
        if section == "Morphology":
            defaults = {"kernel_size": 5}
            definitions = (("Kernel size", 1, 21, "kernel_size"),)
        elif operation in ("motion", "median"):
            defaults = {"kernel_size": 5}
            definitions = (("Kernel size", 3, 31, "kernel_size"),)
        elif operation == "canny":
            defaults = {"low": 70, "high": 150}
            definitions = (
                ("Lower threshold", 0, 254, "low"),
                ("Upper threshold", 1, 255, "high"),
            )
        elif operation in ("harris", "fast"):
            defaults = {"sensitivity": 20}
            definitions = (("Sensitivity", 1, 80, "sensitivity"),)
        else:
            definitions = ()

        parameters = self._tool_settings.setdefault(operation, defaults.copy())
        for caption, low, high, key in definitions:
            settings.append((caption, low, high, parameters[key], key))

        card = ToolCard(
            title=title,
            enabled=self._pipeline.is_enabled(section, operation),
            settings=settings,
            expanded=operation in self._expanded_tools,
            on_toggle=lambda active, sec=section, op=operation:
                self._toggle_tool(sec, op, active),
            on_parameter=lambda key, val, sec=section, op=operation:
                self._set_tool_parameter(sec, op, key, val),
            on_expand=lambda expanded, op=operation:
                self._set_tool_expanded(op, expanded),
        )
        self._tool_cards[operation] = card
        self._controls.addWidget(card)

    def _set_tool_expanded(self, operation: str, expanded: bool) -> None:
        if expanded:
            self._expanded_tools.add(operation)
        else:
            self._expanded_tools.discard(operation)

    def _set_tool_parameter(
        self, section: str, operation: str, key: str, value: int
    ) -> None:
        self._tool_settings.setdefault(operation, {})[key] = value
        # Keep legacy slider variables in sync for programmatic callers.
        if key == "kernel_size":
            if section == "Morphology":
                self._morph_k = value
            else:
                self._filter_k = value
        elif key == "low":
            self._low = value
        elif key == "high":
            self._high = value
        elif key == "sensitivity":
            self._sensitivity = value
        if self._pipeline is not None and self._pipeline.is_enabled(section, operation):
            self._process(operation, section=section)

    def _toggle_tool(self, section: str, operation: str, active: bool) -> None:
        if self._pipeline is None:
            return
        if active:
            self._process(operation, section=section)
            return
        self._result = self._pipeline.remove(section, operation)
        self._operation = "tool disabled"
        self._result_view.set_array(self._result)
        self._result_view.set_caption(f"Processed · {operation} off")
        self.statusBar().showMessage(f"{operation} disabled · other edits retained")

    def _panel_compare(self) -> None:
        self._text("Select 3–10 images for perceptual similarity comparison.")
        self._button("Choose 3–10 images", self._compare_images, primary=True)
        self._text(
            "The smallest difference-hash distance identifies the closest pair. "
            "The results appear in this dedicated workspace."
        )

    def _panel_video(self) -> None:
        self._text("Motion analysis on a dedicated video workspace.")
        self._selector(
            (
                ("Remove background · color subject", "background"),
                ("Foreground mask · white = movement", "mask"),
                ("Blur moving objects", "moving_blur"),
            ),
            self._choose_video_operation,
            self._video_operation,
        )
        self._slider(
            "Motion blur kernel", 15, 121,
            self._video_blur, self._change_video_blur,
        )
        self._button("Open video and play", self._open_video, primary=True)
        self._button("Stop video", self._stop_video)
        self._text(
            "Background removal keeps moving objects in their original colors "
            "and replaces the background with black. In the optional mask view, "
            "white indicates detected movement, black indicates background."
        )
        self._text(
            "Only motion is detected, not animals or people specifically. "
            "Camera movement can make furniture and walls appear to move. "
            "Use a fixed camera; the first frames help train the background."
        )
        self._text("Playback loops automatically until you press Stop video.")

    def _change_video_blur(self, value: int) -> None:
        self._video_blur = value
        if self._video_processor is not None:
            self._video_processor.blur_size = value

    def _change_brightness(self, value: int) -> None:
        self._brightness = value
        self._process("adjust")

    def _change_contrast(self, value: int) -> None:
        self._contrast = value
        self._process("adjust")

    def _change_saturation(self, value: int) -> None:
        self._saturation = value
        self._process("adjust")

    def _change_gamma(self, value: int) -> None:
        self._gamma = value / 100.0
        self._process("gamma")

    def _restore_adjustments(self) -> None:
        # Backwards-compatible alias for the Reset Adjustments control.
        self._reset_current_section("Adjustments")

    def _choose_morph(self, operation: str) -> None:
        self._morph_operation = operation
        if operation:
            self._process(operation)

    def _change_morph_kernel(self, value: int) -> None:
        self._morph_k = value
        if self._morph_operation:
            self._process(self._morph_operation)

    def _choose_filter(self, operation: str) -> None:
        self._filter_operation = operation
        if operation:
            self._process(operation)

    def _change_filter_kernel(self, value: int) -> None:
        self._filter_k = value
        if self._filter_operation:
            self._process(self._filter_operation)

    def _change_low(self, value: int) -> None:
        self._low = value
        if self._filter_operation == "canny":
            self._process("canny")

    def _change_high(self, value: int) -> None:
        self._high = value
        if self._filter_operation == "canny":
            self._process("canny")

    def _choose_feature(self, operation: str) -> None:
        self._feature_operation = operation
        if operation:
            self._process(operation)

    def _change_sensitivity(self, value: int) -> None:
        self._sensitivity = value
        if self._feature_operation:
            self._process(self._feature_operation)

    def _reset_current_section(self, section: str) -> None:
        if self._pipeline is None:
            return
        self._result = self._pipeline.reset_section(section)
        self._reset_section_controls(section)
        self._operation = f"reset {section.lower()}"
        self._result_view.set_array(self._result)
        self._result_view.set_caption(f"Processed · {self._operation}")
        self._render_controls()
        self.statusBar().showMessage(f"Reset {section} only · other edits preserved")

    def _reset_section_controls(self, section: str) -> None:
        if section == "Adjustments":
            self._brightness, self._contrast, self._saturation = 0, 100, 100
            self._gamma = 1.0
        elif section == "Morphology":
            self._morph_operation, self._morph_k = "", 5
            for _, operation in _MORPHOLOGY:
                self._tool_settings.pop(operation, None)
                self._expanded_tools.discard(operation)
        elif section == "Filters":
            self._filter_operation, self._filter_k = "", 5
            self._low, self._high = 70, 150
            for _, operation in _FILTERS:
                self._tool_settings.pop(operation, None)
                self._expanded_tools.discard(operation)
        elif section == "Features":
            self._feature_operation, self._sensitivity = "", 20
            for _, operation in _FEATURES:
                self._tool_settings.pop(operation, None)
                self._expanded_tools.discard(operation)

    def _reset_all_edits(self) -> None:
        if self._pipeline is None:
            return
        self._result = self._pipeline.reset_all()
        for section in ("Adjustments", "Morphology", "Filters", "Features"):
            self._reset_section_controls(section)
        self._operation = "original"
        self._result_view.set_array(self._result)
        self._result_view.set_caption("Processed · original")
        self._render_controls()
        self.statusBar().showMessage("All image edits reset")

    def _confirm_reset_all(self) -> None:
        if self._pipeline is None or not self._pipeline.steps:
            return
        choice = QMessageBox.question(
            self,
            "Reset all image edits?",
            "This will remove edits from ALL sections.\n"
            "To reset only one section, use its Reset button.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if choice == QMessageBox.StandardButton.Yes:
            self._reset_all_edits()

    def _process(self, operation: str, section: str | None = None) -> None:
        if self._pipeline is None:
            self.statusBar().showMessage("Open an image first")
            return
        if operation == "original":
            self._reset_all_edits()
            return

        resolved_section = section or self._section
        if resolved_section in ("Image", "Histogram", "Compare", "Video"):
            if operation == "grayscale":
                resolved_section = "Color"
            elif operation in ("adjust", "linear", "gamma"):
                resolved_section = "Adjustments"
            elif operation in {op for _, op in _MORPHOLOGY}:
                resolved_section = "Morphology"
            elif operation in {op for _, op in _FEATURES}:
                resolved_section = "Features"
            else:
                resolved_section = "Filters"

        if resolved_section in ("Morphology", "Filters", "Features"):
            defaults = {
                "kernel_size": self._morph_k if resolved_section == "Morphology" else self._filter_k,
                "low": self._low,
                "high": self._high,
                "sensitivity": self._sensitivity,
            }
            parameters = self._tool_settings.setdefault(operation, {})
            # Direct callers of _process (including tests) still work.
            for key, value in defaults.items():
                parameters.setdefault(key, value)
        else:
            parameters = {
                "brightness": self._brightness,
                "contrast": self._contrast,
                "saturation": self._saturation,
                "gamma": self._gamma,
            }

        try:
            result = self._pipeline.apply(resolved_section, operation, **parameters)
            self._result = result
            self._operation = operation
            self._result_view.set_array(result)
            self._result_view.set_caption(f"Processed · {operation}")
            if self._section == "Histogram":
                self._histogram_panel.set_images(self._source, self._result)
            self.statusBar().showMessage(f"{operation} enabled / updated")
        except (cv2.error, ValueError, OverflowError) as exc:
            QMessageBox.warning(self, "Processing error", str(exc))

    def _reset_preview(self) -> None:
        # Kept as a compatibility alias; only Image has the global reset UI.
        self._reset_all_edits()

    def _on_pixel_clicked(self, x: int, y: int, processed: bool = False) -> None:
        if self._section != "Color" or self._source is None or self._result is None:
            return
        self._selected_pixel = (x, y)
        self._selected_origin = "Processed" if processed else "Original"
        sample = self._result if processed else self._source
        red, green, blue = (int(value) for value in sample[y, x])
        self.statusBar().showMessage(
            f"{self._selected_origin} pixel ({x}, {y}) · RGB {red}, {green}, {blue}"
        )
        self._render_controls()

    def open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff *.webp);;All files (*.*)",
        )
        if path:
            self._load_image(Path(path))

    def _load_image(self, path: Path) -> None:
        try:
            document = ImageDocument.load(path)
            source = load_rgb(str(path))
        except (FileNotFoundError, OSError, ValueError) as exc:
            QMessageBox.critical(self, "Could not open image", str(exc))
            return

        self._stop_video()
        self._document = document
        self._source = source
        self._result = source.copy()
        self._selected_pixel = None
        self._selected_origin = "Original"
        self._operation = "original"
        self._pipeline = EditPipeline(source)
        for section in ("Adjustments", "Morphology", "Filters", "Features"):
            self._reset_section_controls(section)
        self._source_view.set_array(source)
        self._result_view.set_array(source)
        self._result_view.set_caption("Processed · original")
        if self._section in ("Compare", "Video"):
            self._switch_section("Image")
        self._update_navigation()
        self._render_controls()
        self.statusBar().showMessage(
            f"{document.path.name} · {document.resolution_text} · {document.file_format}"
        )

    def _clear_image(self) -> None:
        self._stop_video()
        self._document = None
        self._source = None
        self._result = None
        self._selected_pixel = None
        self._selected_origin = "Original"
        self._operation = "original"
        self._pipeline = None
        for section in ("Adjustments", "Morphology", "Filters", "Features"):
            self._reset_section_controls(section)
        self._source_view.clear_image()
        self._result_view.clear_image()
        if self._section not in ("Image", "Compare", "Video"):
            self._switch_section("Image")
        self._update_navigation()
        self._render_controls()
        self.statusBar().showMessage("Ready")

    def save_result(self) -> None:
        if self._result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save processed image", "pixelscope-output.png",
            "PNG image (*.png);;JPEG image (*.jpg);;BMP image (*.bmp)",
        )
        if not path:
            return
        try:
            Image.fromarray(self._result).save(path)
            self.statusBar().showMessage(f"Saved: {path}")
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Could not save image", str(exc))

    def _show_histograms(self) -> None:
        if self._source is None or self._result is None:
            self.statusBar().showMessage("Open an image first")
            return
        self._switch_section("Histogram")

    def _compare_images(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Choose 3–10 images", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff *.webp)",
        )
        if not paths:
            return
        if len(paths) < 3 or len(paths) > 10:
            QMessageBox.information(self, "Image count", "Choose between 3 and 10 images.")
            return
        try:
            pair = most_similar([Path(p) for p in paths])
            self._compare_left.set_array(load_rgb(str(pair.first)))
            self._compare_right.set_array(load_rgb(str(pair.second)))
            self._compare_left.set_caption(pair.first.name)
            self._compare_right.set_caption(pair.second.name)
            self._compare_info.setText(
                f"Closest pair from {len(paths)} images · "
                f"dHash distance: {pair.distance} / 64 (lower = closer)"
            )
            self.statusBar().showMessage(
                f"Most similar: {pair.first.name} and {pair.second.name}"
            )
        except (OSError, ValueError, cv2.error) as exc:
            QMessageBox.warning(self, "Image comparison failed", str(exc))

    def _choose_video_operation(self, operation: str) -> None:
        # Preserve the learned background while changing the display mode.
        self._video_operation = operation
        if hasattr(self, "_video_result_view"):
            self._video_result_view.set_caption(f"Video · {operation}")

    def _open_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open video", "",
            "Video files (*.mp4 *.avi *.mov *.mkv *.webm);;All files (*.*)",
        )
        if not path:
            return
        self._stop_video()
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            capture.release()
            QMessageBox.warning(self, "Video error", "Unable to open this video or codec.")
            return
        self._capture = capture
        self._video_path = path
        self._video_processor = VideoProcessor(blur_size=self._video_blur)
        self._video_info.setText(
            f"{Path(path).name} · Learning the background. "
            "Use a steady-camera clip for clear foreground separation."
        )
        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = fps if np.isfinite(fps) and fps > 0 else 25
        self._video_timer.start(max(16, round(1000 / min(fps, 60))))
        self._advance_video()

    def _advance_video(self) -> None:
        if self._capture is None or self._video_processor is None:
            self._stop_video()
            return

        ok, bgr = self._capture.read()
        if not ok:
            # Automatically replay and start learning the background anew.
            # Some codecs cannot seek: reopen the file as a fallback.
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, bgr = self._capture.read()
            if not ok and self._video_path is not None:
                self._capture.release()
                self._capture = cv2.VideoCapture(self._video_path)
                ok, bgr = self._capture.read()
            if not ok:
                self._stop_video()
                self.statusBar().showMessage("Video cannot be replayed")
                return
            self._video_processor = VideoProcessor(blur_size=self._video_blur)
            self._video_info.setText(
                f"{Path(self._video_path).name if self._video_path else 'Video'} · "
                "Looping playback · background model restarted"
            )

        # Bound processing resolution for a responsive side-by-side video UI.
        height, width = bgr.shape[:2]
        largest = max(height, width)
        if largest > 960:
            scale = 960.0 / largest
            bgr = cv2.resize(
                bgr, (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        processed = self._video_processor.process(rgb, self._video_operation)
        self._video_source_view.set_array(rgb)
        self._video_result_view.set_array(processed)
        self._video_result_view.set_caption(f"Video · {self._video_operation}")
        self.statusBar().showMessage(f"Video · {self._video_operation}")

    def _stop_video(self) -> None:
        self._video_timer.stop()
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._video_processor = None
        self._video_path = None

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if any(
            Path(url.toLocalFile()).suffix.lower() in _IMAGE_SUFFIXES
            for url in event.mimeData().urls() if url.isLocalFile()
        ):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        for url in event.mimeData().urls():
            if url.isLocalFile():
                path = Path(url.toLocalFile())
                if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
                    self._load_image(path)
                    event.acceptProposedAction()
                    return
        event.ignore()

    def closeEvent(self, event) -> None:
        self._stop_video()
        super().closeEvent(event)
