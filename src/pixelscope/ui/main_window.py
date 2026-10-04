from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QDragEnterEvent, QDropEvent, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)

from ..core.image_document import ImageDocument
from .image_viewer import ImageViewer


_IMAGE_SUFFIXES = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self._document: ImageDocument | None = None
        self._meta_values: dict[str, QLabel] = {}

        self.setWindowTitle("PixelScope")
        self.resize(1440, 900)
        self.setMinimumSize(1080, 680)
        self.setAcceptDrops(True)

        self._create_actions()
        self._build_ui()
        self.statusBar().showMessage("Ready")

    def _create_actions(self) -> None:
        self._open_action = QAction("Open image…", self)
        self._open_action.setShortcut(QKeySequence.StandardKey.Open)
        self._open_action.triggered.connect(self.open_image)

        exit_action = QAction("Exit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)

        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(self._open_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        root_layout.addWidget(self._build_sidebar())
        root_layout.addWidget(self._build_workspace(), 1)

        self.setCentralWidget(root)

    def _build_sidebar(self) -> QFrame:
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(220)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(18, 22, 18, 18)
        layout.setSpacing(6)

        brand = QLabel("PIXELSCOPE")
        brand.setObjectName("brand")
        layout.addWidget(brand)

        subtitle = QLabel("Image processing lab")
        subtitle.setObjectName("brandSubtitle")
        layout.addWidget(subtitle)
        layout.addSpacing(22)

        sections = (
            ("Image", True),
            ("Color", False),
            ("Adjustments", False),
            ("Histogram", False),
            ("Morphology", False),
            ("Filters", False),
            ("Features", False),
            ("Compare", False),
            ("Video", False),
        )

        for title, active in sections:
            button = QPushButton(title)
            button.setProperty("nav", True)
            button.setProperty("active", active)
            button.setEnabled(active)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            layout.addWidget(button)

        layout.addSpacerItem(
            QSpacerItem(
                20,
                20,
                QSizePolicy.Policy.Minimum,
                QSizePolicy.Policy.Expanding,
            )
        )

        version = QLabel("v0.1 · foundation")
        version.setObjectName("muted")
        layout.addWidget(version)

        return sidebar

    def _build_workspace(self) -> QWidget:
        workspace = QWidget()
        layout = QVBoxLayout(workspace)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        layout.addWidget(self._build_topbar())
        layout.addWidget(self._build_content(), 1)

        return workspace

    def _build_topbar(self) -> QFrame:
        topbar = QFrame()
        topbar.setObjectName("topbar")
        topbar.setFixedHeight(76)

        layout = QHBoxLayout(topbar)
        layout.setContentsMargins(24, 14, 24, 14)

        title_box = QVBoxLayout()
        title_box.setSpacing(1)

        title = QLabel("Image workspace")
        title.setObjectName("sectionTitle")
        title_box.addWidget(title)

        subtitle = QLabel("Inspect an image before applying processing tools")
        subtitle.setObjectName("muted")
        title_box.addWidget(subtitle)

        layout.addLayout(title_box)
        layout.addStretch()

        open_button = QPushButton("Open image")
        open_button.setObjectName("primaryButton")
        open_button.setCursor(Qt.CursorShape.PointingHandCursor)
        open_button.clicked.connect(self.open_image)
        layout.addWidget(open_button)

        return topbar

    def _build_content(self) -> QWidget:
        content = QWidget()
        layout = QHBoxLayout(content)
        layout.setContentsMargins(24, 24, 0, 24)
        layout.setSpacing(24)

        self._viewer = ImageViewer()
        layout.addWidget(self._viewer, 1)

        layout.addWidget(self._build_inspector())

        return content

    def _build_inspector(self) -> QFrame:
        inspector = QFrame()
        inspector.setObjectName("inspector")
        inspector.setFixedWidth(320)

        layout = QVBoxLayout(inspector)
        layout.setContentsMargins(22, 4, 22, 4)
        layout.setSpacing(14)

        title = QLabel("Image info")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        self._path_label = QLabel("No image loaded")
        self._path_label.setObjectName("muted")
        self._path_label.setWordWrap(True)
        layout.addWidget(self._path_label)
        layout.addSpacing(8)

        fields = (
            ("resolution", "Resolution"),
            ("file_size", "File size"),
            ("format", "Format"),
            ("color_mode", "Color mode"),
            ("color_depth", "Color depth"),
            ("channels", "Channels"),
        )

        for key, label in fields:
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(12)

            name_label = QLabel(label)
            name_label.setObjectName("metaName")

            value_label = QLabel("—")
            value_label.setObjectName("metaValue")
            value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            value_label.setWordWrap(True)

            row_layout.addWidget(name_label)
            row_layout.addStretch()
            row_layout.addWidget(value_label)

            self._meta_values[key] = value_label
            layout.addWidget(row)

        layout.addStretch()

        self._close_button = QPushButton("Close image")
        self._close_button.setEnabled(False)
        self._close_button.clicked.connect(self._clear_image)
        layout.addWidget(self._close_button)

        return inspector

    def open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff *.webp);;All files (*.*)",
        )
        if path:
            self._load_image(Path(path))

    def _load_image(self, path: Path) -> None:
        try:
            document = ImageDocument.load(path)
            self._viewer.set_image(path)
        except (FileNotFoundError, OSError, ValueError) as exc:
            QMessageBox.critical(
                self,
                "Could not open image",
                f"PixelScope could not open this file.\n\n{exc}",
            )
            return

        self._document = document
        self._path_label.setText(document.path.name)
        self._path_label.setToolTip(str(document.path))

        self._meta_values["resolution"].setText(document.resolution_text)
        self._meta_values["file_size"].setText(document.file_size_text)
        self._meta_values["format"].setText(document.file_format)
        self._meta_values["color_mode"].setText(document.color_mode)
        self._meta_values["color_depth"].setText(f"{document.color_depth} bit")
        self._meta_values["channels"].setText(document.channels_text)

        self._close_button.setEnabled(True)
        self.statusBar().showMessage(
            f"{document.path.name} · {document.resolution_text} · {document.file_format}"
        )

    def _clear_image(self) -> None:
        self._document = None
        self._viewer.clear_image()
        self._path_label.setText("No image loaded")
        self._path_label.setToolTip("")

        for label in self._meta_values.values():
            label.setText("—")

        self._close_button.setEnabled(False)
        self.statusBar().showMessage("Ready")

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        urls = event.mimeData().urls()
        if any(self._is_supported_image(Path(url.toLocalFile())) for url in urls):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if self._is_supported_image(path):
                self._load_image(path)
                event.acceptProposedAction()
                return

        event.ignore()

    @staticmethod
    def _is_supported_image(path: Path) -> bool:
        return path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
