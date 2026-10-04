import sys

from PySide6.QtWidgets import QApplication

from .ui.main_window import MainWindow
from .ui.theme import APP_STYLESHEET


def run() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("PixelScope")
    app.setOrganizationName("After 4AM")
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLESHEET)

    window = MainWindow()
    window.show()

    return app.exec()
