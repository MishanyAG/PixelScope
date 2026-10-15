"""PixelScope smoke tests. Run with: pytest -q"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest
from PIL import Image
from PySide6.QtWidgets import QApplication

from src.pixelscope.core.image_document import ImageDocument
from src.pixelscope.core.processing import load_rgb, pixel_colors, run_tool
from src.pixelscope.core.similarity import most_similar
from src.pixelscope.core.video import VideoProcessor
from src.pixelscope.ui.main_window import MainWindow


@pytest.fixture
def sample(tmp_path: Path) -> tuple[Path, np.ndarray]:
    y, x = np.mgrid[:90, :130]
    image = np.zeros((90, 130, 3), dtype=np.uint8)
    image[:, :, 0] = (x * 2) % 256
    image[:, :, 1] = (y * 3) % 256
    image[:, :, 2] = (x + y) % 256
    path = tmp_path / "sample.png"
    Image.fromarray(image).save(path)
    return path, image


@pytest.fixture(scope="session")
def app():
    instance = QApplication.instance() or QApplication([])
    yield instance


def test_document(sample):
    path, source = sample
    document = ImageDocument.load(path)
    assert document.resolution_text == "130 × 90 px"
    assert document.file_format == "PNG"
    assert document.color_depth == 24
    assert np.array_equal(load_rgb(str(path)), source)


@pytest.mark.parametrize(
    "operation",
    [
        "original", "grayscale", "adjust", "linear", "gamma",
        "erode", "dilate", "open", "close", "gradient",
        "invert", "sharpen", "motion", "emboss", "median", "canny", "roberts",
        "harris", "sift", "fast",
    ],
)
def test_image_processing(sample, operation):
    _, image = sample
    result = run_tool(
        image, operation,
        brightness=15,
        contrast=115,
        saturation=90,
        kernel_size=5,
        gamma=1.6,
    )
    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert np.array_equal(image[:, :, 0], np.mgrid[:90, :130][1] * 2 % 256)


def test_pixel_color_models():
    models = pixel_colors((255, 0, 0))
    assert set(models) == {"RGB", "CMYK", "HSL", "LAB", "HSV", "YCbCr"}
    assert models["RGB"] == "255, 0, 0"
    assert models["HSV"].startswith("0.0°")


def test_image_comparison(sample, tmp_path: Path):
    first, image = sample
    second = tmp_path / "duplicate.png"
    third = tmp_path / "other.png"
    Image.fromarray(image).save(second)
    Image.fromarray(np.flipud(image)).save(third)
    pair = most_similar([first, second, third])
    assert {pair.first, pair.second} == {first, second}
    assert pair.distance == 0


def test_video_processing(sample):
    _, image = sample
    processor = VideoProcessor()
    for operation in ("background", "moving_blur"):
        result = processor.process(image, operation)
        assert result.shape == image.shape


def test_window_load_and_features(sample, app):
    path, image = sample
    window = MainWindow()
    window._load_image(path)
    assert window._document is not None
    assert window._result is not None
    assert window._result.shape == image.shape
    for section in (
        "Color", "Adjustments", "Histogram", "Morphology",
        "Filters", "Features", "Compare", "Video",
    ):
        window._switch_section(section)
    window._process("canny")
    assert window._operation == "canny"
    window._switch_section("Color")
    window._on_pixel_clicked(10, 20)
    assert window._selected_pixel == (10, 20)
    window._clear_image()
    assert window._source is None
    window.close()
