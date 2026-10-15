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


def test_editor_starts_locked_then_unlocks(sample, app):
    path, _ = sample
    window = MainWindow()
    for section in ("Color", "Adjustments", "Histogram", "Morphology", "Filters", "Features"):
        assert not window._navigation[section].isEnabled()
    for section in ("Image", "Compare", "Video"):
        assert window._navigation[section].isEnabled()

    window._load_image(path)
    for section in ("Color", "Adjustments", "Histogram", "Morphology", "Filters", "Features"):
        assert window._navigation[section].isEnabled()

    window._clear_image()
    for section in ("Color", "Adjustments", "Histogram", "Morphology", "Filters", "Features"):
        assert not window._navigation[section].isEnabled()
    window.close()


def test_grayscale_then_brightness_keeps_prior_edit(sample, app):
    path, original = sample
    window = MainWindow()
    window._load_image(path)
    window._switch_section("Color")
    window._process("grayscale")
    gray = window._result.copy()
    assert np.all(gray[:, :, 0] == gray[:, :, 1])
    assert np.all(gray[:, :, 1] == gray[:, :, 2])

    window._switch_section("Adjustments")
    window._change_brightness(20)
    window._change_brightness(40)
    assert np.array_equal(
        window._result,
        run_tool(gray, "adjust", brightness=40),
    )
    assert np.all(window._result[:, :, 0] == window._result[:, :, 1])
    assert not np.array_equal(window._result, gray)
    assert np.array_equal(window._source, original)

    adjusted = window._result.copy()
    window._switch_section("Filters")
    window._process("invert")
    assert np.array_equal(window._result, 255 - adjusted)
    window.close()


def test_reset_adjustments_preserves_earlier_changes(sample, app):
    path, _ = sample
    window = MainWindow()
    window._load_image(path)
    window._process("grayscale")
    grayscale = window._result.copy()
    window._switch_section("Adjustments")
    window._change_brightness(48)
    window._restore_adjustments()
    assert np.array_equal(window._result, grayscale)
    window._reset_preview()
    assert np.array_equal(window._result, window._source)
    window.close()


def test_standalone_workspaces_do_not_touch_editor(sample, app):
    path, original = sample
    window = MainWindow()
    window._load_image(path)
    window._process("grayscale")
    edited = window._result.copy()

    window._switch_section("Compare")
    assert window._mode_stack.currentIndex() == 1
    assert window._open_button.isHidden()
    assert window._save_button.isHidden()
    assert not window._open_action.isEnabled()
    assert not window._save_action.isEnabled()
    assert window._compare_left._pixmap.isNull()

    window._switch_section("Video")
    assert window._mode_stack.currentIndex() == 2
    assert window._open_button.isHidden()
    assert window._video_source_view._pixmap.isNull()

    window._switch_section("Image")
    assert window._mode_stack.currentIndex() == 0
    assert not window._open_button.isHidden()
    assert window._save_button.isEnabled()
    assert np.array_equal(window._result, edited)
    assert np.array_equal(window._source, original)
    window.close()


def test_video_removes_background_and_blurs_only_movement():
    still = np.full((170, 170, 3), 70, dtype=np.uint8)
    moving = still.copy()
    moving[45:125, 50:130] = 240
    processor = VideoProcessor(blur_size=81)

    for _ in range(40):
        processor.process(still, "mask")
    mask = processor.process(moving, "mask")
    assert int(mask[85, 90, 0]) > 0
    assert int(mask[10, 10, 0]) == 0

    foreground = processor.process(moving, "background")
    assert np.array_equal(foreground[10, 10], np.zeros(3, dtype=np.uint8))
    assert np.array_equal(foreground[85, 90], moving[85, 90])

    blurred = processor.process(moving, "moving_blur")
    assert np.array_equal(blurred[10, 10], moving[10, 10])
    assert not np.array_equal(blurred[45, 50], moving[45, 50])


def test_compare_results_display_in_compare_workspace(sample, tmp_path, monkeypatch, app):
    path, rgb = sample
    second = tmp_path / "copy.png"
    third = tmp_path / "other.png"
    Image.fromarray(rgb).save(second)
    Image.fromarray(255 - rgb).save(third)

    monkeypatch.setattr(
        "src.pixelscope.ui.main_window.QFileDialog.getOpenFileNames",
        lambda *args, **kwargs: ([str(path), str(second), str(third)], ""),
    )

    window = MainWindow()
    window._load_image(path)
    original_result = window._result.copy()
    window._switch_section("Compare")
    window._compare_images()

    assert not window._compare_left._pixmap.isNull()
    assert not window._compare_right._pixmap.isNull()
    assert "dHash distance: 0 / 64" in window._compare_info.text()
    assert np.array_equal(window._result, original_result)
    assert window._mode_stack.currentIndex() == 1
    window.close()


def test_video_frames_display_only_in_video_workspace(sample, app):
    _, rgb = sample

    class FakeCapture:
        def __init__(self, rgb):
            self.rgb = rgb
            self.released = False

        def read(self):
            return True, cv2.cvtColor(self.rgb, cv2.COLOR_RGB2BGR)

        def release(self):
            self.released = True

    window = MainWindow()
    window._switch_section("Video")
    window._capture = FakeCapture(rgb)
    window._video_processor = VideoProcessor()
    window._advance_video()

    assert not window._video_source_view._pixmap.isNull()
    assert not window._video_result_view._pixmap.isNull()
    assert window._source is None
    assert window._source_view._pixmap.isNull()
    assert window._result_view._pixmap.isNull()

    capture = window._capture
    window._switch_section("Compare")
    assert capture.released
    assert window._mode_stack.currentIndex() == 1
    window.close()


def test_tab_reset_replays_other_tabs_in_correct_order(sample, app):
    path, rgb = sample
    window = MainWindow()
    window._load_image(path)
    window._switch_section("Color")
    window._process("grayscale")
    gray = window._result.copy()

    window._switch_section("Adjustments")
    window._change_brightness(30)
    light = window._result.copy()
    window._switch_section("Morphology")
    window._choose_morph("erode")
    window._change_morph_kernel(9)
    assert window._morph_k == 9
    window._switch_section("Filters")
    window._choose_filter("invert")
    assert not np.array_equal(window._result, 255 - light)

    window._switch_section("Morphology")
    window._reset_current_section("Morphology")
    assert np.array_equal(window._result, 255 - light)
    assert window._morph_k == 5
    assert window._morph_operation == ""
    assert not any(step.section == "Morphology" for step in window._pipeline.steps)
    assert len(window._pipeline.steps) == 3

    window._switch_section("Adjustments")
    window._reset_current_section("Adjustments")
    assert np.array_equal(window._result, 255 - gray)
    assert window._brightness == 0
    assert window._contrast == 100
    assert window._saturation == 100
    assert window._gamma == 1.0

    window._switch_section("Color")
    window._reset_current_section("Color")
    assert np.array_equal(window._result, 255 - rgb)
    window.close()


def test_live_sliders_do_not_stack_changes(sample, app):
    path, _ = sample
    window = MainWindow()
    window._load_image(path)
    window._switch_section("Adjustments")
    window._change_brightness(20)
    window._change_brightness(40)
    window._change_contrast(120)
    assert len(window._pipeline.steps) == 1
    assert window._pipeline.steps[0].operation == "adjust"
    assert np.array_equal(
        window._result,
        run_tool(window._source, "adjust", brightness=40, contrast=120),
    )
    window._change_gamma(150)
    window._change_gamma(160)
    assert len(window._pipeline.steps) == 2
    assert window._pipeline.steps[-1].operation == "gamma"
    window._reset_current_section("Adjustments")
    assert not window._pipeline.steps
    assert np.array_equal(window._result, window._source)
    window.close()


def test_no_redundant_apply_buttons_or_menus(sample, app):
    window = MainWindow()
    assert window.menuBar().actions() == []
    window._load_image(sample[0])
    from PySide6.QtWidgets import QPushButton
    for section in ("Morphology", "Filters", "Features"):
        window._switch_section(section)
        labels = [button.text() for button in window._controls.parentWidget().findChildren(QPushButton)]
        assert not any(text.startswith("Apply ") for text in labels)
        assert any(text.startswith("Reset ") for text in labels)
    window.close()


def test_video_loops_when_reaching_last_frame(sample, app):
    _, rgb = sample

    class LoopingCapture:
        def __init__(self, image):
            self.image = image
            self.index = 0
            self.released = False
            self.seek_calls = 0

        def read(self):
            if self.index >= 2:
                return False, None
            self.index += 1
            return True, cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)

        def set(self, prop, index):
            assert prop == cv2.CAP_PROP_POS_FRAMES
            self.index = int(index)
            self.seek_calls += 1
            return True

        def release(self):
            self.released = True

    window = MainWindow()
    window._switch_section("Video")
    cap = LoopingCapture(rgb)
    window._capture = cap
    window._video_path = "dummy.mp4"
    window._video_processor = VideoProcessor()
    window._advance_video()
    window._advance_video()
    first_processor = window._video_processor
    window._advance_video()

    assert cap.seek_calls == 1
    assert window._capture is cap
    assert not cap.released
    assert window._video_processor is not first_processor
    assert "Looping playback" in window._video_info.text()
    assert not window._video_source_view._pixmap.isNull()
    window._stop_video()
    assert cap.released
    window.close()
