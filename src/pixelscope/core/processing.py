"""Pure image-processing operations. All public functions use RGB uint8 arrays."""

from __future__ import annotations

import colorsys
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps


def load_rgb(path: str) -> np.ndarray:
    """Decode a file, honor EXIF orientation, and composite transparency on white."""
    with Image.open(path) as raw:
        image = ImageOps.exif_transpose(raw)
        if image.mode in ("RGBA", "LA") or "transparency" in image.info:
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, rgba).convert("RGB")
        else:
            image = image.convert("RGB")
        return np.asarray(image, dtype=np.uint8).copy()


def to_rgb(gray: np.ndarray) -> np.ndarray:
    if gray.ndim == 2:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return gray


def grayscale(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return to_rgb(gray)


def adjust(rgb: np.ndarray, brightness: int, contrast: int, saturation: int) -> np.ndarray:
    """Brightness is additive; contrast and saturation are 0..200 percent."""
    source = rgb.astype(np.float32)
    source = (source - 127.5) * (contrast / 100.0) + 127.5 + brightness
    adjusted = np.clip(source, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(adjusted, cv2.COLOR_RGB2HSV)
    hsv[:, :, 1] = np.clip(
        hsv[:, :, 1].astype(np.float32) * saturation / 100.0, 0, 255
    ).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def grayscale_correction(rgb: np.ndarray, operation: str, gamma: float = 1.0) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if operation == "linear":
        corrected = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    elif operation == "gamma":
        corrected = np.clip(
            255 * np.power(gray.astype(np.float32) / 255, gamma), 0, 255
        ).astype(np.uint8)
    else:
        raise ValueError(f"Unknown grayscale correction: {operation}")
    return to_rgb(corrected)


_MORPH_OPERATIONS = {
    "erode": cv2.MORPH_ERODE,
    "dilate": cv2.MORPH_DILATE,
    "open": cv2.MORPH_OPEN,
    "close": cv2.MORPH_CLOSE,
    "gradient": cv2.MORPH_GRADIENT,
}


def morphology(rgb: np.ndarray, operation: str, kernel_size: int = 5) -> np.ndarray:
    """Binary morphology with automatic Otsu threshold selection."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    output = cv2.morphologyEx(binary, _MORPH_OPERATIONS[operation], kernel)
    return to_rgb(output)


def apply_filter(
    rgb: np.ndarray,
    operation: str,
    kernel_size: int = 5,
    canny_low: int = 70,
    canny_high: int = 150,
) -> np.ndarray:
    """Apply spatial filters or edge operators."""
    k = max(3, kernel_size | 1)
    if operation == "sharpen":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(rgb, -1, kernel)
    if operation == "motion":
        kernel = np.eye(k, dtype=np.float32) / k
        return cv2.filter2D(rgb, -1, kernel)
    if operation == "emboss":
        kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)
        return cv2.convertScaleAbs(
            cv2.filter2D(rgb.astype(np.float32), cv2.CV_32F, kernel) + 128,
            alpha=1, beta=0,
        )
    if operation == "median":
        return cv2.medianBlur(rgb, k)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if operation == "canny":
        return to_rgb(cv2.Canny(gray, canny_low, max(canny_low + 1, canny_high)))
    if operation == "roberts":
        source = gray.astype(np.float32)
        gx = cv2.filter2D(source, cv2.CV_32F, np.array([[1, 0], [0, -1]], dtype=np.float32))
        gy = cv2.filter2D(source, cv2.CV_32F, np.array([[0, 1], [-1, 0]], dtype=np.float32))
        edge = np.clip(np.hypot(gx, gy), 0, 255).astype(np.uint8)
        return to_rgb(edge)
    raise ValueError(f"Unknown filter: {operation}")


def features(rgb: np.ndarray, detector: str, sensitivity: int = 20) -> np.ndarray:
    """Mark interest points for Harris, SIFT, or FAST."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    canvas = rgb.copy()

    if detector == "harris":
        response = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
        response = cv2.dilate(response, None)
        threshold = (sensitivity / 100.0) * float(response.max())
        mask = (response > threshold).astype(np.uint8) * 255
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        for x, y in centroids[1:501]:
            cv2.circle(canvas, (round(x), round(y)), 4, (255, 72, 95), 1, cv2.LINE_AA)
        return canvas

    if detector == "sift":
        sift = cv2.SIFT_create(nfeatures=500)
        keypoints = sift.detect(gray, None)
    elif detector == "fast":
        fast = cv2.FastFeatureDetector_create(
            threshold=max(1, sensitivity), nonmaxSuppression=True
        )
        keypoints = fast.detect(gray, None)
        keypoints = sorted(keypoints, key=lambda p: p.response, reverse=True)[:500]
    else:
        raise ValueError(f"Unknown detector: {detector}")

    return cv2.drawKeypoints(
        rgb, keypoints, None,
        color=(255, 72, 95),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )


def pixel_colors(rgb: tuple[int, int, int]) -> dict[str, str]:
    """Display conversions for a selected pixel (sRGB / approximate device CMYK)."""
    r, g, b = (int(v) for v in rgb)
    rn, gn, bn = (r / 255.0, g / 255.0, b / 255.0)
    h, s, v = colorsys.rgb_to_hsv(rn, gn, bn)
    hh, lightness, hs = colorsys.rgb_to_hls(rn, gn, bn)

    k = 1.0 - max(rn, gn, bn)
    if k >= 1.0:
        c = m = y = 0.0
    else:
        c, m, y = ((1 - rn - k) / (1 - k),
                   (1 - gn - k) / (1 - k),
                   (1 - bn - k) / (1 - k))

    one = np.asarray([[[rn, gn, bn]]], dtype=np.float32)
    lab = cv2.cvtColor(one, cv2.COLOR_RGB2Lab)[0, 0]

    # OpenCV's 8-bit YCrCb representation is ordered Y, Cr, Cb.
    ycrcb = cv2.cvtColor(
        np.asarray([[[r, g, b]]], dtype=np.uint8), cv2.COLOR_RGB2YCrCb
    )[0, 0]
    y_value, cr, cb = (int(x) for x in ycrcb)

    return {
        "RGB": f"{r}, {g}, {b}",
        "CMYK": f"{c * 100:.1f}%, {m * 100:.1f}%, {y * 100:.1f}%, {k * 100:.1f}%",
        "HSL": f"{hh * 360:.1f}°, {hs * 100:.1f}%, {lightness * 100:.1f}%",
        "HSV": f"{h * 360:.1f}°, {s * 100:.1f}%, {v * 100:.1f}%",
        "LAB": f"L* {lab[0]:.1f}, a* {lab[1]:.1f}, b* {lab[2]:.1f}",
        "YCbCr": f"Y {y_value}, Cb {cb}, Cr {cr}",
    }


def run_tool(
    rgb: np.ndarray,
    operation: str,
    *,
    brightness: int = 0,
    contrast: int = 100,
    saturation: int = 100,
    kernel_size: int = 5,
    low: int = 70,
    high: int = 150,
    sensitivity: int = 20,
    gamma: float = 1.0,
) -> np.ndarray:
    if operation == "original":
        return rgb.copy()
    if operation == "grayscale":
        return grayscale(rgb)
    if operation == "adjust":
        return adjust(rgb, brightness, contrast, saturation)
    if operation in ("linear", "gamma"):
        return grayscale_correction(rgb, operation, gamma)
    if operation in _MORPH_OPERATIONS:
        return morphology(rgb, operation, kernel_size)
    if operation in ("sharpen", "motion", "emboss", "median", "canny", "roberts"):
        return apply_filter(rgb, operation, kernel_size, low, high)
    if operation in ("harris", "sift", "fast"):
        return features(rgb, operation, sensitivity)
    raise ValueError(f"Unknown operation: {operation}")
