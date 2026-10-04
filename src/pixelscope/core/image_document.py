from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image


_MODE_DEPTH = {
    "1": 1,
    "L": 8,
    "P": 8,
    "I;16": 16,
    "I;16L": 16,
    "I;16B": 16,
    "I": 32,
    "F": 32,
    "RGB": 24,
    "RGBA": 32,
    "CMYK": 32,
    "YCbCr": 24,
    "LAB": 24,
    "HSV": 24,
}


@dataclass(slots=True, frozen=True)
class ImageDocument:
    path: Path
    width: int
    height: int
    file_size: int
    file_format: str
    color_mode: str
    color_depth: int
    channels: tuple[str, ...]

    @classmethod
    def load(cls, path: str | Path) -> "ImageDocument":
        image_path = Path(path).expanduser().resolve()

        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with Image.open(image_path) as image:
            width, height = image.size
            file_format = image.format or image_path.suffix.lstrip(".").upper() or "Unknown"
            color_mode = image.mode
            channels = tuple(image.getbands())
            color_depth = _MODE_DEPTH.get(color_mode, max(8, 8 * len(channels)))

        return cls(
            path=image_path,
            width=width,
            height=height,
            file_size=image_path.stat().st_size,
            file_format=file_format,
            color_mode=color_mode,
            color_depth=color_depth,
            channels=channels,
        )

    @property
    def resolution_text(self) -> str:
        return f"{self.width} × {self.height} px"

    @property
    def file_size_text(self) -> str:
        size = float(self.file_size)
        units = ("B", "KB", "MB", "GB")

        for unit in units:
            if size < 1024 or unit == units[-1]:
                precision = 0 if unit == "B" else 2
                return f"{size:.{precision}f} {unit}"
            size /= 1024

        return f"{self.file_size} B"

    @property
    def channels_text(self) -> str:
        return ", ".join(self.channels)
