# PixelScope

**A desktop playground for image processing and computer vision.**

PixelScope is a Python desktop app with a dark Qt UI for exploring color spaces, image enhancement, mathematical morphology, keypoint detection and basic video analysis. Built for the university course *Methods of Signal and Image Processing* and developed as a portfolio project.

## Features

| Workspace | Tools |
|---|---|
| **Image** | Open PNG/JPG/BMP/GIF/TIFF/WebP, drag & drop, inspect image metadata, preview original vs result, export output |
| **Color** | Click a pixel and inspect RGB, CMYK, HSL, LAB, HSV and YCbCr; convert image to grayscale |
| **Adjustments** | Brightness, contrast, saturation, linear grayscale contrast stretch and nonlinear gamma correction |
| **Histogram** | RGB/R/G/B/luminance graphs with side-by-side before/after comparison |
| **Morphology** | Erosion, dilation, opening, closing, morphological gradient (automatic Otsu binarization) |
| **Filters** | Sharpen, motion blur, emboss, median, Canny edges, Roberts edges |
| **Features** | Harris, SIFT and FAST detectors |
| **Compare** | Find two most similar images among 3–10 using difference hashes |
| **Video** | MOG2 background subtraction and Gaussian blur of moving objects |

All image tools recompute from the **original** image. Changes are previewed without overwriting the source file.

## Run on Windows

Requirements: **Python 3.11+** with an internet connection for dependency installation.

Open PowerShell in a chosen development directory and run:

\`\`\`powershell
git clone https://github.com/MishanyAG/PixelScope.git
cd PixelScope

py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
\`\`\`

If you've already cloned the repository, update your local copy instead:

\`\`\`powershell
git pull
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
\`\`\`

**Shortcuts:** Ctrl+O opens an image; Ctrl+S exports the processed result. Click the original image to sample a pixel; use the left navigation to switch tool categories.

## Tests

\`\`\`powershell
.\.venv\Scripts\python.exe -m pip install pytest
.\.venv\Scripts\python.exe -m pytest -q
\`\`\`

Automated smoke tests run on GitHub Actions with the Qt offscreen platform.

## Project layout

\`\`\`text
PixelScope/
├── main.py
├── requirements.txt
├── tests/
│   └── test_pixelscope.py
└── src/pixelscope/
    ├── app.py
    ├── core/
    │   ├── image_document.py
    │   ├── processing.py
    │   ├── similarity.py
    │   └── video.py
    └── ui/
        ├── main_window.py
        ├── image_viewer.py
        ├── dialogs.py
        └── theme.py
\`\`\`

## Notes and limitations

- CMYK conversion shows an **approximation** without a printer ICC profile; LAB assumes sRGB with the D65 reference white.
- The video tools depend on the file codecs available in the installed OpenCV build.
- dHash ranks overall visual similarity; it is not neural semantic image matching.
- Large source images and video frames may require substantial CPU/RAM. PySide6 tools run on the GUI thread in the current prototype, so very large inputs can reduce responsiveness.
- Tests validate algorithm outputs and the main window under an offscreen Qt runner; interactive Windows acceptance testing is still recommended.

**Status:** v0.2 — functional prototype.

Built by Mikhail Aleshin.
