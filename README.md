# PixelScope

**A desktop playground for image processing and computer vision.**

PixelScope is a Python desktop app with a dark Qt UI for exploring color spaces, image enhancement, mathematical morphology, keypoint detection and basic video analysis. Built for the university course *Methods of Signal and Image Processing* and developed as a portfolio project.

## Features

| Workspace | Tools |
|---|---|
| **Image** | Open PNG/JPG/BMP/GIF/TIFF/WebP, drag & drop, inspect image metadata, preview original vs result, export output |
| **Color** | Click either the original or processed image to inspect RGB, CMYK, HSL, LAB, HSV and YCbCr; convert image to grayscale |
| **Adjustments** | Brightness, contrast, saturation, linear grayscale contrast stretch and nonlinear gamma correction |
| **Histogram** | Embedded original/processed histograms below the two images, with an RGB/R/G/B/luminance switch |
| **Morphology** | Erosion, dilation, opening, closing, morphological gradient (automatic Otsu binarization) |
| **Filters** | Color inversion, sharpen, motion blur, emboss, median, Canny edges, Roberts edges |
| **Features** | Harris, SIFT and FAST detectors |
| **Compare** | Find two most similar images among 3–10 using difference hashes |
| **Video** | Dedicated video workspace: color foreground extraction, binary motion mask, and adjustable blur of moving objects |

Image tools **build on the currently edited result**, using replayable edit history. For example, grayscale → brightness keeps the image grayscale. In Morphology, Filters and Features, every tool has an **obvious square ON/OFF checkbox**. The settings arrow appears only for tools with adjustable parameters. Opening settings changes nothing, enabling adds one effect, and disabling removes just that effect. Multiple effects can be combined deliberately and tuned independently. Adjustment sliders still update immediately without stacking the same effect repeatedly. Each editing tab has a **Reset** button that removes only edits from that tab, then recomputes the remaining edits in their original order. Reset also returns that tab's controls to their defaults. The Image tab has a separate, clearly labeled **Reset all edits** command with confirmation. When exporting, PixelScope suggests the source filename plus **_edited** (for example, `portrait.jpg` becomes `portrait_edited.jpg`). For GIF sources it suggests PNG, because it edits a single frame. PixelScope protects the original file from accidental overwrite.

## Run on Windows

Requirements: **Python 3.11+** with an internet connection for dependency installation.

Open PowerShell in a chosen development directory and run:

```powershell
git clone https://github.com/MishanyAG/PixelScope.git
cd PixelScope

py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

If you've already cloned the repository, update your local copy instead:

```powershell
git pull
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

**Shortcuts:** Ctrl+O opens an image; Ctrl+S exports the processed result. On the Color tab, click either image to sample that pixel. Pixel sampling is inactive on all other tabs. Use the left navigation to switch tools. Image processing tools are disabled until an image is loaded. Compare and Video have independent viewers, and image Open/Save controls disappear there. The redundant File/View menu bar is hidden.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pip install pytest
.\.venv\Scripts\python.exe -m pytest -q
```

Automated smoke tests run on GitHub Actions with the Qt offscreen platform.

## Project layout

```text
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
```

## Course materials

- [Theoretical assignment (Russian, all 11 questions)](docs/theory.md)
- [Oral defense quick reference (Russian)](docs/defense.md)

## Notes and limitations

- CMYK conversion shows an **approximation** without a printer ICC profile; LAB assumes sRGB with the D65 reference white.
- Video processing is motion-based, not dog/person detection. MOG2 estimates a background over time; moving the camera can cause furniture and walls to be detected as foreground. Video automatically loops and resets background training at the beginning of each replay. Codecs still depend on the installed OpenCV build.
- dHash ranks overall visual similarity; it is not neural semantic image matching.
- Large source images and video frames may require substantial CPU/RAM. PySide6 tools run on the GUI thread in the current prototype, so very large inputs can reduce responsiveness.
- Tests validate algorithm outputs and the main window under an offscreen Qt runner; interactive Windows acceptance testing is still recommended.

**Status:** v0.5.1 — clearer effect checkboxes, settings-only expanders, sensible export filenames.

Built by Mikhail Aleshin.
