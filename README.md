# PixelScope

A desktop playground for image processing and computer vision.

PixelScope is a student-built desktop application for exploring raster images, color models, image enhancement, morphology, feature detection, image comparison, and basic video processing through a clean GUI.

> **Status:** early development.

## Planned features

- Load and preview raster images
- Inspect image metadata: dimensions, file size, bit depth, format, channels
- Convert a selected pixel between RGB, CMYK, HSL, HSV, LAB, and YCbCr
- Grayscale conversion
- Brightness, saturation, and contrast adjustment
- RGB histograms with before/after comparison
- Linear and nonlinear grayscale correction
- Morphological operations: erosion, dilation, opening, closing, gradient
- Sharpening, motion blur, embossing, and median filtering
- Edge detection with Canny and Roberts operators
- Keypoint detection with Harris, SIFT, and FAST
- Find the two most similar images in a set
- Video background subtraction
- Blur moving objects in video

## Tech stack

- Python
- PySide6
- OpenCV
- NumPy
- Matplotlib

## Run locally

```bash
git clone https://github.com/MishanyAG/PixelScope.git
cd PixelScope

python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python main.py
```

## Project structure

```text
PixelScope/
├─ main.py
├─ requirements.txt
└─ src/
   └─ pixelscope/
      ├─ app.py
      ├─ core/
      └─ ui/
```

## About

PixelScope started as a practical assignment for the university course **Methods of Signal and Image Processing** and is being developed as a standalone portfolio project.
