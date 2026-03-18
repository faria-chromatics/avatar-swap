# Avatar Swap Image

Face swap in images using InsightFace + InSwapper + CodeFormer + Real-ESRGAN. Available in both Python and .NET implementations with identical pipelines.

## Architecture

```
Source Face Image    Target Image
       |                  |
       v                  v
  [Face Detection] — SCRFD det_10g (640x640, 3 stride levels)
       |                  |
       v                  |
  [Face Embedding]        |
   ArcFace w600k_r50      |
   112x112 aligned face   |
   → 512-dim vector       |
       |                  |
       v                  v
  [Face Swap] — InSwapper 128x128
   source embedding + target aligned face → swapped face
   paste back onto target image
       |
       v
  [Full Image Upscale] — Real-ESRGAN x2
   64x64 tile-based processing
   entire image upscaled (not just face)
       |
       v
  [Face Enhancement] — CodeFormer 512x512
   re-detect face in upscaled image
   align → enhance → color correct → blend back
       |
       v
  [Downscale] — Lanczos4
   back to original resolution
       |
       v
   output/result.jpg
```

### Models (6 ONNX files, ~1.1 GB total)

| Model | File | Size | Input | Purpose |
|-------|------|------|-------|---------|
| SCRFD | `det_10g.onnx` | 17 MB | [1,3,640,640] | Face detection + 5-point landmarks |
| ArcFace | `w600k_r50.onnx` | 167 MB | [1,3,112,112] | 512-dim face identity embedding |
| InSwapper | `inswapper_128.onnx` | 529 MB | [1,3,128,128] + [1,512] | Face swap at 128x128 |
| CodeFormer | `codeformer.onnx` | 360 MB | [1,3,512,512] + weight | Face restoration/enhancement |
| Real-ESRGAN | `real_esrgan_x2.onnx` | 64 MB | [1,3,64,64] | 2x image upscale |
| emap | `emap.bin` | 1 MB | — | InSwapper embedding transform matrix |

### Why upscale then downscale?

InSwapper processes faces at 128x128 — the swapped face region is blurry compared to the rest of the image. Real-ESRGAN upscales the **entire** image uniformly so there's no quality mismatch between the face and background. CodeFormer then sharpens just the face. Downscaling back to original resolution produces a clean, consistent result.

---

## face-swap-python

### Prerequisites

- Python 3.10+
- ~1.1 GB disk space for models

### Setup

```bash
cd face-swap-python
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### Model setup

Models download automatically on first run (`buffalo_l` from InsightFace). For `inswapper_128.onnx`, `codeformer.onnx`, and `real_esrgan_x2.onnx`, place them in `face-swap-python/models/`.

### Run

```bash
# Place images
#   input/source_face.jpg  — the face to swap in
#   input/target.jpg       — the image whose face gets replaced

python demo.py
```

Output saved to `output/result.jpg` (enhanced) and `output/result_raw.jpg` (raw swap).

---

## face-swap-dotnet

### Prerequisites

- .NET 8.0 SDK
- ~1.1 GB disk space for models

### Setup

```bash
cd face-swap-dotnet
dotnet restore
```

### Model setup

All 6 model files must be in `face-swap-dotnet/models/`:

```
models/
├── det_10g.onnx          # from ~/.insightface/models/buffalo_l/
├── w600k_r50.onnx        # from ~/.insightface/models/buffalo_l/
├── inswapper_128.onnx
├── codeformer.onnx
├── real_esrgan_x2.onnx
└── emap.bin              # extracted from inswapper_128.onnx (see below)
```

To extract `emap.bin` (one-time, requires Python):

```bash
cd face-swap-python
python -c "import onnx; from onnx import numpy_helper; import numpy as np; m=onnx.load('models/inswapper_128.onnx'); np.save('models/emap.bin', numpy_helper.to_array(m.graph.initializer[-1]).astype('float32').tobytes())"
```

Or if already generated, copy from `face-swap-python/models/emap.bin`.

### Run

```bash
# Place images
#   input/source_face.jpg
#   input/target.jpg

dotnet run
```

### .NET project structure

```
FaceSwap.csproj       — project file (OnnxRuntime + OpenCvSharp4)
Program.cs            — main pipeline (8 steps)
FaceDetector.cs       — SCRFD face detection (det_10g.onnx)
FaceSwapper.cs        — InSwapper + ArcFace embedding
FaceEnhancer.cs       — CodeFormer face enhancement
ImageUpscaler.cs      — Real-ESRGAN tile-based upscaling
ImageUtils.cs         — face alignment, elliptical blending, color transfer, NMS
FaceInfo.cs           — face data model (bbox, landmarks, embedding)
```

The .NET version reimplements InsightFace's face detection pipeline from scratch (SCRFD anchor generation, bbox/landmark decoding, NMS) since there is no InsightFace library for .NET.

---

## Performance (CPU)

| Step | Time |
|------|------|
| Face detection | < 1s |
| Face swap | ~1s |
| Real-ESRGAN upscale | 30-40s |
| CodeFormer enhance | 4-5s |
| **Total** | **~40-50s** |

All inference runs on CPU via ONNX Runtime's `CPUExecutionProvider`.
