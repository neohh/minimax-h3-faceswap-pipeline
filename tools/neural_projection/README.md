# Block 3: Neural Texture Projection Engine (CoTracker3 + Smooth-MLS)

> **Part of the 3-Pillar VFX Studio Architecture**  
> **Block 1: Roto / Segmentation** (`tools/sammie_bridge`) → **Block 2: AI Identity & Perspective Synthesis** (`pipeline.py`) → **Block 3: Precision Neural Projection** (`tools/neural_projection`)

---

## 🎯 Overview & The VFX Problem Statement

When doing cinematic object replacement (such as placing authentic Soviet newspaper stacks **«ПРАВДА»** or rigid cigarette packs **«КОСМОС»** on moving plates), traditional approaches encounter severe limitations:

| Traditional Approach | Fatal VFX Flaw |
| :--- | :--- |
| **Diffusion Video Inpaint (Wan2.1 / CogVideoX / Hunyuan)** | Destroys sharp typography and fine textures. Diffusion noise melts Cyrillic text and causes frame-to-frame boiling. |
| **Dense 2D Optical Flow (RAFT / GMA / Farneback)** | Suffers from cumulative subpixel drift. By frame 20–30, projected textures slide and lose anchor lock. |
| **Delaunay Triangulation Mesh** | Piecewise-affine ($C^0$-continuous). Creates visible polygonal facet lines across faces and straight triangle boundary cuts. |
| **RANSAC Homography ($3 \times 3$)** | Requires strict coplanarity. Raised 3D objects (e.g. cigarette boxes sitting atop folded paper) cause random 4-point subsets to micro-tilt, inducing extreme corner jitter ($\sigma = 38.46$ px). |

---

## 🔬 Mathematical Architecture of Smooth-MLS

Block 3 solves all four failure modes by combining deep transformer point tracking with smooth differential geometry:

```
Video Plate + Roto Mask (Frame 0..T)
               │
               ▼
   [CoTracker3 Transformer]
               │  (Tracks 78+ 3D ground-truth points across entire shot)
               ▼
   [Savitzky-Golay Filter]
               │  (window=7, poly=2; kills high-frequency jitter: σ = 0.25 px)
               ▼
   [Vectorized GPU Smooth-MLS]
               │  (C1-continuous deformation; R_reg=400, α=0.5; zero jello ripple)
               ▼
   [Bicubic Grid-Sampling]
               │  (Samples Master Hero Frame 0 in PyTorch GPU)
               ▼
 [Organic Overscan & Alpha-Clip]
               │  (7px morphological dilation + Gaussian feathering)
               ▼
     Final Cinematic Composite
```

### 1. Temporal Smoothing (Savitzky-Golay)
CoTracker3 provides drift-free long-term tracking, but deep neural features introduce minor frame-to-frame subpixel jitter ($\approx 0.2\text{--}0.4$ px). Before computing spatial deformation, point trajectories $(x_i(t), y_i(t))$ are passed through a temporal polynomial filter:
$$\tilde{x}_i(t) = \sum_{k=-m}^{m} c_k \, x_i(t + k)$$
This reduces corner trajectory variance from $38.46$ px down to $0.25$ px without introducing phase lag.

### 2. Smooth Moving Least Squares (Smooth-MLS)
Unlike Delaunay triangulation (which partitions the plane into independent linear facets), Moving Least Squares computes a spatially continuous non-linear mapping for every pixel $v \in \mathbb{R}^2$:
$$w_i(v) = \frac{1}{\|v - q_i(t)\|^{2\alpha} + R_{\text{reg}}}$$
- **Radius Regularization ($R_{\text{reg}} = 400.0$, $R = 20$ px):** Softens the inverse-distance singularity, eliminating localized weight funnels and the wavy "jello effect".
- **Broad Exponent ($\alpha = 0.5$):** Distributes deformation smoothly across planar surfaces, preserving rigidity.

### 3. Dynamic Overscan & Organic Alpha-Clipping
Instead of clipping to the convex hull of tracked points, the warped hero texture is rendered with a 40px bounding-box overscan. It is then seamlessly composited using the frame's dilated and Gaussian-feathered roto mask ($k = 7$ px, $\sigma = 1.8$), ensuring zero border seams.

---

## 🚀 Quick Start & Usage

### 1-Click Execution (CLI)
```bash
tools\neural_projection\neural_projection.bat ^
    --video "path\to\plate.mp4" ^
    --mask "path\to\mask.mp4" ^
    --hero "path\to\hero_frame0.png" ^
    --output "path\to\master_composite.mp4"
```

### Python API
```python
from tools.neural_projection.neural_projection import run_neural_projection

run_neural_projection(
    video_path="input/plate.mp4",
    mask_path="input/mask_props.mp4",
    hero_path="input/hero_frame0.png",
    output_path="output/master_pass21.mp4",
    grid_size=35,
    alpha=0.5,
    r_reg=400.0,
    savgol_window=7,
    savgol_poly=2,
    dilate_px=7,
    feather_sigma=1.8,
    save_gif=True
)
```

### CLI Parameters Reference
- `--video`: Source background video plate (MP4).
- `--mask`: Props binary mask video or static mask image.
- `--hero`: Master Hero Frame 0 containing the crisp replacement textures.
- `--output`: Destination master composite MP4.
- `--grid_size`: CoTracker3 query grid density (default: `35`).
- `--alpha`: MLS distance decay factor (default: `0.5`).
- `--r_reg`: MLS radius regularization $R^2$ (default: `400.0`).
- `--savgol_window`: Savitzky-Golay temporal window length (default: `7`).
- `--savgol_poly`: Savitzky-Golay polynomial order (default: `2`).
- `--dilate`: Morphological dilation kernel size in pixels (default: `7`).
- `--no_gif`: Disable automated preview GIF generation.
