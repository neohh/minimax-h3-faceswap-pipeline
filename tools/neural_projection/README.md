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

## 🔬 Mathematical Architecture of Depth-Aware Smooth-MLS

Block 3 solves all failure modes by combining deep transformer point tracking, 3D bilateral depth decoupling, and smooth differential geometry:

```
Video Plate + Roto Mask + Depth Map Video (Frame 0..T)
                          │
                          ▼
              [CoTracker3 Transformer]
                          │  (Tracks 78+ 3D ground-truth points across entire shot)
                          ▼
              [Savitzky-Golay Filter]
                          │  (window=7, poly=2; kills high-frequency jitter: σ = 0.25 px)
                          ▼
     [Depth-Aware Bilateral Smooth-MLS]
                          │  (w = w_space * exp(-ΔD^2 / 2σ_d^2); decouples 3D objects)
                          ▼
     [Flow Field Gaussian Smoothing]
                          │  (GaussianBlur on dx, dy with k=31; silences micro-ripples)
                          ▼
            [Bicubic Grid-Sampling]
                          │  (Samples Master Hero Frame 0 in PyTorch GPU)
                          ▼
       [Organic Overscan & Alpha-Clip]
                          │  (7px morphological dilation + Gaussian feathering)
                          ▼
              Final Cinematic Composite
```

### 1. Depth-Aware Bilateral Weighting (3D Layer Isolation)
When raised 3D objects (like a cigarette pack standing 4 cm above a table) share a scene with flat objects (newspaper), 3D parallax causes differing motion vectors. Bilateral weighting penalizes depth divergence:
$$w_i(v) = \frac{1}{\|v - q_i(t)\|^{2\alpha} + R_{\text{reg}}} \cdot \exp\left(-\frac{|D(v) - D(q_i)|^2}{2\sigma_{\text{depth}}^2}\right)$$
- If pixel $v$ is on the newspaper ($D \approx 18$) and point $q_i$ is on the pack ($D \approx 103$), the weight drops to zero. Points on the pack never deform the paper.

### 2. Golden Regularization & Flow Field Smoothing
- **Golden Parameters:** $R_{\text{reg}} = 250.0, \alpha = 0.65$ provides the optimal planar rigidity without global centroid pull.
- **Flow Field Smoothing ($k=31$):** Gaussian blur applied directly to the displacement field $(dx, dy)$ eliminates high-frequency surface wrinkling while preserving exact subpixel boundary anchors.

---

## 🚀 Quick Start & Usage

### 1-Click Execution (CLI)
```bash
tools\neural_projection\neural_projection.bat ^
    --video "path\to\plate.mp4" ^
    --mask "path\to\mask.mp4" ^
    --hero "path\to\hero_frame0.png" ^
    --depth "path\to\depth_map.mp4" ^
    --output "path\to\master_composite.mp4"
```

### Python API
```python
from tools.neural_projection.neural_projection import run_neural_projection

run_neural_projection(
    video_path="input/plate.mp4",
    mask_path="input/mask_props.mp4",
    hero_path="input/hero_frame0.png",
    output_path="output/master_final.mp4",
    depth_path="input/depth_map.mp4",
    grid_size=35,
    alpha=0.65,
    r_reg=250.0,
    sigma_depth=25.0,
    flow_blur=31,
    savgol_window=7,
    savgol_poly=2,
    dilate_px=7,
    feather_sigma=1.8,
    save_gif=True
)
```
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
