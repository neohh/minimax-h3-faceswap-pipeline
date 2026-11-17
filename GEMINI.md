# Antigravity Persistent VFX Memory & Guidelines 🎬📐

## The 3-Pillar VFX Studio Architecture

When performing actor replacement or high-fidelity micro-prop replacement (e.g. authentic Cyrillic typography, folded newspapers «ПРАВДА», rigid cigarette packaging «КОСМОС»), ALWAYS adhere to the decoupled 3-block pipeline:

```
Block 1: Roto & Matting (tools/sammie_bridge)
   ↓  [Zero-Click IPC Handshake: temp/latest_export.json]
Block 2: AI Identity & Perspective Synthesis (pipeline.py)
   ↓  [Dynamic 768p Zoom Crop + DiT / FLUX Inpaint -> Hero Frame 0]
Block 3: Precision Neural Projection (tools/neural_projection)
   ↓  [CoTracker3 + Depth Anything V2 + Smooth-MLS + Flow Blur]
Final Cinematic Composite MP4
```

---

## Block 3: The Golden Neural Projection Specification

### Mathematical Formulation
To eliminate polygon seams, jello ripples, homography corner jitter, and 3D parallax drift:

1. **Temporal Jitter Filtering (Savitzky-Golay):**
   - Filter all tracked point coordinates over time using a polynomial filter ($w = 7, p = 2$).
   - Reduces corner jitter from $38.46$ px to $0.25$ px.

2. **Depth-Aware Bilateral Weighting (3D Decoupling):**
   - When raised objects (e.g. cigarette pack, $\Delta D \approx 85$) sit on flat surfaces (newspaper), isolate their motion planes using the Depth Map (Depth Anything V2):
   $$w_i(v) = \frac{1}{\|v - q_i(t)\|^{2\alpha} + R_{\text{reg}}} \cdot \exp\left(-\frac{|D(v) - D(q_i)|^2}{2\sigma_{\text{depth}}^2}\right)$$
   - Prevents the raised object from dragging or distorting the flat plane.

3. **Golden Regularization Parameters:**
   - **$R_{\text{reg}} = 250.0$** (radius $\approx 15.8$ px)
   - **$\alpha = 0.65$**
   - **$\sigma_{\text{depth}} = 25.0$**

4. **Flow Field Gaussian Smoothing:**
   - Apply Gaussian blur ($k = 31$, $\sigma = 0$) directly to the displacement vector maps $(dx, dy)$ before bicubic grid-sampling.
   - Annihilates high-frequency surface wrinkling while preserving exact subpixel boundary anchors.

5. **Organic Overscan & Alpha-Clipping:**
   - Render with dynamic 40px bounding-box overscan.
   - Composite using a 7px morphologically dilated and Gaussian-feathered roto mask ($\sigma = 1.8$).

---

## Proven VFX Failure Modes & Solutions

| Flaw / Artifact | Root Cause | Solution |
| :--- | :--- | :--- |
| **Faceted triangle lines / straight cuts** | Delaunay triangulation is $C^0$ piecewise affine. | Replace Delaunay with $C^1$-smooth Moving Least Squares (MLS). |
| **Corner jitter / edge shaking ($\sigma = 38\text{ px}$)** | Frame-by-frame RANSAC Homography tilts under non-coplanar points. | Discard RANSAC; use pure CoTracker points with Savitzky-Golay. |
| **Wavy ripples / Jello effect** | Steep inverse-distance singularity $w = 1/d^2$ creates local funnels. | Apply radius regularization $R_{\text{reg}} = 250.0$ and Flow Blur ($k=31$). |
| **Object dragging neighbor plane (Shear)** | 3D parallax between raised objects and flat surfaces. | Integrate Depth Map Bilateral Weighting ($\sigma_d = 25$). |
| **Centroid drift / floating** | Dense point clusters (e.g. 35 pts on pack vs 40 on paper) pull the centroid. | Apply Spatial Voxel Subsampling (25px grid) or Depth Bilateral Weighting. |
