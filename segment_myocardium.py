"""
Myocardial wall segmentation for the raw z=104 2D+t echo slice.

Method:
  1. High threshold (90th pct) → find brightest connected component
  2. Centroid of CC → ring centre
  3. Distance of every CC pixel to centre → r_inner (10th pct), r_outer (90th pct)
  4. Ring mask = filled_circle(r_outer) - filled_circle(r_inner)
  5. Save masks as NIfTI (nx, ny, 1, nt)  +  QC figure
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from scipy import ndimage
from skimage.morphology import closing, disk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def filled_circle(shape, cx, cy, radius):
    H, W = shape
    yy, xx = np.ogrid[:H, :W]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2


def segment_frame(frame_hw, thresh_pct=90, r_inner_pct=10, r_outer_pct=90):
    """
    1. Threshold at thresh_pct → brightest connected component (myocardial wall)
    2. Centroid of CC = ring centre
    3. Distance histogram → r_inner, r_outer
    4. Ring mask = epi circle - endo circle

    Returns: myo_mask, epi_mask, endo_mask, cx, cy, r_inner, r_outer
    """
    H, W = frame_hw.shape

    # ── Step 1: bright threshold + largest CC ──
    thresh = np.percentile(frame_hw, thresh_pct)
    bright = frame_hw >= thresh
    bright = closing(bright, disk(2))

    labeled, n = ndimage.label(bright)
    if n == 0:
        cx, cy = W / 2, H / 2
        epi  = filled_circle((H, W), cx, cy, 50)
        endo = filled_circle((H, W), cx, cy, 20)
        return epi & ~endo, epi, endo, cx, cy, 20.0, 50.0

    sizes = ndimage.sum(bright, labeled, range(1, n + 1))
    cc = labeled == (np.argmax(sizes) + 1)

    # ── Step 2: keep top 50% of CC rows → clean myocardial pixels ──────────
    rows_all, cols_all = np.where(cc)
    row_thresh = rows_all.min() + 0.50 * (rows_all.max() - rows_all.min())
    mask50 = rows_all <= row_thresh
    rows50 = rows_all[mask50].astype(float)
    cols50 = cols_all[mask50].astype(float)

    # ── Step 3: algebraic circle fit on top-50% pixels ───────────────────
    # (x-cx)^2 + (y-cy)^2 = R^2
    # → x^2+y^2 = 2cx·x + 2cy·y + (R^2−cx^2−cy^2)
    # linear system: [x, y, 1] · [a, b, c]' = x^2+y^2
    A = np.column_stack([cols50, rows50, np.ones(len(cols50))])
    z = cols50 ** 2 + rows50 ** 2
    params, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
    cx = float(params[0] / 2.0)
    cy = float(params[1] / 2.0)

    # ── Step 4: distances from fitted centre → inner/outer radius ─────────
    dists50 = np.sqrt((cols50 - cx) ** 2 + (rows50 - cy) ** 2)
    r_inner = float(np.percentile(dists50, r_inner_pct))
    r_outer = float(np.percentile(dists50, r_outer_pct))

    # ── Step 4: ring mask ─────────────────────
    epi_mask  = filled_circle((H, W), cx, cy, r_outer)
    endo_mask = filled_circle((H, W), cx, cy, r_inner)
    myo_mask  = epi_mask & ~endo_mask

    return myo_mask, epi_mask, endo_mask, cx, cy, r_inner, r_outer


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    RAW_NII  = Path("slice_z104_raw.nii.gz")
    OUT_MASK = Path("slice_z104_mask.nii.gz")
    QC_PNG   = Path("segmentation_qc.png")

    nii  = nib.load(str(RAW_NII))
    data = np.asarray(nii.dataobj)      # (nx, ny, 1, nt)
    nx, ny, _, nt = data.shape
    frames = [data[:, :, 0, t].T.astype(np.uint8) for t in range(nt)]  # (H=ny, W=nx)

    masks = np.zeros((nx, ny, 1, nt), dtype=np.uint8)

    # ── reference from frame 0 ───────────────────────────────────────────────
    _, _, _, cx0, cy0, r_inner0, r_outer0 = segment_frame(frames[0])
    R_TOL   = 3   # pixels — max radii deviation from reference
    CXY_TOL = 10  # pixels — max centre deviation from reference
    print(f"  Reference (t=00): cx={cx0:.1f} cy={cy0:.1f}  "
          f"r_inner={r_inner0:.1f}  r_outer={r_outer0:.1f}")

    H, W = frames[0].shape
    for t, frame in enumerate(frames):
        _, _, _, cx, cy, r_inner, r_outer = segment_frame(frame)

        # clamp centre to stay within ±CXY_TOL of reference
        cx = float(np.clip(cx, cx0 - CXY_TOL, cx0 + CXY_TOL))
        cy = float(np.clip(cy, cy0 - CXY_TOL, cy0 + CXY_TOL))
        # clamp radii to stay within ±R_TOL of reference
        r_inner = float(np.clip(r_inner, r_inner0 - R_TOL, r_inner0 + R_TOL))
        r_outer = float(np.clip(r_outer, r_outer0 - R_TOL, r_outer0 + R_TOL))

        epi  = filled_circle((H, W), cx, cy, r_outer)
        endo = filled_circle((H, W), cx, cy, r_inner)
        myo  = epi & ~endo

        masks[:, :, 0, t] = myo.T.astype(np.uint8)
        print(f"  frame {t+1:02d}/{nt}  centre=({cx:.1f},{cy:.1f})  "
              f"r_inner={r_inner:.1f}  r_outer={r_outer:.1f}  myo_px={myo.sum()}")

    # save NIfTI
    mask_img = nib.Nifti1Image(masks, affine=nii.affine, header=nii.header)
    nib.save(mask_img, str(OUT_MASK))
    print(f"Mask saved → {OUT_MASK}")

    # QC figure: 3 rows × all frames
    show_frames = list(range(nt))
    n_show = len(show_frames)
    fig, axes = plt.subplots(3, n_show, figsize=(2.5 * n_show, 8))

    for i, t in enumerate(show_frames):
        frame = frames[t]
        myo   = masks[:, :, 0, t].T
        _, _, _, cx, cy, r_inner, r_outer = segment_frame(frame)
        cx = float(np.clip(cx, cx0 - CXY_TOL, cx0 + CXY_TOL))
        cy = float(np.clip(cy, cy0 - CXY_TOL, cy0 + CXY_TOL))
        r_inner = float(np.clip(r_inner, r_inner0 - R_TOL, r_inner0 + R_TOL))
        r_outer = float(np.clip(r_outer, r_outer0 - R_TOL, r_outer0 + R_TOL))
        epi  = filled_circle((frame.shape[0], frame.shape[1]), cx, cy, r_outer)
        endo = filled_circle((frame.shape[0], frame.shape[1]), cx, cy, r_inner)

        # row 0: raw
        axes[0, i].imshow(frame, cmap="gray", origin="upper", vmin=0, vmax=255)
        axes[0, i].set_title(f"t={t}", fontsize=9)
        axes[0, i].axis("off")

        # row 1: epi (yellow) + endo (cyan) contours
        axes[1, i].imshow(frame, cmap="gray", origin="upper", vmin=0, vmax=255)
        axes[1, i].contour(epi,  levels=[0.5], colors="yellow", linewidths=1.2)
        axes[1, i].contour(endo, levels=[0.5], colors="cyan",   linewidths=1.2)
        axes[1, i].plot(cx, cy, "r+", markersize=8)
        axes[1, i].axis("off")

        # row 2: myo overlay
        overlay = np.zeros((*frame.shape, 4), dtype=np.uint8)
        overlay[myo > 0] = [0, 200, 0, 130]
        axes[2, i].imshow(frame, cmap="gray", origin="upper", vmin=0, vmax=255)
        axes[2, i].imshow(overlay, origin="upper")
        axes[2, i].axis("off")

    for ax, lbl in zip(axes[:, 0], ["Raw", "Epi (yellow) / Endo (cyan)", "Myo mask"]):
        ax.set_ylabel(lbl, fontsize=8)

    fig.suptitle("Myocardial segmentation QC  —  z=104 raw slice", fontsize=11)
    plt.tight_layout()
    fig.savefig(str(QC_PNG), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"QC figure → {QC_PNG}")


if __name__ == "__main__":
    main()
