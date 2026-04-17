"""
track_3d.py — v5
================
3D speckle tracking using Direct NCC (templates anchored to frame 0).

Why Direct (not Sequential)?
─────────────────────────────
Sequential tracking uses the previous-frame patch as template, so errors
accumulate over the 24-frame cycle (drift).  Direct tracking always compares
against the frame-0 template — no accumulation.  The trade-off is that the
template may decorrelate in the late systolic frames, but the NCC peak value
(`ncc_peaks`) warns you when this happens, and the validity filter removes
those frames from the twist fit.

3D advantage over 2D slice-by-slice
────────────────────────────────────
The template is a 3D cube (default 9×9×5 voxels, ~7.6×7.6×8.2 mm).
When a speckle moves along the LV long axis (z-direction), the 2D approach
sees it disappear from its original slice.  The 3D search volume
(default 31×31×17, search range ±12.6×12.6×±13.2 mm) captures that z-motion
and returns the true 3D displaced position.

Template / search sizes (voxels)
─────────────────────────────────
Voxel spacing:  dx≈dy≈0.84 mm,  dz≈1.65 mm  (anisotropic!)
Template half:  (4, 4, 2) → 9×9×5 px   ≈ 7.6×7.6×8.2 mm
Search half:    (15, 15, 8) → 31×31×17 px  (search range ±12.6×12.6×13.2 mm)
These defaults give roughly isotropic physical extents.

Validity filter
───────────────
A tracked position is marked invalid at frame t if:
  • NCC peak < ncc_thresh  (poor match quality)
  • 2D radial distance from LV centroid falls outside
    [r_inner - tol, r_outer + tol] at the seed's current z-position

Invalid seeds are excluded from the rotation fit in twist_3d.py.
"""

from __future__ import annotations

import numpy as np
from .ncc_3d import ncc_3d, extract_patch_3d
from .segment_3d import MaskInfo3D

# ─────────────────────────────────────────────────────────
# Default parameters
# ─────────────────────────────────────────────────────────
TEMPLATE_HALF = (4, 4, 2)    # (hx, hy, hz) → 9×9×5 voxel template
SEARCH_HALF   = (15, 15, 8)  # (hx, hy, hz) → 31×31×17 voxel search volume
NCC_THRESH    = 0.25          # minimum peak NCC value for valid track
RING_TOL_PX   = 5.0           # relaxed ring tolerance (voxels)


# ─────────────────────────────────────────────────────────
# Main tracking function
# ─────────────────────────────────────────────────────────

def track_3d_direct(
    vol4d    : np.ndarray,                       # (nx, ny, nz, nt) uint8
    seeds    : np.ndarray,                       # (N, 3) int  (ix, iy, iz)
    t_half   : tuple[int,int,int] = TEMPLATE_HALF,
    s_half   : tuple[int,int,int] = SEARCH_HALF,
    verbose  : bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Track each seed through all frames using 3D Direct NCC.

    Templates are extracted from frame 0 and kept fixed — no drift.

    Parameters
    ----------
    vol4d  : (nx, ny, nz, nt) uint8 4-D volume
    seeds  : (N, 3) int array of (ix, iy, iz) starting positions (frame 0)
    t_half : template half-sizes in (x, y, z)
    s_half : search half-sizes in (x, y, z)
    verbose: print per-frame progress

    Returns
    -------
    trajectories : (nt, N, 3) float32  — (ix, iy, iz) at each frame
    ncc_peaks    : (nt, N)    float32  — peak NCC value at each frame
                  (frame 0 is always 1.0)
    """
    nx, ny, nz, nt = vol4d.shape
    N = len(seeds)

    trajectories = np.zeros((nt, N, 3), dtype=np.float32)
    ncc_peaks    = np.zeros((nt, N),    dtype=np.float32)

    trajectories[0] = seeds.astype(np.float32)
    ncc_peaks[0]    = 1.0

    vol0 = vol4d[..., 0]

    # Pre-extract and keep frame-0 templates (fixed reference)
    templates = []
    for i, seed in enumerate(seeds):
        tmpl = extract_patch_3d(vol0, tuple(seed), t_half)
        templates.append(tmpl)

    for t in range(1, nt):
        vol_t = vol4d[..., t]
        for i, seed in enumerate(seeds):
            search  = extract_patch_3d(vol_t, tuple(seed), s_half)
            ncc_map = ncc_3d(templates[i], search)

            peak_val = float(ncc_map.max())
            peak_idx = np.unravel_index(np.argmax(ncc_map), ncc_map.shape)

            # Convert NCC peak back to volume coordinates
            # valid NCC pos (px,py,pz) → search window at [seed-s_half+px : +tp]
            # → window centre at seed - s_half + px + t_half
            new_ix = int(seed[0]) - s_half[0] + peak_idx[0] + t_half[0]
            new_iy = int(seed[1]) - s_half[1] + peak_idx[1] + t_half[1]
            new_iz = int(seed[2]) - s_half[2] + peak_idx[2] + t_half[2]

            # Clamp to volume bounds
            new_ix = int(np.clip(new_ix, 0, nx - 1))
            new_iy = int(np.clip(new_iy, 0, ny - 1))
            new_iz = int(np.clip(new_iz, 0, nz - 1))

            trajectories[t, i] = (new_ix, new_iy, new_iz)
            ncc_peaks[t, i]    = peak_val

        if verbose:
            valid_t = (ncc_peaks[t] >= NCC_THRESH).sum()
            print(f"  frame {t:02d}/{nt-1}  valid={valid_t}/{N}  "
                  f"peak_mean={ncc_peaks[t].mean():.3f}", flush=True)

    return trajectories, ncc_peaks


# ─────────────────────────────────────────────────────────
# Validity computation
# ─────────────────────────────────────────────────────────

def compute_validity_3d(
    trajectories : np.ndarray,        # (nt, N, 3)
    ncc_peaks    : np.ndarray,        # (nt, N)
    mask_info    : MaskInfo3D,
    ncc_thresh   : float = NCC_THRESH,
    ring_tol_px  : float = RING_TOL_PX,
) -> np.ndarray:
    """
    Compute per-seed per-frame validity mask.

    A seed i is valid at frame t if:
      1. ncc_peaks[t,i] >= ncc_thresh  (NCC match quality)
      2. 2-D radial distance from the LV centroid (at the seed's current z)
         lies within [r_inner(iz) - ring_tol, r_outer(iz) + ring_tol]

    Returns
    -------
    valid : (nt, N) bool
    """
    nt, N, _ = trajectories.shape
    valid = np.zeros((nt, N), dtype=bool)
    valid[0] = True

    for t in range(1, nt):
        for i in range(N):
            ix, iy, iz = trajectories[t, i]

            # LV geometry at the seed's current z-position
            cx, cy = mask_info.centre_at(iz)
            ri, ro  = mask_info.radii_at(iz)

            dist    = np.sqrt((ix - cx) ** 2 + (iy - cy) ** 2)
            in_ring = (dist >= ri - ring_tol_px) and (dist <= ro + ring_tol_px)
            good    = bool(ncc_peaks[t, i] >= ncc_thresh)

            valid[t, i] = in_ring and good

    return valid
