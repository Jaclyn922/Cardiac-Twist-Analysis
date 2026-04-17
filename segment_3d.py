"""
segment_3d.py — v5
==================
Load the 3D Slicer manual segmentation (Segmentation.seg.nrrd) and compute
the LV centreline parameters needed for:
  • 3D seed placement and ring-validity checking
  • Twist computation relative to each tracked seed's current z-position

Outputs a MaskInfo3D dataclass:
  mask_3d  : (nx, ny, nz) bool   — myocardial voxel mask
  cx_arr   : (nz,) float         — LV centre in the ix direction, per z-slice
  cy_arr   : (nz,) float         — LV centre in the iy direction, per z-slice
  ri_arr   : (nz,) float         — inner radius (10th pct of distances), per z-slice
  ro_arr   : (nz,) float         — outer radius (90th pct of distances), per z-slice
  z_valid  : 1-D int array       — z-indices that have non-zero mask voxels
  z_lo/z_hi: int                 — min/max of z_valid

Coordinate convention (same as vol[ix, iy, iz, t]):
  ix ∈ [0, nx-1],  iy ∈ [0, ny-1],  iz ∈ [0, nz-1]
"""

from __future__ import annotations

import numpy as np
import nrrd
from dataclasses import dataclass
from scipy.ndimage import label as _cc_label


# ─────────────────────────────────────────────────────────
# Data container
# ─────────────────────────────────────────────────────────

@dataclass
class MaskInfo3D:
    mask_3d : np.ndarray   # (nx, ny, nz) bool
    cx_arr  : np.ndarray   # (nz,) float  – LV cx per z-slice (ix-coord)
    cy_arr  : np.ndarray   # (nz,) float  – LV cy per z-slice (iy-coord)
    ri_arr  : np.ndarray   # (nz,) float  – inner radius per z-slice
    ro_arr  : np.ndarray   # (nz,) float  – outer radius per z-slice
    z_valid : np.ndarray   # sorted int array of z-slices with mask
    z_lo    : int
    z_hi    : int

    def centre_at(self, iz: float) -> tuple[float, float]:
        """Return (cx, cy) at the given z-position (clamped to valid range)."""
        iz_c = int(np.clip(round(iz), 0, len(self.cx_arr) - 1))
        return float(self.cx_arr[iz_c]), float(self.cy_arr[iz_c])

    def radii_at(self, iz: float) -> tuple[float, float]:
        """Return (r_inner, r_outer) at the given z-position."""
        iz_c = int(np.clip(round(iz), 0, len(self.ri_arr) - 1))
        return float(self.ri_arr[iz_c]), float(self.ro_arr[iz_c])


# ─────────────────────────────────────────────────────────
# Loader
# ─────────────────────────────────────────────────────────

def load_mask_3d(nrrd_path: str) -> MaskInfo3D:
    """
    Load the NRRD segmentation and compute per-slice LV ring parameters.

    The NRRD file is expected to be a binary mask with shape (nx, ny, nz),
    matching the vol[ix, iy, iz, t] coordinate system.

    For z-slices without mask voxels, cx/cy/ri/ro are interpolated from
    neighbouring slices so that validity checks remain well-defined when a
    3D-tracked seed drifts into an un-segmented slice.
    """
    seg_data, _ = nrrd.read(nrrd_path)    # (nx, ny, nz) uint8
    nx, ny, nz  = seg_data.shape
    mask_3d     = seg_data.astype(bool)

    # ── Keep only the largest 3-D connected component ─────────────────────────
    labeled, n_cc = _cc_label(mask_3d)
    if n_cc > 1:
        sizes   = np.bincount(labeled.ravel())
        sizes[0] = 0                          # ignore background
        largest  = int(np.argmax(sizes))
        mask_3d  = (labeled == largest)
        print(f"  kept largest CC : {sizes[largest]} voxels  "
              f"(discarded {n_cc-1} smaller component(s), "
              f"{int(sizes[1:].sum()) - sizes[largest]} voxels total)")

    cx_arr = np.full(nz, np.nan, dtype=np.float64)
    cy_arr = np.full(nz, np.nan, dtype=np.float64)
    ri_arr = np.full(nz, np.nan, dtype=np.float64)
    ro_arr = np.full(nz, np.nan, dtype=np.float64)
    z_valid_list: list[int] = []

    for z in range(nz):
        ixs, iys = np.where(mask_3d[:, :, z])   # positions in (ix, iy) space
        if len(ixs) == 0:
            continue

        cx    = float(ixs.mean())
        cy    = float(iys.mean())
        dists = np.sqrt((ixs - cx) ** 2 + (iys - cy) ** 2)
        ri    = float(np.percentile(dists, 10))
        ro    = float(np.percentile(dists, 90))

        cx_arr[z] = cx
        cy_arr[z] = cy
        ri_arr[z] = ri
        ro_arr[z] = ro
        z_valid_list.append(z)

    if len(z_valid_list) == 0:
        raise ValueError(f"No mask voxels found in {nrrd_path}")

    z_valid = np.array(sorted(z_valid_list), dtype=int)
    z_lo    = int(z_valid.min())
    z_hi    = int(z_valid.max())

    # ── interpolate NaN slices so that out-of-mask z-positions are still
    #    well-defined (linear interp between the nearest valid slices)
    for arr in (cx_arr, cy_arr, ri_arr, ro_arr):
        nan_mask = np.isnan(arr)
        if nan_mask.any() and (~nan_mask).any():
            good_z = np.where(~nan_mask)[0]
            arr[nan_mask] = np.interp(np.where(nan_mask)[0], good_z, arr[good_z])

    print(f"[segment_3d] loaded {nrrd_path}")
    print(f"  seg shape  : {seg_data.shape}")
    print(f"  z_valid    : {z_lo} – {z_hi}  ({len(z_valid)} slices)")
    print(f"  cx range   : {np.nanmin(cx_arr):.1f} – {np.nanmax(cx_arr):.1f}")
    print(f"  cy range   : {np.nanmin(cy_arr):.1f} – {np.nanmax(cy_arr):.1f}")
    print(f"  r_inner    : {np.nanmin(ri_arr):.1f} – {np.nanmax(ri_arr):.1f}")
    print(f"  r_outer    : {np.nanmin(ro_arr):.1f} – {np.nanmax(ro_arr):.1f}")

    return MaskInfo3D(
        mask_3d=mask_3d,
        cx_arr=cx_arr, cy_arr=cy_arr,
        ri_arr=ri_arr, ro_arr=ro_arr,
        z_valid=z_valid, z_lo=z_lo, z_hi=z_hi,
    )
