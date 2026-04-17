"""
select_seeds_3d.py — v5 (global 3D)
=====================================
Select N meaningful speckle seeds directly from the **full 3D myocardial
volume**, without constraining seeds to individual z-slices.

Strategy
────────
1. Collect all mask voxels in the full 3D volume (with boundary padding).
2. Score each candidate by brightness (local mean) + texture (local std).
   Optionally augment with PSR from a one-step NCC check (frame 0→1).
3. Distribute seeds spatially via a z-level × angular-sector grid:
     n_z_levels bands along the LV long axis  ×
     n_sectors  angular sectors around the LV centroid
   Within each (band, sector) cell the highest-scored candidate that
   satisfies a 3D minimum-distance NMS is selected.
4. A global greedy pass fills any remaining slots up to n_seeds.

Returns
───────
seeds : (K, 3) int  — (ix, iy, iz) positions in volume coordinates,
        K ≤ n_seeds, distributed throughout the 3D myocardium.
"""

from __future__ import annotations

import numpy as np
from .ncc_3d import extract_patch_3d, ncc_3d, peak_to_sidelobe_ratio
from .segment_3d import MaskInfo3D

# Half-size of the scoring neighbourhood (voxels) — identical to template size
_SCORE_HALF = (4, 4, 2)   # → 9×9×5 voxel patch, ~7.6×7.6×8.2 mm


# ─────────────────────────────────────────────────────────
# Candidate scoring
# ─────────────────────────────────────────────────────────

def _score_candidates(
    vol0       : np.ndarray,
    candidates : np.ndarray,          # (M, 3) int — (ix, iy, iz)
    score_half : tuple[int, int, int] = _SCORE_HALF,
) -> np.ndarray:
    """
    Score = center voxel brightness.

    We look for bright speckle points: high intensity at the center voxel,
    ideally with intensity falling off toward the edges (local maximum).
    Spatial NMS (min_dist_mm) ensures the selected seeds are spread out.
    """
    nx, ny, nz = vol0.shape
    scores = np.zeros(len(candidates), dtype=np.float64)
    for k, (ix, iy, iz) in enumerate(candidates):
        scores[k] = float(vol0[
            int(np.clip(ix, 0, nx - 1)),
            int(np.clip(iy, 0, ny - 1)),
            int(np.clip(iz, 0, nz - 1)),
        ]) / 255.0
    return scores


def _psr_scores(
    vol0       : np.ndarray,
    vol1       : np.ndarray,
    candidates : np.ndarray,
    t_half     : tuple[int, int, int] = _SCORE_HALF,
    s_half     : tuple[int, int, int] = (10, 10, 5),
) -> np.ndarray:
    """
    PSR from a one-step 3D NCC (frame 0→1) for each candidate.
    High PSR means the match peak is sharp and unambiguous.
    """
    psrs = np.zeros(len(candidates), dtype=np.float64)
    for k, (ix, iy, iz) in enumerate(candidates):
        tmpl   = extract_patch_3d(vol0, (ix, iy, iz), t_half)
        search = extract_patch_3d(vol1, (ix, iy, iz), s_half)
        ncc    = ncc_3d(tmpl, search)
        psrs[k] = peak_to_sidelobe_ratio(ncc)
    return psrs


# ─────────────────────────────────────────────────────────
# Main: global 3D seed selection
# ─────────────────────────────────────────────────────────

def select_seeds_3d_global(
    vol0        : np.ndarray,           # (nx, ny, nz) uint8 — frame-0 volume
    mask_info   : MaskInfo3D,
    scale       : np.ndarray,           # [dx, dy, dz] mm/voxel
    n_seeds     : int   = 30,
    min_dist_mm : float = 12.0,         # minimum 3D seed spacing (mm)
    vol1        : np.ndarray | None = None,
    w_psr       : float = 0.25,
    n_z_levels  : int   = 5,            # z-bands for spatial distribution
    n_sectors   : int   = 8,            # angular sectors per z-band
    z_lo_limit  : int | None = None,    # restrict candidates to z >= z_lo_limit
    z_hi_limit  : int | None = None,    # restrict candidates to z <= z_hi_limit
) -> np.ndarray:
    """
    Select `n_seeds` meaningful 3D speckle seeds from the full myocardial mask.

    Parameters
    ----------
    vol0        : (nx, ny, nz) uint8 frame-0 volume
    mask_info   : MaskInfo3D from segment_3d.load_mask_3d()
    scale       : [dx, dy, dz] mm/voxel (handles anisotropic z-spacing)
    n_seeds     : total number of seeds to return
    min_dist_mm : minimum 3D Euclidean distance between any two seeds (mm)
    vol1        : (nx, ny, nz) uint8 frame-1 volume for PSR scoring; None=skip
    w_psr       : weight of PSR term when vol1 is provided
    n_z_levels  : number of z-depth bands for spatial coverage
    n_sectors   : number of angular sectors per z-band

    Returns
    -------
    seeds : (K, 3) int  — (ix, iy, iz), K ≤ n_seeds
    """
    nx, ny, nz = vol0.shape
    hx, hy, hz = _SCORE_HALF

    # ── 1. All valid mask voxels with enough boundary clearance for patches
    ixs, iys, izs = np.where(mask_info.mask_3d)
    keep = (
        (ixs >= hx) & (ixs < nx - hx) &
        (iys >= hy) & (iys < ny - hy) &
        (izs >= hz) & (izs < nz - hz)
    )
    if z_lo_limit is not None:
        keep &= (izs >= z_lo_limit)
    if z_hi_limit is not None:
        keep &= (izs <= z_hi_limit)
    ixs, iys, izs = ixs[keep], iys[keep], izs[keep]

    if len(ixs) == 0:
        return np.empty((0, 3), dtype=int)

    candidates = np.column_stack([ixs, iys, izs])

    # ── 2. Composite brightness+texture score (+ optional PSR)
    bt_scores = _score_candidates(vol0, candidates)
    if vol1 is not None:
        psr = _psr_scores(vol0, vol1, candidates)
        psr_max = psr.max()
        if psr_max > 1e-6:
            psr = psr / psr_max
        scores = (1.0 - w_psr) * bt_scores + w_psr * psr
    else:
        scores = bt_scores

    # ── 3. Per-candidate angular position around LV centroid at that z
    cx_cand = np.array([mask_info.cx_arr[int(np.clip(iz, 0, nz-1))] for iz in izs])
    cy_cand = np.array([mask_info.cy_arr[int(np.clip(iz, 0, nz-1))] for iz in izs])
    angles  = np.arctan2(iys - cy_cand, ixs - cx_cand)   # ∈ (-π, π]

    # ── 4. z-level × angular-sector grid
    z_lo = int(izs.min());  z_hi = int(izs.max())
    z_edges     = np.linspace(z_lo, z_hi + 1, n_z_levels + 1)
    sector_size = 2.0 * np.pi / n_sectors

    selected: list[np.ndarray] = []

    # Pass A — best NMS-passing seed per (z-band, sector) cell
    for zi in range(n_z_levels):
        if len(selected) >= n_seeds:
            break
        in_band = (izs >= z_edges[zi]) & (izs < z_edges[zi + 1])
        for s in range(n_sectors):
            if len(selected) >= n_seeds:
                break
            a_lo = -np.pi + s * sector_size
            a_hi = a_lo + sector_size
            in_cell = in_band & (angles >= a_lo) & (angles < a_hi)
            if not in_cell.any():
                continue
            # Greedy: pick highest-scored candidate that passes 3D NMS
            order = np.where(in_cell)[0]
            order = order[np.argsort(scores[order])[::-1]]
            for idx in order:
                if _far_enough_3d(candidates[idx].astype(float),
                                  selected, min_dist_mm, scale):
                    selected.append(candidates[idx].copy())
                    break

    # Pass B — global greedy fill if grid didn't provide enough seeds
    if len(selected) < n_seeds:
        order = np.argsort(scores)[::-1]
        for idx in order:
            if len(selected) >= n_seeds:
                break
            if _far_enough_3d(candidates[idx].astype(float),
                               selected, min_dist_mm, scale):
                selected.append(candidates[idx].copy())

    if not selected:
        return np.empty((0, 3), dtype=int)
    return np.array(selected, dtype=int)


# ─────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────

def _far_enough_3d(
    pos         : np.ndarray,
    selected    : list[np.ndarray],
    min_dist_mm : float,
    scale       : np.ndarray,
) -> bool:
    """
    True if `pos` is at least `min_dist_mm` away (in physical mm) from every
    seed in `selected`.  Uses full 3D Euclidean distance — correctly accounts
    for anisotropic z-spacing (dz ≈ 1.65 mm vs dx/dy ≈ 0.84 mm).
    """
    if not selected:
        return True
    pos_mm = pos * scale[:3]
    for s in selected:
        if np.linalg.norm(pos_mm - s.astype(float) * scale[:3]) < min_dist_mm:
            return False
    return True
