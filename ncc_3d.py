"""
ncc_3d.py — v5
==============
3D Normalized Cross-Correlation primitives for speckle tracking.

Key functions
─────────────
ncc_3d(template, search)
    Compute the 3D NCC map between a template patch and a larger search volume.
    Returns values in [-1, 1]; higher = better match.

extract_patch_3d(vol, pos, half)
    Extract a 3D patch centred at `pos` with given half-sizes.
    Edge-pads with the border value so seeds near volume edges are handled
    without raising IndexError.

Algorithm
─────────
NCC[x,y,z] = Σ_{i,j,k} t̃[i,j,k] · (s[x+i,y+j,z+k] - μ_local) / σ_local
where t̃ = (T - μ_T) / ‖T - μ_T‖₂  (zero-mean, unit-norm template)

Cross-correlation via scipy.signal.correlate (uses FFT internally when beneficial).
Local mean/variance of the search via scipy.ndimage.uniform_filter.

Alignment note
─────────────
For a template of shape (tp, tq, tr) and search of shape (sp, sq, sr):
  valid NCC map shape = (sp-tp+1, sq-tq+1, sr-tr+1)
  NCC[px, py, pz] compares template against search[px:px+tp, py:py+tq, pz:pz+tr]
  → centre of that window at (px+tp//2, py+tq//2, pz+tr//2) in search coords
  → in original volume: seed - s_half + peak + t_half
"""

from __future__ import annotations

import numpy as np
from scipy.signal import correlate
from scipy.ndimage import uniform_filter


# ─────────────────────────────────────────────────────────
# Core NCC
# ─────────────────────────────────────────────────────────

def ncc_3d(template: np.ndarray, search: np.ndarray) -> np.ndarray:
    """
    3D normalized cross-correlation.

    Parameters
    ----------
    template : (tp, tq, tr) array  — reference patch from frame 0
    search   : (sp, sq, sr) array  — search volume from frame t
               must satisfy sp≥tp, sq≥tq, sr≥tr

    Returns
    -------
    ncc_map  : (sp-tp+1, sq-tq+1, sr-tr+1) float32 in [-1, 1]
    """
    tp, tq, tr = template.shape
    N = float(tp * tq * tr)

    # Zero-mean, unit-norm template
    t = template.astype(np.float32)
    t -= t.mean()
    t_energy = float(np.sqrt((t ** 2).sum()))
    if t_energy < 1e-6:
        vshape = (search.shape[0]-tp+1, search.shape[1]-tq+1, search.shape[2]-tr+1)
        return np.zeros(vshape, dtype=np.float32)
    t /= t_energy

    s = search.astype(np.float32)

    # Cross-correlation — scipy selects direct vs FFT automatically
    # correlate(s, t, 'valid')[px,py,pz] = Σ s[px+i,py+j,pz+k]·t[i,j,k]
    # Because t is zero-mean: Σ t[i,j,k]·s[...] = Σ t[i,j,k]·(s[...] - μ_local)
    cc = correlate(s, t, mode='valid', method='auto')  # (sp-tp+1, sq-tq+1, sr-tr+1)

    # Local statistics of search — uniform_filter gives the local mean.
    # Output at position (i,j,k) covers the window centred at (i,j,k).
    # The valid NCC position (px,py,pz) corresponds to the window whose centre
    # is at (px + tp//2, py + tq//2, pz + tr//2) in search coords.
    ox, oy, oz = tp // 2, tq // 2, tr // 2
    sp, sq, sr = s.shape
    v_sp, v_sq, v_sr = sp - tp + 1, sq - tq + 1, sr - tr + 1

    s_mean  = uniform_filter(s,      size=(tp, tq, tr), mode='nearest')
    s_mean2 = uniform_filter(s ** 2, size=(tp, tq, tr), mode='nearest')

    lm  = s_mean [ox:ox+v_sp, oy:oy+v_sq, oz:oz+v_sr]
    lm2 = s_mean2[ox:ox+v_sp, oy:oy+v_sq, oz:oz+v_sr]

    local_var    = np.maximum(lm2 - lm ** 2, 0.0)
    local_energy = np.sqrt(local_var * N)          # ≈ ‖s_window - μ_local‖₂

    ncc_map = cc / (local_energy + 1e-7)
    return np.clip(ncc_map, -1.0, 1.0).astype(np.float32)


# ─────────────────────────────────────────────────────────
# Patch extraction
# ─────────────────────────────────────────────────────────

def extract_patch_3d(
    vol : np.ndarray,
    pos : tuple[int, int, int],
    half: tuple[int, int, int],
) -> np.ndarray:
    """
    Extract a 3D patch of size (2*hx+1, 2*hy+1, 2*hz+1) centred at pos.
    Pads with the edge ('nearest-neighbour') values when pos is near a border.

    Parameters
    ----------
    vol  : (nx, ny, nz) array
    pos  : (ix, iy, iz) centre voxel (ints)
    half : (hx, hy, hz) half-extents in each dimension

    Returns
    -------
    patch : (2*hx+1, 2*hy+1, 2*hz+1) float32
    """
    ix, iy, iz = int(pos[0]), int(pos[1]), int(pos[2])
    hx, hy, hz = int(half[0]), int(half[1]), int(half[2])
    nx, ny, nz = vol.shape

    x0, x1 = ix - hx, ix + hx + 1
    y0, y1 = iy - hy, iy + hy + 1
    z0, z1 = iz - hz, iz + hz + 1

    # Amount of padding required on each side
    px0, px1 = max(0, -x0),      max(0, x1 - nx)
    py0, py1 = max(0, -y0),      max(0, y1 - ny)
    pz0, pz1 = max(0, -z0),      max(0, z1 - nz)

    # Clamped slice indices into vol
    cx0, cx1 = max(0, x0), min(nx, x1)
    cy0, cy1 = max(0, y0), min(ny, y1)
    cz0, cz1 = max(0, z0), min(nz, z1)

    patch = vol[cx0:cx1, cy0:cy1, cz0:cz1].astype(np.float32)

    if (px0, px1, py0, py1, pz0, pz1) != (0, 0, 0, 0, 0, 0):
        patch = np.pad(patch,
                       ((px0, px1), (py0, py1), (pz0, pz1)),
                       mode='edge')
    return patch


# ─────────────────────────────────────────────────────────
# Peak quality metric (PSR)
# ─────────────────────────────────────────────────────────

def peak_to_sidelobe_ratio(ncc_map: np.ndarray, peak_half: int = 2) -> float:
    """
    Peak-to-Sidelobe Ratio of a 3D NCC map.

    PSR = (peak_val - μ_sidelobe) / (σ_sidelobe + ε)

    The sidelobe region excludes a (2*peak_half+1)³ cube around the peak.
    Higher PSR → sharper, more trustworthy NCC peak.
    """
    peak_val = float(ncc_map.max())
    peak_idx = np.array(np.unravel_index(np.argmax(ncc_map), ncc_map.shape))

    # Build sidelobe mask: everything outside the exclusion cube
    sl_mask = np.ones(ncc_map.shape, dtype=bool)
    slices = tuple(
        slice(max(0, p - peak_half), min(s, p + peak_half + 1))
        for p, s in zip(peak_idx, ncc_map.shape)
    )
    sl_mask[slices] = False

    sidelobe = ncc_map[sl_mask]
    if sidelobe.size == 0:
        return 0.0

    return float((peak_val - sidelobe.mean()) / (sidelobe.std() + 1e-7))
