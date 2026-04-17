"""
twist_3d.py — v5  (anatomical split · transverse-plane projection · time-varying centroid)
===========================================================================================
Compute cardiac twist as the rotation difference between two groups of seeds:

    twist(t)  =  θ_apex(t)  −  θ_base(t)

Scientific design choices
--------------------------
1. **Anatomical base / apex split**
   Seeds are assigned to base (basal third) or apex (apical third) of the LV
   long axis; the middle third is excluded so the two groups capture maximally
   different twist.  The LV long-axis extent is taken directly from the mask
   (z_lo … z_hi).

2. **LV long-axis projection (transverse-plane)**
   The LV long axis is estimated by PCA of the mask-centroid curve
   {(cx(z), cy(z), z)}.  All seed positions are projected onto the plane
   perpendicular to this axis (the "transverse plane") before rotation is
   computed.  When the LV is tilted relative to the acquisition z-axis this
   removes the out-of-plane contamination that would otherwise pollute the
   in-plane rotation estimate.

3. **Time-varying centroid**
   For each seed i at frame t the rotation reference point is
   mask_info.centre_at(z_i^t) — the mask centroid at the seed's CURRENT
   z-level.  This removes the translation bias that arises when the LV
   shortens and seeds migrate to different z-levels during systole.
   (The reference positions use the frame-0 z-level centroid, also projected
   onto the transverse plane, giving a consistent treatment of both ends of
   the rotation formula.)

Returns
-------
twist      : (nt,) degrees — θ_apex − θ_base, Savitzky-Golay smoothed
twist_std  : (nt,) degrees — jackknife leave-one-out std of each group
rot_base   : (nt,) degrees — base group rotation (smoothed)
rot_apex   : (nt,) degrees — apex group rotation (smoothed)
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter
from .segment_3d import MaskInfo3D

# RANSAC inlier threshold (pixels in the transverse plane)
INLIER_PX = 3.0

# Savitzky-Golay smoothing
_SAVGOL_WINDOW  = 11
_SAVGOL_POLYORD = 3


# ─────────────────────────────────────────────────────────
# Rotation fit utilities
# (now accept separate reference & current centroids)
# ─────────────────────────────────────────────────────────

def _fit_rotation(
    seeds_xy : np.ndarray,            # (N, 2) reference positions
    traj_xy  : np.ndarray,            # (N, 2) current positions
    cx_s     : np.ndarray,            # (N,)   centroid for reference positions
    cy_s     : np.ndarray,            # (N,)   centroid for reference positions
    cx_t     : np.ndarray | None = None,  # (N,) centroid for current positions
    cy_t     : np.ndarray | None = None,  # defaults to cx_s / cy_s
) -> float:
    """
    Closed-form least-squares rotation from seeds_xy → traj_xy.

    Reference positions are centred by (cx_s, cy_s); current positions by
    (cx_t, cy_t).  Using different centroids for the two frames correctly
    removes the translational component of the motion before fitting rotation.
    """
    if len(seeds_xy) < 2:
        return 0.0
    if cx_t is None: cx_t = cx_s
    if cy_t is None: cy_t = cy_s
    x = seeds_xy[:, 0] - cx_s;  y = seeds_xy[:, 1] - cy_s
    u = traj_xy [:, 0] - cx_t;  v = traj_xy [:, 1] - cy_t
    return float(np.degrees(np.arctan2(
        np.sum(x * v - y * u),
        np.sum(x * u + y * v),
    )))


def _fit_rotation_robust(
    seeds_xy  : np.ndarray,
    traj_xy   : np.ndarray,
    cx_s      : np.ndarray,
    cy_s      : np.ndarray,
    cx_t      : np.ndarray | None = None,
    cy_t      : np.ndarray | None = None,
    inlier_px : float = INLIER_PX,
) -> float:
    if len(seeds_xy) < 3:
        return _fit_rotation(seeds_xy, traj_xy, cx_s, cy_s, cx_t, cy_t)
    if cx_t is None: cx_t = cx_s
    if cy_t is None: cy_t = cy_s

    theta_rad = np.radians(_fit_rotation(seeds_xy, traj_xy, cx_s, cy_s, cx_t, cy_t))
    cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)

    x = seeds_xy[:, 0] - cx_s;  y = seeds_xy[:, 1] - cy_s
    u = traj_xy [:, 0] - cx_t;  v = traj_xy [:, 1] - cy_t

    exp_u     = cos_t * x - sin_t * y
    exp_v     = sin_t * x + cos_t * y
    residuals = np.sqrt((u - exp_u) ** 2 + (v - exp_v) ** 2)
    inliers   = residuals < inlier_px

    if inliers.sum() < 3:
        return float(np.degrees(theta_rad))
    return _fit_rotation(seeds_xy[inliers], traj_xy[inliers],
                         cx_s[inliers], cy_s[inliers],
                         cx_t[inliers], cy_t[inliers])


# ─────────────────────────────────────────────────────────
# Main twist computation
# ─────────────────────────────────────────────────────────

def compute_twist_3d(
    trajectories        : np.ndarray,    # (nt, N, 3) float32
    valid               : np.ndarray,    # (nt, N)   bool
    mask_info           : MaskInfo3D,
    inlier_px           : float = INLIER_PX,
    max_angular_dev_deg : float = 10.0,
    base_frac           : tuple  = (0.0, 1/3),   # fraction of LV length → base
    apex_frac           : tuple  = (2/3, 1.0),   # fraction of LV length → apex
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    trajectories    : (nt, N, 3) tracked positions in volume voxel coordinates
    valid           : (nt, N) bool validity mask
    mask_info       : MaskInfo3D
    base_frac       : (lo, hi) fractions of [z_lo … z_hi] that define base
    apex_frac       : (lo, hi) fractions of [z_lo … z_hi] that define apex

    Returns
    -------
    twist     : (nt,) degrees  — θ_apex − θ_base, smoothed
    twist_std : (nt,) degrees  — jackknife intra-group angular std
    rot_base  : (nt,) degrees  — base group rotation (smoothed)
    rot_apex  : (nt,) degrees  — apex group rotation (smoothed)
    """
    nt, N, _ = trajectories.shape

    # ── 1. LV long axis via PCA of mask-centroid curve ────────────────────────
    # Fit the 3D polyline {(cx(z), cy(z), z)} to find the principal direction.
    z_v   = mask_info.z_valid.astype(float)
    pts   = np.column_stack([mask_info.cx_arr[mask_info.z_valid],
                             mask_info.cy_arr[mask_info.z_valid], z_v])
    lv_center = pts.mean(axis=0)           # (3,) centroid of the centroid-curve
    _, _, Vt  = np.linalg.svd(pts - lv_center, full_matrices=False)
    long_axis = Vt[0]                      # dominant direction = LV long axis
    if long_axis[2] < 0:
        long_axis = -long_axis             # orient so z-component > 0 (base→apex)

    # Two orthonormal basis vectors spanning the transverse plane
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(long_axis, tmp)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(long_axis, tmp);  e1 /= np.linalg.norm(e1)
    e2 = np.cross(long_axis, e1);   e2 /= np.linalg.norm(e2)

    tilt_deg = float(np.degrees(np.arccos(np.clip(long_axis[2], -1.0, 1.0))))
    print(f'  [twist] LV long-axis tilt from z: {tilt_deg:.2f}°')

    # ── 2. Project all trajectories onto the transverse plane ─────────────────
    traj_f = trajectories.astype(np.float64) - lv_center   # (nt, N, 3) centred
    # (nt, N, 2) coordinates in (e1, e2) frame
    traj_t = np.stack([traj_f @ e1, traj_f @ e2], axis=-1)
    seeds_xy = traj_t[0]                   # reference positions at frame 0

    # ── 3. Time-varying per-seed mask centroids (projected) ───────────────────
    # ctr_all[t, i] = projected mask centroid at seed i's z-level at frame t.
    # Using the current z-level (not the fixed frame-0 z) removes the
    # translation bias caused by LV longitudinal shortening.
    z_all   = np.clip(trajectories[:, :, 2].astype(int),
                      0, len(mask_info.cx_arr) - 1)         # (nt, N)
    pts_ctr = np.stack([mask_info.cx_arr[z_all],
                        mask_info.cy_arr[z_all],
                        z_all.astype(float)], axis=-1) - lv_center  # (nt, N, 3)
    ctr_all = np.stack([pts_ctr @ e1, pts_ctr @ e2], axis=-1)       # (nt, N, 2)
    ctr0    = ctr_all[0]                   # reference centroids (frame 0)

    # ── 4. Anatomical base / apex split ───────────────────────────────────────
    z_lo, z_hi = mask_info.z_lo, mask_info.z_hi
    lv_len     = z_hi - z_lo
    z_base_lo  = z_lo + lv_len * base_frac[0]
    z_base_hi  = z_lo + lv_len * base_frac[1]
    z_apex_lo  = z_lo + lv_len * apex_frac[0]
    z_apex_hi  = z_lo + lv_len * apex_frac[1]

    seeds_z   = trajectories[0, :, 2].astype(int)
    base_mask = (seeds_z >= z_base_lo) & (seeds_z <= z_base_hi)
    apex_mask = (seeds_z >= z_apex_lo) & (seeds_z <= z_apex_hi)
    n_base, n_apex = int(base_mask.sum()), int(apex_mask.sum())

    print(f'  [twist] anatomical split:  '
          f'base z∈[{z_base_lo:.0f},{z_base_hi:.0f}] n={n_base}  |  '
          f'apex z∈[{z_apex_lo:.0f},{z_apex_hi:.0f}] n={n_apex}  '
          f'(middle third z∈[{z_base_hi:.0f},{z_apex_lo:.0f}] excluded)')
    if n_base < 2 or n_apex < 2:
        raise ValueError(
            f"Too few seeds in a group: base={n_base}, apex={n_apex}. "
            "Increase N_SEEDS or adjust base_frac / apex_frac.")

    # ── 5. Per-seed angular position in the transverse plane (for delta_deg) ──
    # Angle of each seed relative to its current-z centroid, in (e1, e2) coords.
    dx_all    = traj_t[:, :, 0] - ctr_all[:, :, 0]   # (nt, N)
    dy_all    = traj_t[:, :, 1] - ctr_all[:, :, 1]
    angles_uw = np.unwrap(np.arctan2(dy_all, dx_all), axis=0)
    delta_deg = np.degrees(angles_uw - angles_uw[0:1, :])

    # ── 6. Outlier-frame interpolation helper ─────────────────────────────────
    def _outlier_interp(arr: np.ndarray,
                        window  : int   = 5,
                        n_sigma : float = 2.5) -> np.ndarray:
        out  = arr.copy()
        n    = len(out)
        if n <= window:
            return out
        half     = window // 2
        baseline = np.array([np.median(out[max(0, i - half): min(n, i + half + 1)])
                             for i in range(n)])
        resid    = out - baseline
        mad      = np.median(np.abs(resid))
        if mad < 1e-9:
            return out
        flag     = np.abs(resid) > n_sigma * 1.4826 * mad
        flag[0]  = False
        if flag.any():
            good, bad = np.where(~flag)[0], np.where(flag)[0]
            out[bad] = np.interp(bad.astype(float), good.astype(float), out[good])
        return out

    # ── 7. Group rotation fitting ─────────────────────────────────────────────
    def _fit_group_rotation(group_mask: np.ndarray) -> np.ndarray:
        rot = np.zeros(nt, dtype=np.float64)
        for t in range(1, nt):
            v = valid[t] & group_mask
            if v.sum() < 2:
                rot[t] = rot[t - 1]
                continue
            # Mild pre-filter: exclude seeds with implausibly large angular drift
            ok = (np.abs(delta_deg[t]) <= max_angular_dev_deg) & v
            if ok.sum() >= 2:
                v = ok
            rot[t] = _fit_rotation_robust(
                seeds_xy[v],       traj_t[t, v],
                ctr0[v, 0],        ctr0[v, 1],      # reference centroid (frame-0 z)
                ctr_all[t, v, 0],  ctr_all[t, v, 1],  # current centroid (frame-t z)
                inlier_px=inlier_px,
            )
        return np.degrees(np.unwrap(np.radians(rot)))

    rot_base_raw = _outlier_interp(_fit_group_rotation(base_mask))
    rot_apex_raw = _outlier_interp(_fit_group_rotation(apex_mask))
    twist_raw    = rot_apex_raw - rot_base_raw

    # ── 8. Savitzky-Golay smoothing ───────────────────────────────────────────
    wl = min(_SAVGOL_WINDOW, nt)
    if wl % 2 == 0:
        wl -= 1

    def _smooth(arr: np.ndarray) -> np.ndarray:
        if wl >= _SAVGOL_POLYORD + 1 and nt > wl:
            out = savgol_filter(arr, wl, _SAVGOL_POLYORD)
            out[0] = 0.0
            return out
        return arr.copy()

    rot_base = _smooth(rot_base_raw)
    rot_apex = _smooth(rot_apex_raw)
    twist    = _smooth(twist_raw)

    # ── 9. twist_std: jackknife leave-one-out ─────────────────────────────────
    twist_std = np.zeros(nt, dtype=np.float64)
    for t in range(1, nt):
        vars_ = []
        for grp in (base_mask, apex_mask):
            v    = valid[t] & grp
            vidx = np.where(v)[0]
            if len(vidx) < 3:
                continue
            loo_rots = []
            for k in range(len(vidx)):
                loo = np.ones(len(vidx), dtype=bool); loo[k] = False
                sel = vidx[loo]
                loo_rots.append(_fit_rotation(
                    seeds_xy[sel],      traj_t[t, sel],
                    ctr0[sel, 0],       ctr0[sel, 1],
                    ctr_all[t, sel, 0], ctr_all[t, sel, 1],
                ))
            vars_.append(float(np.std(loo_rots)))
        if len(vars_) == 2:
            twist_std[t] = np.sqrt(vars_[0] ** 2 + vars_[1] ** 2)
        elif len(vars_) == 1:
            twist_std[t] = vars_[0]

    return twist, twist_std, rot_base, rot_apex
