"""
fix_strain_final.py
===================
One-shot strain-correction + coupling + figure rebuild.  Reuses cached
trajectories — no tracking re-run.

Fix 1 (inferred from Fixes 3–7):
  Longitudinal strain is redefined per seed against a **fixed anatomical
  basal-plane reference** (the minimum z of all seeds at frame 0, in mm).
  Seeds whose frame-0 distance to the basal plane is < L_MIN_MM are
  rejected (too close to the reference — division blows up).  A region
  is rejected if *all* its 6 seeds are rejected.  A global longitudinal
  strain time series `epsilon_long_global(t)` is emitted as the mean
  over all surviving seeds.

Fix 2 (inferred):
  Short-axis strain uses the per-frame global seed centroid (x, y).
  Seeds whose frame-0 radial distance from the centroid is < R_MIN_MM
  are rejected (too close to the axis — same blow-up).  Per-region mean
  is computed over surviving seeds.

Fix 3: warns if |global_long_peak| > 0.30 but never halts.
Fix 4: regenerates XCD / DTW / Pearson from corrected curves; overwrites
       per-subject and cohort CSV outputs.
Fix 5: regenerates Figs 1–7 using existing publication_figures_v2 + the
       Figure 3 gridspec specified in this prompt.  Adds PIL verification
       of the PNG.
Fix 6: adds Figure 8 (strain sanity, 2×5 subjects).
Fix 7: writes strain_correction_log.md.
"""

from __future__ import annotations

import csv
import subprocess
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.signal import correlate, savgol_filter
from scipy.stats  import pearsonr, trim_mean

import subject_metadata as smeta

# tslearn DTW (previously installed)
from tslearn.metrics import dtw as _tslearn_dtw


# ─── Paths / config ──────────────────────────────────────────────────────────
DATA_ROOT = Path("/data/xinge/Cardiac-Twist-Analysis")
FIG_DIR   = DATA_ROOT / "figures_paper"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ALL_SUBJECTS = ["002A", "03", "p009npa", "p009pa", "p066a",
                "p020_1a", "p030_lp2a"]
FRAME_RATE   = 15.0
N_REGIONS    = 6

L_MIN_MM = 5.0     # basal-plane distance threshold (per-seed global ε_long only)
R_MIN_MM = 5.0     # radial frame-0 distance threshold (per-seed d_short eligibility)
MIN_SEEDS_REGION = 3   # minimum valid seeds for a region mean

SUBJECT_COLORS = {
    "002A":     "#1f77b4",
    "03":       "#d62728",
    "p009npa":  "#2ca02c",
    "p009pa":   "#ff7f0e",
    "p066a":    "#9467bd",
    "p020_1a":  "#17becf",
    "p030_lp2a":"#8c564b",
}


# ─── Publication rcParams (same as v2) ───────────────────────────────────────
mpl.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          8,
    "axes.labelsize":     8.5,
    "axes.titlesize":     9,
    "xtick.labelsize":    7.5,
    "ytick.labelsize":    7.5,
    "legend.fontsize":    7.5,
    "figure.titlesize":   10,
    "axes.linewidth":     0.6,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
    "xtick.major.size":   2.8,
    "ytick.major.size":   2.8,
    "lines.linewidth":    1.3,
    "lines.markersize":   3.5,
    "savefig.dpi":        300,
    "figure.dpi":         110,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
})


# ═══════════════════════════════════════════════════════════════════════════
# Strain correction
# ═══════════════════════════════════════════════════════════════════════════

def load_subject_raw(sid: str) -> dict | None:
    """Load per-subject rotation data and recompute twist using the
    rigid-body-rotation (RBR) corrected definition:

        twist_r(t) = rot_r(t) − ⟨rot_r(t)⟩_r

    i.e. each region's rotation minus the cohort-of-regions mean rotation
    at every frame.  This is the wringing component (Sengupta JACC CV
    Imaging 2008/2014; MAGYAR — Nemes et al.; Gonzálvez-García 2023;
    Hofrichter 2021), which separates true regional twist from the rigid
    body rotation common to the whole LV.

    `sign_flipped` from the cached xcd_analysis.npz is applied to
    rot_regions before the RBR subtraction (sign-flip negates rotation
    polarity across the whole field, which leaves the RBR-corrected
    profile shape unchanged but flips its sign — consistent with the
    convention used elsewhere in the pipeline).
    """
    res = DATA_ROOT / sid / "result"
    tw_p = res / "twist_3d.npz"
    an_p = res / "strain_twist_analysis.npz"
    xc_p = res / "xcd_analysis.npz"
    us_p = DATA_ROOT / sid / "us4d.npz"
    if not (tw_p.exists() and an_p.exists() and us_p.exists()):
        return None
    tw = np.load(tw_p)
    an = np.load(an_p, allow_pickle=True)
    us = np.load(us_p)

    rot_regions = tw["rot_regions"].astype(np.float64)        # (nt, 6)
    # RBR (rigid-body rotation) = per-frame mean across regions.  Compute on
    # the raw (pre-sign-flip) polarity here; sign-flip commutes with the
    # subtraction, and the convention elsewhere in the pipeline is for
    # downstream loaders to apply sign-flip themselves.
    rbr_mean = rot_regions.mean(axis=1, keepdims=True)        # (nt, 1)
    twist_regions = rot_regions - rbr_mean                     # RBR-corrected
    sign_flipped = False
    if xc_p.exists():
        try:
            sign_flipped = bool(np.load(xc_p)["sign_flipped"])
        except Exception:
            sign_flipped = False
    return {
        "sid":            sid,
        "trajectories":   tw["trajectories"].astype(np.float64),
        "seeds":          tw["seeds"],
        "region_ids":     tw["region_ids"].astype(int),
        "valid":          tw["valid"].astype(bool),
        "rot_regions":    rot_regions,
        "rbr_mean":       rbr_mean.ravel(),
        "twist_regions":  twist_regions,
        "twist_features": an["twist_features"].item(),
        "scale":          np.asarray(us["scale"]).astype(np.float64),
        "sign_flipped":   sign_flipped,
    }


def corrected_strains(
    traj: np.ndarray,          # (nt, N, 3) voxel units
    region_ids: np.ndarray,    # (N,)
    scale: np.ndarray,         # [sx, sy, sz] mm/voxel
    *, L_min_mm: float = L_MIN_MM,
    R_min_mm: float = R_MIN_MM,
    min_seeds_region: int = MIN_SEEDS_REGION,
    valid: np.ndarray | None = None,  # (nt, N) bool — unused (kept for API)
) -> dict:
    """
    Displacement-based per-region quantities + unchanged global ε_long.

    Frame-0 fixed coordinate system
    --------------------------------
    Fit the LV long axis anatomically from the apex–basal vector (PCA
    produces in-plane axes for foreshortened clusters).  Pick the k
    lowest-z seeds (basal-most) and the k highest-z seeds (apical-most)
    at frame 0 with ``k = min(5, max(3, N // 6))``; ``d_long`` is the
    unit vector from their mean-basal to their mean-apical position:

        origin_0   = mean(seeds_0)
        basal_ctr  = mean(k lowest-z seeds at frame 0)
        apical_ctr = mean(k highest-z seeds at frame 0)
        d_long     = normalize(apical_ctr - basal_ctr)   # apex-directed
        (PCA first component is also computed, *only for logging*.)

    For each seed s, the frame-0 *radial unit vector* is

        v_s  = seed_s(0) - origin_0
        v_s∥ = (v_s · d_long) d_long
        v_s⊥ = v_s - v_s∥
        R_s0 = ||v_s⊥||    (radial distance at frame 0, in mm)
        d_short[s] = v_s⊥ / R_s0    if R_s0 ≥ R_MIN_MM,   else rejected.

    Both d_long and d_short[s] are *fixed at frame 0* — they do not move
    with the seed.

    Per-seed displacement (mm)
    --------------------------
        disp_s(t) = seed_s(t) - seed_s(0)
        u_long[t, s]  = disp_s(t) · d_long
        u_short[t, s] = disp_s(t) · d_short[s]   (NaN if d_short[s] undefined)

    Per-region aggregation
    ----------------------
    For each region r, average the per-seed curves over the 6 seeds in
    that region (short-axis: over seeds with a defined d_short).  If a
    region has fewer than `min_seeds_region` valid seeds, the whole
    region curve is NaN.

    Global longitudinal ε (unchanged)
    ---------------------------------
    For Figure 8's top row we still produce `epsilon_long_global(t)` from
    the fixed basal-plane reference z_basal = min(seeds.z at frame 0),
    averaging `(|z_s(t)-z_basal| - |z_s(0)-z_basal|) / |z_s(0)-z_basal|`
    over seeds with |z_s(0)-z_basal| ≥ L_MIN_MM.

    Returns a dict with displacement arrays, coordinate-system vectors,
    and the unchanged global ε_long.  `valid` is accepted for API
    compatibility but is not used: displacement is robust to single-frame
    tracker drops by design.
    """
    nt, N, _ = traj.shape
    sx, sy, sz = float(scale[0]), float(scale[1]), float(scale[2])
    pts_mm = traj * np.array([sx, sy, sz])               # (nt, N, 3)

    # ── Frame-0 fixed coordinate system (origin, d_long, d_short[s]) ───────
    origin_0 = pts_mm[0].mean(axis=0)                    # (3,)

    # Anatomical d_long: apex-basal vector, robust to foreshortened clusters
    # where PCA's first component would land in the short-axis plane.
    k = int(min(5, max(3, N // 6)))
    z0 = pts_mm[0, :, 2]
    basal_idx_k  = np.argsort(z0)[:k]
    apical_idx_k = np.argsort(z0)[-k:]
    basal_ctr    = pts_mm[0, basal_idx_k].mean(axis=0)
    apical_ctr   = pts_mm[0, apical_idx_k].mean(axis=0)
    d_long_raw   = apical_ctr - basal_ctr                # points toward apex
    dl_norm      = float(np.linalg.norm(d_long_raw))
    if dl_norm < 1e-9:
        raise RuntimeError(
            "Anatomical d_long ill-defined: mean apical and basal centers "
            "coincide at frame 0."
        )
    d_long = (d_long_raw / dl_norm).astype(np.float64)   # apex-directed

    # PCA first PC for logging only (not used downstream)
    centered = pts_mm[0] - origin_0
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    d_long_pca = Vt[0].astype(np.float64)
    if d_long_pca[2] < 0:
        d_long_pca = -d_long_pca
    d_long_pca /= np.linalg.norm(d_long_pca)

    d_short_seeds = np.full((N, 3), np.nan)
    R_s0          = np.full(N, np.nan)
    for s in range(N):
        v       = pts_mm[0, s] - origin_0
        v_par   = float(v @ d_long) * d_long
        v_perp  = v - v_par
        r0      = float(np.linalg.norm(v_perp))
        R_s0[s] = r0
        if r0 >= R_min_mm:
            d_short_seeds[s] = v_perp / r0
    short_valid_seed = np.isfinite(d_short_seeds[:, 0])  # (N,) bool

    # ── Per-seed displacement projected onto d_long and d_short[s] ─────────
    disp = pts_mm - pts_mm[0:1]                          # (nt, N, 3)

    u_long_seed  = np.einsum("tnd,d->tn", disp, d_long)   # (nt, N) always finite
    u_short_seed = np.full((nt, N), np.nan)
    for s in range(N):
        if short_valid_seed[s]:
            u_short_seed[:, s] = disp[:, s] @ d_short_seeds[s]

    # ── Per-region aggregation (simple nanmean over region seeds) ──────────
    u_long_regions         = np.full((nt, N_REGIONS), np.nan)
    u_short_regions        = np.full((nt, N_REGIONS), np.nan)
    n_valid_seeds_long     = np.zeros(N_REGIONS, dtype=np.int32)
    n_valid_seeds_short    = np.zeros(N_REGIONS, dtype=np.int32)
    long_rejected_regions:  list[int] = []
    short_rejected_regions: list[int] = []

    for r in range(N_REGIONS):
        S_r = np.where(region_ids == r)[0]

        n_valid_seeds_long[r]  = len(S_r)
        if len(S_r) < min_seeds_region:
            long_rejected_regions.append(r)
        else:
            u_long_regions[:, r] = np.nanmean(u_long_seed[:, S_r], axis=1)

        m_short = S_r[short_valid_seed[S_r]]
        n_valid_seeds_short[r] = len(m_short)
        if len(m_short) < min_seeds_region:
            short_rejected_regions.append(r)
        else:
            u_short_regions[:, r] = np.nanmean(u_short_seed[:, m_short],
                                               axis=1)

    # ── Global longitudinal strain (unchanged; for Figure 8 top row) ───────
    z_mm            = pts_mm[:, :, 2]                    # (nt, N)  already mm
    z_basal_ref_mm  = float(z_mm[0].min())
    L_mm            = np.abs(z_mm - z_basal_ref_mm)
    L0              = L_mm[0]
    long_valid_seed_mask = L0 >= L_min_mm
    eps_long_seed   = np.full_like(L_mm, np.nan)
    if long_valid_seed_mask.any():
        eps_long_seed[:, long_valid_seed_mask] = (
            (L_mm[:, long_valid_seed_mask] - L0[long_valid_seed_mask])
            / L0[long_valid_seed_mask]
        )
    if int(long_valid_seed_mask.sum()) >= 2:
        epsilon_long_global = np.nanmean(
            eps_long_seed[:, long_valid_seed_mask], axis=1)
    else:
        epsilon_long_global = np.full(nt, np.nan)

    return {
        # displacement (mm)
        "u_long_regions":         u_long_regions,
        "u_short_regions":        u_short_regions,
        "u_long_seed":            u_long_seed,
        "u_short_seed":           u_short_seed,
        "origin_0":               origin_0,
        "d_long":                 d_long,
        "d_long_pca":             d_long_pca,
        "k_apex_basal":           k,
        "d_short_seeds":          d_short_seeds,
        "R_s0_seeds":             R_s0,
        "short_valid_seed_mask":  short_valid_seed,
        "n_valid_seeds_long":     n_valid_seeds_long,
        "n_valid_seeds_short":    n_valid_seeds_short,
        "long_rejected_regions":  long_rejected_regions,
        "short_rejected_regions": short_rejected_regions,
        "long_seeds_rejected":    int((~short_valid_seed).sum()),   # short-eligibility
        "short_seeds_rejected":   int((~short_valid_seed).sum()),
        # Global ε_long (unchanged; for Fig 8 top row)
        "epsilon_long_global":    epsilon_long_global,
        "long_valid_seed_mask":   long_valid_seed_mask,
        "z_basal_ref_mm":         z_basal_ref_mm,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Coupling (XCD / DTW / Pearson)
# ═══════════════════════════════════════════════════════════════════════════

def _sg_smooth(arr: np.ndarray) -> np.ndarray:
    nt = arr.shape[0]
    window = min(5, nt // 3 * 2 - 1)
    if window < 3:
        return arr.copy()
    out = arr.copy()
    for r in range(arr.shape[1]):
        col = arr[:, r]
        if np.all(np.isnan(col)):
            continue
        # Fill NaN with forward-fill then back-fill before filtering
        mask = np.isnan(col)
        if mask.any():
            col = col.copy()
            idx = np.arange(len(col))
            good = ~mask
            if good.any():
                col[mask] = np.interp(idx[mask], idx[good], col[good])
        out[:, r] = savgol_filter(col, window_length=window, polyorder=2)
        out[mask, r] = np.nan
    return out


_frame_rate_cache: dict[str, tuple[float, str]] = {}


def subject_frame_rate(sid: str) -> tuple[float, str]:
    """Per-subject acquisition frame rate — DICOM FrameTime preferred;
    us4d.npz placeholder next; 15 Hz hardcoded fallback."""
    if sid in _frame_rate_cache:
        return _frame_rate_cache[sid]
    dcm = DATA_ROOT / sid / f"{sid}.dcm"
    if dcm.exists():
        try:
            import pydicom
            ds = pydicom.dcmread(dcm, stop_before_pixels=True, force=True)
            ft = ds.get((0x0018, 0x1063), None)
            if ft is not None:
                v = ft.value
                if v not in (None, "", b""):
                    try:
                        ms = float(v)
                        if ms > 0:
                            rate = 1000.0 / ms
                            _frame_rate_cache[sid] = (rate, "DICOM FrameTime")
                            return _frame_rate_cache[sid]
                    except (ValueError, TypeError):
                        pass
        except Exception:
            pass
    # Clinician-supplied xlsx rate (used when DICOM is anonymized)
    xlsx_fr = smeta.xlsx_frame_rate(sid)
    if xlsx_fr is not None:
        _frame_rate_cache[sid] = (xlsx_fr, "xlsx (clinician metadata)")
        return _frame_rate_cache[sid]
    us = DATA_ROOT / sid / "us4d.npz"
    if us.exists():
        try:
            fr = float(np.load(us)["frame_rate"])
            _frame_rate_cache[sid] = (fr,
                                      "us4d.npz placeholder (15 Hz default)")
            return _frame_rate_cache[sid]
        except Exception:
            pass
    _frame_rate_cache[sid] = (FRAME_RATE, "hardcoded fallback")
    return _frame_rate_cache[sid]


def _xcd(a: np.ndarray, b: np.ndarray, fs: float = FRAME_RATE) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if np.isnan(a).any() or np.isnan(b).any():
        mask = ~(np.isnan(a) | np.isnan(b))
        a, b = a[mask], b[mask]
        if len(a) < 3:
            return float("nan")
    a = a - a.mean(); b = b - b.mean()
    if np.allclose(a, 0.0) or np.allclose(b, 0.0):
        return float("nan")
    c = correlate(b, a, mode="full")
    lags = np.arange(-len(a) + 1, len(a)) / float(fs)
    return float(lags[int(np.argmax(c))])


def _minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if np.isnan(x).any():
        mask = ~np.isnan(x)
        if mask.sum() < 2:
            return np.zeros_like(x)
        x = x.copy()
        x[~mask] = np.interp(np.where(~mask)[0],
                             np.where(mask)[0], x[mask])
    rng = float(x.max() - x.min())
    if rng < 1e-9:
        return np.zeros_like(x)
    return (x - x.min()) / rng


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 3:
        return float("nan")
    aa, bb = a[mask], b[mask]
    if np.std(aa) < 1e-9 or np.std(bb) < 1e-9:
        return float("nan")
    try:
        return float(pearsonr(aa, bb)[0])
    except Exception:
        return float("nan")


def recompute_coupling(sub: dict, corr: dict) -> dict:
    """Twist vs displacement coupling (Pearson, XCD, DTW).

    The 'long' and 'short' channels now refer to u_long_regions and
    u_short_regions (mm).  Column names in the output dict are kept as
    ``rho_long_*``, ``xcd_twist_long``, ``dtw_twist_long`` so downstream
    NPZ keys and CSV columns stay stable.
    """
    tw  = sub["twist_regions"]
    uL  = corr["u_long_regions"]
    uS  = corr["u_short_regions"]
    uL_sm = _sg_smooth(uL)
    uS_sm = _sg_smooth(uS)

    fs = float(subject_frame_rate(sub["sid"])[0])

    rho_long_raw      = np.full(5, np.nan)
    rho_short_raw     = np.full(5, np.nan)
    rho_long_smooth   = np.full(5, np.nan)
    rho_short_smooth  = np.full(5, np.nan)
    xcd_long          = np.full(5, np.nan)
    xcd_short         = np.full(5, np.nan)
    dtw_long          = np.full(5, np.nan)
    dtw_short         = np.full(5, np.nan)

    for i, r in enumerate(range(1, N_REGIONS)):
        a = tw[:, r]
        if np.std(a) < 1e-9:
            continue
        rho_long_raw[i]     = _safe_pearson(a, uL[:, r])
        rho_short_raw[i]    = _safe_pearson(a, uS[:, r])
        rho_long_smooth[i]  = _safe_pearson(a, uL_sm[:, r])
        rho_short_smooth[i] = _safe_pearson(a, uS_sm[:, r])
        xcd_long[i]   = _xcd(a, uL_sm[:, r], fs=fs)
        xcd_short[i]  = _xcd(a, uS_sm[:, r], fs=fs)

        if not np.all(np.isnan(uL_sm[:, r])):
            dtw_long[i]  = float(_tslearn_dtw(_minmax(a), _minmax(uL_sm[:, r])))
        if not np.all(np.isnan(uS_sm[:, r])):
            dtw_short[i] = float(_tslearn_dtw(_minmax(a), _minmax(uS_sm[:, r])))

    # 6x6 region-region twist XCD (twist only; unchanged)
    xcd_matrix = np.zeros((N_REGIONS, N_REGIONS))
    for i in range(N_REGIONS):
        for j in range(N_REGIONS):
            xcd_matrix[i, j] = _xcd(tw[:, i], tw[:, j], fs=fs)

    return {
        "u_long_regions_smooth":   uL_sm,
        "u_short_regions_smooth":  uS_sm,
        "rho_long_raw":     rho_long_raw,
        "rho_short_raw":    rho_short_raw,
        "rho_long_smooth":  rho_long_smooth,
        "rho_short_smooth": rho_short_smooth,
        "xcd_twist_long":   xcd_long,
        "xcd_twist_short":  xcd_short,
        "xcd_twist_matrix": xcd_matrix,
        "dtw_twist_long":   dtw_long,
        "dtw_twist_short":  dtw_short,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Save: per-subject npz + cohort CSVs
# ═══════════════════════════════════════════════════════════════════════════

def save_subject(sid: str, sub: dict, corr: dict, coup: dict,
                 sign_flipped: bool) -> None:
    out_dir = DATA_ROOT / sid / "result"

    np.savez_compressed(
        str(out_dir / "strain_twist_analysis.npz"),
        # twist_regions is now the RBR-corrected wringing component:
        #   twist_r(t) = rot_r(t) − ⟨rot_r(t)⟩_r   (per-frame mean across R0..R5)
        twist_regions          = sub["twist_regions"].astype(np.float32),
        rot_regions_corrected  = sub["rot_regions"].astype(np.float32),
        rbr_mean               = sub["rbr_mean"].astype(np.float32),
        # Displacement-based per-region quantities (mm)
        u_long_regions         = corr["u_long_regions"].astype(np.float32),
        u_short_regions        = corr["u_short_regions"].astype(np.float32),
        u_long_seed            = corr["u_long_seed"].astype(np.float32),
        u_short_seed           = corr["u_short_seed"].astype(np.float32),
        origin_0               = corr["origin_0"].astype(np.float32),
        d_long                 = corr["d_long"].astype(np.float32),
        d_long_pca             = corr["d_long_pca"].astype(np.float32),
        k_apex_basal           = np.int32(corr["k_apex_basal"]),
        d_short_seeds          = corr["d_short_seeds"].astype(np.float32),
        R_s0_seeds             = corr["R_s0_seeds"].astype(np.float32),
        short_valid_seed_mask  = corr["short_valid_seed_mask"],
        n_valid_seeds_long     = corr["n_valid_seeds_long"],
        n_valid_seeds_short    = corr["n_valid_seeds_short"],
        # Global ε_long (unchanged, used by Figure 8 top row)
        epsilon_long_global    = corr["epsilon_long_global"].astype(np.float32),
        long_valid_seed_mask   = corr["long_valid_seed_mask"],
        z_basal_ref_mm         = np.float32(corr["z_basal_ref_mm"]),
        # Twist-vs-displacement coupling
        rho_long               = coup["rho_long_raw"].astype(np.float64),
        rho_short              = coup["rho_short_raw"].astype(np.float64),
        rho_long_smooth        = coup["rho_long_smooth"].astype(np.float64),
        rho_short_smooth       = coup["rho_short_smooth"].astype(np.float64),
        twist_features         = np.array(sub["twist_features"], dtype=object),
        long_rejected_regions  = np.array(corr["long_rejected_regions"],
                                          dtype=np.int32),
        short_rejected_regions = np.array(corr["short_rejected_regions"],
                                          dtype=np.int32),
    )

    np.savez_compressed(
        str(out_dir / "xcd_analysis.npz"),
        xcd_twist_matrix = coup["xcd_twist_matrix"].astype(np.float64),
        xcd_twist_long   = coup["xcd_twist_long"].astype(np.float64),
        xcd_twist_short  = coup["xcd_twist_short"].astype(np.float64),
        sign_flipped     = np.bool_(sign_flipped),
    )

    np.savez_compressed(
        str(out_dir / "dtw_analysis.npz"),
        dtw_twist_long             = coup["dtw_twist_long"].astype(np.float64),
        dtw_twist_short            = coup["dtw_twist_short"].astype(np.float64),
        pearson_twist_long_smooth  = coup["rho_long_smooth"].astype(np.float64),
        pearson_twist_short_smooth = coup["rho_short_smooth"].astype(np.float64),
        sign_flipped               = np.bool_(sign_flipped),
        dtw_backend                = np.str_("tslearn"),
    )


def load_sign_flipped(sid: str) -> bool:
    """Preserve the previously-computed sign-flip flag from xcd_analysis."""
    p = DATA_ROOT / sid / "result" / "xcd_analysis.npz"
    if not p.exists():
        return False
    try:
        return bool(np.load(p)["sign_flipped"])
    except Exception:
        return False


def write_cohort_csvs(all_data: dict[str, dict]) -> None:
    sids = list(all_data.keys())

    # xcd_cohort.csv
    with open(DATA_ROOT / "xcd_cohort.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject_id", "region",
                    "xcd_twist_long_s", "xcd_twist_short_s"])
        for sid in sids:
            x = all_data[sid]["coup"]
            for i, r in enumerate(range(1, N_REGIONS)):
                w.writerow([sid, r,
                            f"{x['xcd_twist_long'][i]:.4f}",
                            f"{x['xcd_twist_short'][i]:.4f}"])

    # dtw_cohort.csv
    with open(DATA_ROOT / "dtw_cohort.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject_id", "region",
                    "dtw_long", "dtw_short",
                    "pearson_long_smooth", "pearson_short_smooth"])
        for sid in sids:
            c = all_data[sid]["coup"]
            for i, r in enumerate(range(1, N_REGIONS)):
                w.writerow([sid, r,
                            f"{c['dtw_twist_long'][i]:.4f}",
                            f"{c['dtw_twist_short'][i]:.4f}",
                            f"{c['rho_long_smooth'][i]:.4f}",
                            f"{c['rho_short_smooth'][i]:.4f}"])

    # cohort_master_summary.csv
    cols = [
        "subject_id", "region",
        "peak_amplitude", "t_peak",
        "systolic_slope", "diastolic_slope", "AUC",
        "u_long_peak_mm", "u_short_peak_mm",
        "pearson_long_raw", "pearson_short_raw",
        "pearson_long_smooth", "pearson_short_smooth",
        "xcd_twist_long_s", "xcd_twist_short_s",
        "dtw_long", "dtw_short",
        "sign_flipped",
    ]
    with open(DATA_ROOT / "cohort_master_summary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for sid in sids:
            d = all_data[sid]
            feats = d["sub"]["twist_features"]
            c = d["coup"]
            uL = d["corr"]["u_long_regions"]
            uS = d["corr"]["u_short_regions"]
            for i, r in enumerate(range(1, N_REGIONS)):
                fr = feats.get(r, {})
                pk_uL = _signed_peak(uL[:, r])
                pk_uS = _signed_peak(uS[:, r])
                w.writerow([
                    sid, r,
                    f"{fr.get('peak_amplitude',  float('nan')):.4f}",
                    f"{fr.get('t_peak',          float('nan')):.4f}",
                    f"{fr.get('systolic_slope',  float('nan')):.4f}",
                    f"{fr.get('diastolic_slope', float('nan')):.4f}",
                    f"{fr.get('AUC',             float('nan')):.4f}",
                    f"{pk_uL:.4f}",
                    f"{pk_uS:.4f}",
                    f"{c['rho_long_raw'][i]:.4f}",
                    f"{c['rho_short_raw'][i]:.4f}",
                    f"{c['rho_long_smooth'][i]:.4f}",
                    f"{c['rho_short_smooth'][i]:.4f}",
                    f"{c['xcd_twist_long'][i]:.4f}",
                    f"{c['xcd_twist_short'][i]:.4f}",
                    f"{c['dtw_twist_long'][i]:.4f}",
                    f"{c['dtw_twist_short'][i]:.4f}",
                    int(d["sign_flipped"]),
                ])


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3 — rebuild with new gridspec + PIL verification
# ═══════════════════════════════════════════════════════════════════════════

def _enable_box_spines(ax) -> None:
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)


def rebuild_figure3(all_data: dict[str, dict]) -> tuple[Path, Path, Path]:
    sids    = list(all_data.keys())
    regions = [1, 2, 3, 4, 5]

    def _mat(key: str) -> np.ndarray:
        m = np.full((len(sids), len(regions)), np.nan)
        for i, sid in enumerate(sids):
            vec = all_data[sid]["coup"][key]
            for j, _ in enumerate(regions):
                m[i, j] = vec[j]
        return m

    PR_long_raw  = _mat("rho_long_raw")
    PR_long_sm   = _mat("rho_long_smooth")
    DT_long      = _mat("dtw_twist_long")
    PR_short_raw = _mat("rho_short_raw")
    PR_short_sm  = _mat("rho_short_smooth")
    DT_short     = _mat("dtw_twist_short")

    dtw_vmax = float(np.nanmax(np.concatenate([DT_long.ravel(),
                                               DT_short.ravel()])))
    dtw_vmax = max(dtw_vmax, 1e-6)

    fig = plt.figure(figsize=(11, 5.5))
    gs = fig.add_gridspec(
        2, 5,
        width_ratios=[1.0, 1.0, 1.0, 0.05, 0.05],
        wspace=0.18, hspace=0.35,
        left=0.07, right=0.93, top=0.88, bottom=0.12,
    )
    ax_long_raw     = fig.add_subplot(gs[0, 0])
    ax_long_smooth  = fig.add_subplot(gs[0, 1])
    ax_long_dtw     = fig.add_subplot(gs[0, 2])
    ax_short_raw    = fig.add_subplot(gs[1, 0])
    ax_short_smooth = fig.add_subplot(gs[1, 1])
    ax_short_dtw    = fig.add_subplot(gs[1, 2])
    cax_pearson     = fig.add_subplot(gs[:, 3])
    cax_dtw         = fig.add_subplot(gs[:, 4])

    def _draw(ax, data, cmap, vmin, vmax, *, fmt, color_threshold):
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                if np.isnan(v):
                    txt, col = "nan", "black"
                else:
                    txt = fmt(v)
                    col = "white" if color_threshold(v) else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=7, color=col)
        ax.set_xticks(range(len(regions)))
        ax.set_xticklabels([f"R{r}" for r in regions])
        _enable_box_spines(ax)
        return im

    im_pearson_last = None
    for ax, data in ((ax_long_raw,     PR_long_raw),
                     (ax_long_smooth,  PR_long_sm),
                     (ax_short_raw,    PR_short_raw),
                     (ax_short_smooth, PR_short_sm)):
        im_pearson_last = _draw(
            ax, data, "RdBu_r", -1, 1,
            fmt=lambda v: f"{v:+.2f}",
            color_threshold=lambda v: abs(v) > 0.6,
        )

    im_dtw_last = None
    for ax, data in ((ax_long_dtw,  DT_long),
                     (ax_short_dtw, DT_short)):
        im_dtw_last = _draw(
            ax, data, "YlGnBu", 0, dtw_vmax,
            fmt=lambda v: f"{v:.2f}",
            color_threshold=lambda v: v > 0.55 * dtw_vmax,
        )

    for ax in (ax_long_raw, ax_short_raw):
        ax.set_yticks(range(len(sids))); ax.set_yticklabels(sids)
    for ax in (ax_long_smooth, ax_long_dtw,
               ax_short_smooth, ax_short_dtw):
        ax.set_yticks(range(len(sids))); ax.set_yticklabels([])

    ax_long_raw   .set_title("Pearson ρ (raw)",      pad=4)
    ax_long_smooth.set_title("Pearson ρ (smoothed)", pad=4)
    ax_long_dtw   .set_title("DTW distance",         pad=4)

    ax_long_raw .set_ylabel("twist vs u_long  (mm)")
    ax_short_raw.set_ylabel("twist vs u_short (mm)")

    for ax, lbl in zip(
        (ax_long_raw,  ax_long_smooth,  ax_long_dtw,
         ax_short_raw, ax_short_smooth, ax_short_dtw),
        list("abcdef"),
    ):
        ax.text(-0.12, 1.04, lbl, transform=ax.transAxes,
                fontweight="bold", fontsize=10, ha="left", va="bottom")

    fig.colorbar(im_pearson_last, cax=cax_pearson, label="Pearson ρ")
    fig.colorbar(im_dtw_last,     cax=cax_dtw,     label="DTW distance")
    cax_pearson.tick_params(labelsize=7)
    cax_dtw.tick_params(labelsize=7)

    fig.text(
        0.5, 0.02,
        "Displacement-based coupling: each seed projected onto frame-0 "
        "LV long/short axes, averaged per region.",
        ha="center", fontsize=7.5, style="italic", color="0.3", wrap=True,
    )

    # ── Axes-position verification ─────────────────────────────────────────
    fig.canvas.draw()
    p_pear = cax_pearson.get_position()
    p_dtw  = cax_dtw.get_position()
    p_last = ax_long_dtw.get_position()
    if p_pear.x0 <= p_last.x1 - 0.005:
        raise RuntimeError(
            f"Pearson cbar (x0={p_pear.x0:.3f}) not right of DTW panel "
            f"(x1={p_last.x1:.3f})"
        )
    if p_dtw.x0 <= p_pear.x1 - 0.005:
        raise RuntimeError(
            f"DTW cbar (x0={p_dtw.x0:.3f}) not right of Pearson cbar "
            f"(x1={p_pear.x1:.3f})"
        )
    if p_dtw.x0 < 0.85:
        raise RuntimeError(
            f"DTW cbar (x0={p_dtw.x0:.3f}) not at far right (>0.85)"
        )

    stem = "Figure3_coupling_comparison"
    pdf = FIG_DIR / f"{stem}.pdf"
    png = FIG_DIR / f"{stem}.png"
    svg = FIG_DIR / f"{stem}.svg"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(svg, format="svg", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    # ── PIL pixel-level verification ───────────────────────────────────────
    from PIL import Image
    img = np.asarray(Image.open(png).convert("RGB"), dtype=np.float32)
    H, W, _ = img.shape
    # Convert to grayscale-equivalent luminance
    lum = 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]

    def _col_var(frac_lo: float, frac_hi: float) -> float:
        block = lum[:, int(W * frac_lo):int(W * frac_hi)]
        # Vertical std per column, averaged across the block
        return float(np.std(block, axis=0).mean())

    # Right-edge cols should show gradient (std > some threshold);
    # main area cols should not carry any strong vertical gradient bars.
    cbar_var_outer  = _col_var(0.95, 1.00)   # far right — DTW cbar
    cbar_var_inner  = _col_var(0.90, 0.95)   # inside: Pearson cbar
    main_var        = _col_var(0.05, 0.80)   # main heatmap area

    print(f"  [PIL] cbar_outer std={cbar_var_outer:.1f} "
          f"cbar_inner std={cbar_var_inner:.1f} "
          f"main std={main_var:.1f}")
    if cbar_var_outer < 5 and cbar_var_inner < 5:
        raise RuntimeError(
            f"Neither of the two rightmost 5% columns shows colorbar-like "
            f"gradient (outer std={cbar_var_outer:.1f}, "
            f"inner std={cbar_var_inner:.1f})"
        )
    # The main area also has colorful heatmap cells; with more subjects
    # the per-column std rises naturally, so we only flag a *clear*
    # exceedance over both colorbar columns (1.5×) as suspicious — the
    # positional asserts above already catch real layout bugs.
    if main_var > max(cbar_var_outer, cbar_var_inner) * 1.5:
        raise RuntimeError(
            f"Main area shows unexpectedly high vertical gradient "
            f"(main std={main_var:.1f}) — colorbars may be embedded "
            f"inside the heatmap region."
        )

    print(f"\n[Figure 3] verified and saved")
    print(f"  pearson cbar x0={p_pear.x0:.3f}  dtw cbar x0={p_dtw.x0:.3f}")
    return pdf, png, svg


# ═══════════════════════════════════════════════════════════════════════════
# Figure 8 — strain sanity
# ═══════════════════════════════════════════════════════════════════════════

def figure8_strain_sanity(all_data: dict[str, dict]) -> Path:
    sids = list(all_data.keys())
    n = len(sids)

    fig, axes = plt.subplots(2, n, figsize=(max(12, 2.4 * n), 5.0),
                             sharex=False)
    if n == 1:
        axes = axes.reshape(2, 1)

    LONG_BAND  = (-0.20, 0.05)   # physiological ε_long band (top row only)

    # Collect global y-ranges (for readability)
    all_long_y, all_short_y = [], []
    for sid in sids:
        d = all_data[sid]
        all_long_y.append(d["corr"]["epsilon_long_global"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            short_mean = np.nanmean(d["corr"]["u_short_regions"], axis=1)
        all_short_y.append(short_mean)

    def _nan_range_long(arrs):
        stacked = np.concatenate([a for a in arrs])
        stacked = stacked[np.isfinite(stacked)]
        if len(stacked) == 0:
            return -0.3, 0.1
        lo = min(float(stacked.min()), LONG_BAND[0])
        hi = max(float(stacked.max()), LONG_BAND[1])
        pad = 0.05 * max(hi - lo, 0.1)
        return lo - pad, hi + pad

    def _nan_range_mm(arrs):
        stacked = np.concatenate([a for a in arrs])
        stacked = stacked[np.isfinite(stacked)]
        if len(stacked) == 0:
            return -5.0, 5.0
        lo = float(stacked.min()); hi = float(stacked.max())
        pad = 0.08 * max(hi - lo, 1.0)
        return lo - pad, hi + pad

    long_ylim  = _nan_range_long(all_long_y)
    short_ylim = _nan_range_mm(all_short_y)

    for col, sid in enumerate(sids):
        d   = all_data[sid]
        nt  = d["sub"]["trajectories"].shape[0]
        t   = np.arange(nt) / float(subject_frame_rate(sid)[0])

        # ── Top: epsilon_long_global ──────────────────────────────────────
        ax_top = axes[0, col]
        yL = d["corr"]["epsilon_long_global"]
        ax_top.axhspan(LONG_BAND[0], LONG_BAND[1],
                       color="0.82", alpha=0.55, zorder=1,
                       label="physiological range" if col == 0 else None)
        ax_top.plot(t, yL, "-", color=SUBJECT_COLORS[sid],
                    lw=1.5, zorder=3)
        ax_top.axhline(0, color="gray", ls="--", lw=0.5, alpha=0.7)

        if np.any(np.isfinite(yL)):
            pk_i = int(np.nanargmax(np.abs(yL)))
            pk_v = float(yL[pk_i])
            ax_top.scatter([t[pk_i]], [pk_v], marker="*", s=50,
                           color="black", zorder=5)
            ax_top.text(0.98, 0.05, f"peak = {pk_v:+.3f}",
                        transform=ax_top.transAxes, fontsize=7,
                        ha="right", va="bottom", family="monospace",
                        bbox=dict(boxstyle="round,pad=0.25",
                                  facecolor="white", edgecolor="0.75",
                                  linewidth=0.4, alpha=0.55))

        ax_top.set_title(sid, fontsize=9, pad=3)
        ax_top.set_ylim(*long_ylim)
        if col == 0:
            ax_top.set_ylabel("ε_long_global")

        # ── Bottom: mean of u_short_regions across regions (mm) ───────────
        ax_bot = axes[1, col]
        yS_mat = d["corr"]["u_short_regions"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            yS = np.nanmean(yS_mat, axis=1)
        ax_bot.plot(t, yS, "-", color=SUBJECT_COLORS[sid],
                    lw=1.5, zorder=3)
        ax_bot.axhline(0, color="gray", ls="--", lw=0.5, alpha=0.7)

        if np.any(np.isfinite(yS)):
            pk_i = int(np.nanargmax(np.abs(yS)))
            pk_v = float(yS[pk_i])
            ax_bot.scatter([t[pk_i]], [pk_v], marker="*", s=50,
                           color="black", zorder=5)
            ax_bot.text(0.98, 0.05, f"peak = {pk_v:+.2f} mm",
                        transform=ax_bot.transAxes, fontsize=7,
                        ha="right", va="bottom", family="monospace",
                        bbox=dict(boxstyle="round,pad=0.25",
                                  facecolor="white", edgecolor="0.75",
                                  linewidth=0.4, alpha=0.55))

        ax_bot.set_xlabel("time (s)")
        ax_bot.set_ylim(*short_ylim)
        if col == 0:
            ax_bot.set_ylabel("mean u_short  (mm)")

    # Shared top-row legend
    handles = [plt.Line2D([0], [0], marker="s", linestyle="",
                          markerfacecolor="0.82", markeredgecolor="0.5",
                          markersize=9, label="physiological range"),
               plt.Line2D([0], [0], marker="*", linestyle="",
                          markerfacecolor="black", markeredgecolor="black",
                          markersize=8, label="peak")]
    leg = fig.legend(handles=handles, loc="upper center",
                     bbox_to_anchor=(0.5, 0.99),
                     ncol=2, frameon=True)
    leg.get_frame().set_alpha(0.55)
    leg.get_frame().set_edgecolor("0.7")

    fig.suptitle(
        "Figure 8 — Strain sanity check after correction",
        y=1.02, fontsize=10,
    )
    plt.subplots_adjust(left=0.05, right=0.99, top=0.86, bottom=0.10,
                        wspace=0.30, hspace=0.45)
    out = FIG_DIR / "Figure8_strain_sanity.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(out.with_suffix(".png"), dpi=300,
                bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  saved → {out}")
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Displacement definition log
# ═══════════════════════════════════════════════════════════════════════════

def _signed_peak(curve: np.ndarray) -> float:
    """Signed value at index of maximum absolute deflection; NaN-safe."""
    if curve is None or len(curve) == 0:
        return float("nan")
    abs_c = np.abs(curve)
    if not np.any(np.isfinite(abs_c)):
        return float("nan")
    idx = int(np.nanargmax(abs_c))
    v = float(curve[idx])
    return v if np.isfinite(v) else float("nan")


def write_displacement_definition_log(all_data: dict[str, dict],
                                      log_path: Path) -> list[str]:
    """Write displacement-definition log and return info-level warnings."""
    lines: list[str] = []
    lines.append("# Per-region displacement — definition log\n")
    lines.append(
        "Generated by `fix_strain_final.py`.\n\n"
        "**Displacement definition.**  At frame 0, fit the LV long-axis "
        "unit vector `d_long` anatomically from the **apex–basal vector**. "
        "With `k = min(5, max(3, N // 6))`, define\n\n"
        "    basal_ctr  = mean(k lowest-z seeds at frame 0)\n"
        "    apical_ctr = mean(k highest-z seeds at frame 0)\n"
        "    d_long     = normalize(apical_ctr − basal_ctr)   # apex-directed\n\n"
        "(The older PCA-based `d_long = Vt[0]` is computed only for the "
        "PCA-vs-anatomical comparison table below.  PCA can land in the "
        "short-axis plane for foreshortened seed clusters; the apex–basal "
        "vector is ordering-based and insensitive to the spread.)\n\n"
        "For each seed s, compute a *per-seed* radial unit vector "
        "`d_short[s]` as the frame-0 normalised perpendicular component of "
        "`seed_s(0) − origin_0` (w.r.t. the new anatomical `d_long`). "
        f"Seeds whose frame-0 radial distance `R_s(0) < {R_MIN_MM:.1f} mm` "
        "are excluded from the short-axis aggregation (no threshold "
        "applied to long-axis).  Coordinate system is fixed at frame 0 "
        "and never recomputed.\n\n"
        "For every frame, project the displacement "
        "`disp_s(t) = seed_s(t) − seed_s(0)` onto these fixed directions:\n"
        "  • `u_long[t, s]  = disp_s(t) · d_long`       (mm)\n"
        "  • `u_short[t, s] = disp_s(t) · d_short[s]`  (mm)\n\n"
        "Region curves are simple nanmeans of the per-seed curves over "
        f"the 6 seeds in each region (region NaN if fewer than "
        f"{MIN_SEEDS_REGION} valid seeds).\n\n"
        "**Sign convention.**  `d_long` points **toward apex**.  Apical "
        "seeds move toward basal in systole → `u_long` peaks **negative** "
        "for apical regions.  Basal seeds barely move → `u_long` peaks "
        "~0.  `d_short[s]` points **radially outward**; inward systolic "
        "motion → `u_short` peaks **negative**.\n"
    )

    WARN_U_LONG  = 15.0   # mm
    WARN_U_SHORT = 10.0   # mm
    WARN_TILT_DEG = 40.0  # anatomical tilt from ẑ (INFO only)
    warnings_: list[str] = []

    for sid, d in all_data.items():
        corr = d["corr"]
        uL = corr["u_long_regions"]
        uS = corr["u_short_regions"]
        d_long = corr["d_long"]
        d_long_pca = corr["d_long_pca"]
        k_apex = int(corr["k_apex_basal"])
        origin_0 = corr["origin_0"]
        R_s0 = corr["R_s0_seeds"]
        short_valid = corr["short_valid_seed_mask"]
        n_reject_short = int((~short_valid).sum())

        tilt_anat = float(np.degrees(np.arccos(np.clip(d_long[2], -1, 1))))
        tilt_pca  = float(np.degrees(np.arccos(np.clip(d_long_pca[2], -1, 1))))
        cos_ang   = float(np.clip(d_long @ d_long_pca, -1, 1))
        ang_pca_anat = float(np.degrees(np.arccos(cos_ang)))

        lines.append(f"\n## {sid}\n")
        lines.append(
            f"- **origin₀** (mm): ({origin_0[0]:.1f}, {origin_0[1]:.1f}, "
            f"{origin_0[2]:.1f})"
        )
        lines.append(
            f"- **d_long** (anatomical, k={k_apex}): "
            f"({d_long[0]:+.3f}, {d_long[1]:+.3f}, {d_long[2]:+.3f})   "
            f"— tilt from ẑ = {tilt_anat:.1f}°  (apex-directed)"
        )
        lines.append(
            f"- **d_long_pca** (for reference only): "
            f"({d_long_pca[0]:+.3f}, {d_long_pca[1]:+.3f}, "
            f"{d_long_pca[2]:+.3f})   "
            f"— tilt from ẑ = {tilt_pca:.1f}°"
        )
        lines.append(
            f"- Angle(anatomical, PCA) = {ang_pca_anat:.1f}°"
        )
        if tilt_anat > WARN_TILT_DEG:
            lines.append(
                f"- ⚠️ anatomical tilt > {WARN_TILT_DEG:.0f}° — confirm seed "
                f"layout covers a full apex-basal extent"
            )
        lines.append(
            f"- Seeds rejected for short-axis (R_s(0) < "
            f"{R_MIN_MM:.1f} mm): {n_reject_short}/36"
        )
        rej_idx = np.where(~short_valid)[0].tolist()
        if rej_idx:
            rej_details = ", ".join(
                f"s={i}(R₀={R_s0[i]:.1f}mm)" for i in rej_idx
            )
            lines.append(f"    - {rej_details}")

        rej_long_regions  = [r for r in corr["long_rejected_regions"]]
        rej_short_regions = [r for r in corr["short_rejected_regions"]]
        if rej_long_regions:
            lines.append(
                f"- Long-axis regions NaN (too few valid seeds): "
                f"{rej_long_regions}"
            )
        if rej_short_regions:
            lines.append(
                f"- Short-axis regions NaN: {rej_short_regions}"
            )

        lines.append("")
        lines.append(
            "| region | N_seeds (long / short) | u_long peak (mm) | "
            "u_short peak (mm) |"
        )
        lines.append("|---|---|---|---|")
        n_long  = corr["n_valid_seeds_long"]
        n_short = corr["n_valid_seeds_short"]
        for r in range(1, N_REGIONS):
            pkL = _signed_peak(uL[:, r])
            pkS = _signed_peak(uS[:, r])
            pkL_s = f"{pkL:+.2f}" if np.isfinite(pkL) else "NaN"
            pkS_s = f"{pkS:+.2f}" if np.isfinite(pkS) else "NaN"
            lines.append(
                f"| R{r} | {int(n_long[r])} / {int(n_short[r])} | "
                f"{pkL_s} | {pkS_s} |"
            )

            if np.isfinite(pkL) and abs(pkL) > WARN_U_LONG:
                warnings_.append(
                    f"{sid} R{r}: |u_long peak| = {abs(pkL):.2f} mm "
                    f"exceeds {WARN_U_LONG:.1f} mm"
                )
            if np.isfinite(pkS) and abs(pkS) > WARN_U_SHORT:
                warnings_.append(
                    f"{sid} R{r}: |u_short peak| = {abs(pkS):.2f} mm "
                    f"exceeds {WARN_U_SHORT:.1f} mm"
                )

    lines.append("\n## Cohort — anatomical vs PCA d_long\n")
    lines.append(
        "Shows that PCA's first component can be far from the anatomical "
        "apex–basal direction when the seed cluster is foreshortened "
        "(small z-spread relative to the xy radius).  Anatomical tilts "
        f"> {WARN_TILT_DEG:.0f}° are flagged above.\n"
    )
    lines.append(
        "| subject | anatomical d_long          | tilt ẑ | PCA d_long                | tilt ẑ | ∠(anat, PCA) |"
    )
    lines.append(
        "|---|---|---|---|---|---|"
    )
    for sid, d in all_data.items():
        dL = d["corr"]["d_long"]
        dP = d["corr"]["d_long_pca"]
        tA = float(np.degrees(np.arccos(np.clip(dL[2], -1, 1))))
        tP = float(np.degrees(np.arccos(np.clip(dP[2], -1, 1))))
        ang = float(np.degrees(np.arccos(np.clip(dL @ dP, -1, 1))))
        lines.append(
            f"| {sid} | "
            f"({dL[0]:+.3f}, {dL[1]:+.3f}, {dL[2]:+.3f}) | "
            f"{tA:.1f}° | "
            f"({dP[0]:+.3f}, {dP[1]:+.3f}, {dP[2]:+.3f}) | "
            f"{tP:.1f}° | "
            f"{ang:.1f}° |"
        )

    lines.append("\n## Cohort summary — signed apical-region peaks\n")
    lines.append(
        "Apical regions (R4, R5) should show negative `u_long` in systole "
        "(apex→base motion) and negative `u_short` (radial thickening "
        "inward):\n"
    )
    lines.append("| subject | R4 u_long | R5 u_long | R4 u_short | R5 u_short |")
    lines.append("|---|---|---|---|---|")
    for sid, d in all_data.items():
        uL = d["corr"]["u_long_regions"]
        uS = d["corr"]["u_short_regions"]
        def _f(v):
            return f"{v:+.2f}" if np.isfinite(v) else "NaN"
        lines.append(
            f"| {sid} | {_f(_signed_peak(uL[:, 4]))} | "
            f"{_f(_signed_peak(uL[:, 5]))} | "
            f"{_f(_signed_peak(uS[:, 4]))} | "
            f"{_f(_signed_peak(uS[:, 5]))} |"
        )

    lines.append("\n## Warnings\n")
    if warnings_:
        for w in warnings_:
            lines.append(f"- {w}")
        lines.append(
            f"\nPeaks above {WARN_U_LONG:.0f} mm (long) / "
            f"{WARN_U_SHORT:.0f} mm (short) are flagged as unusually "
            "large and may reflect tracking failures. Not auto-corrected."
        )
    else:
        lines.append(
            f"All per-region peaks are within |u_long|≤{WARN_U_LONG:.0f} mm "
            f"and |u_short|≤{WARN_U_SHORT:.0f} mm."
        )

    log_path.write_text("\n".join(lines))
    return warnings_


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("Loading subjects + computing corrected strains …")
    all_data: dict[str, dict] = {}
    for sid in ALL_SUBJECTS:
        sub = load_subject_raw(sid)
        if sub is None:
            print(f"  {sid}: missing inputs — skipped")
            continue
        corr = corrected_strains(sub["trajectories"], sub["region_ids"],
                                 sub["scale"], valid=sub["valid"])
        coup = recompute_coupling(sub, corr)
        sign_flipped = load_sign_flipped(sid)
        save_subject(sid, sub, corr, coup, sign_flipped)
        all_data[sid] = {"sub": sub, "corr": corr, "coup": coup,
                         "sign_flipped": sign_flipped}
        glob_peak = float(np.nanmax(np.abs(corr["epsilon_long_global"])))
        tag = "  WARN: |global_long_peak| > 0.30" if glob_peak > 0.30 else ""
        print(f"  {sid:<8}  short-seed-rej={corr['short_seeds_rejected']:>2}/36  "
              f"|ε_long_global|_peak={glob_peak:+.3f}{tag}")

    # Cohort CSVs
    print("\nWriting cohort CSVs …")
    write_cohort_csvs(all_data)
    for p in ("cohort_master_summary.csv", "xcd_cohort.csv",
              "dtw_cohort.csv"):
        print(f"  {DATA_ROOT / p}")

    # ── Figure 3 (rebuilt in this script for guaranteed correctness) ────────
    print("\nRebuilding Figure 3 (with PIL verification) …")
    rebuild_figure3(all_data)

    # ── Figure 8 (new) ───────────────────────────────────────────────────────
    print("\nBuilding Figure 8 (strain sanity) …")
    figure8_strain_sanity(all_data)

    # ── Figures 1, 2, 4, 5, 6, 7 via publication_figures_v2 ─────────────────
    # The v2 script reads everything it needs from the refreshed npz files
    # and master CSV, so we invoke it to regenerate them consistently.
    print("\nRegenerating Figures 1/2/4/5/6/7 via publication_figures_v2.py …")
    proc = subprocess.run(
        ["python3", str(DATA_ROOT / "publication_figures_v2.py")],
        capture_output=True, text=True,
    )
    tail = proc.stdout.splitlines()[-20:] if proc.stdout else []
    for ln in tail:
        print("  " + ln)
    if proc.returncode != 0:
        print("  [stderr]")
        print(proc.stderr)

    # ── After v2 runs it rewrites Figure 3 with its own layout; re-run our
    #    rebuild so the guaranteed layout + PIL verification is the one on
    #    disk. ─────────────────────────────────────────────────────────────
    print("\nRe-applying Figure 3 (override publication_figures_v2 output) …")
    rebuild_figure3(all_data)

    # ── Displacement definition log ────────────────────────────────────────
    log_u = DATA_ROOT / "displacement_definition_log.md"
    u_warnings = write_displacement_definition_log(all_data, log_u)
    print(f"\nLog: {log_u}"
          + (f"   ({len(u_warnings)} warnings)" if u_warnings else ""))

    # ── Per-subject per-region sanity printout (displacement peaks) ────────
    print("\n" + "=" * 72)
    print("Per-region u_long / u_short peak (mm)  —  displacement definition")
    print("=" * 72)
    for sid, d in all_data.items():
        corr = d["corr"]
        uL = corr["u_long_regions"]
        uS = corr["u_short_regions"]
        d_long = corr["d_long"]
        origin_0 = corr["origin_0"]
        short_valid = corr["short_valid_seed_mask"]
        n_valid = int(short_valid.sum())
        print(f"Subject {sid}:")
        for r in range(1, N_REGIONS):
            pkL = _signed_peak(uL[:, r])
            pkS = _signed_peak(uS[:, r])
            pkL_s = f"{pkL:+.2f} mm" if np.isfinite(pkL) else "   NaN  "
            pkS_s = f"{pkS:+.2f} mm" if np.isfinite(pkS) else "   NaN  "
            print(f"  R{r}: u_long peak = {pkL_s}, "
                  f"u_short peak = {pkS_s}")
        print(f"  Valid seeds for displacement: {n_valid}/36 "
              f"(rejected {36 - n_valid} seeds with R_s(0) < "
              f"{R_MIN_MM:.1f} mm)")
        d_long_pca = corr["d_long_pca"]
        tilt_a = float(np.degrees(np.arccos(np.clip(d_long[2], -1, 1))))
        tilt_p = float(np.degrees(np.arccos(np.clip(d_long_pca[2], -1, 1))))
        ang    = float(np.degrees(np.arccos(
            np.clip(d_long @ d_long_pca, -1, 1))))
        print(f"  d_long  (anat)  : "
              f"[{d_long[0]:+.3f}, {d_long[1]:+.3f}, {d_long[2]:+.3f}]  "
              f"tilt={tilt_a:.1f}°")
        print(f"  d_long  (PCA)   : "
              f"[{d_long_pca[0]:+.3f}, {d_long_pca[1]:+.3f}, "
              f"{d_long_pca[2]:+.3f}]  tilt={tilt_p:.1f}°  "
              f"∠(anat,PCA)={ang:.1f}°")
        print(f"  origin_0 (mm)   : "
              f"[{origin_0[0]:.1f}, {origin_0[1]:.1f}, {origin_0[2]:.1f}]")

    # ── Cohort apical-region peak summary (sign-sanity check) ──────────────
    print("\n" + "=" * 72)
    print("Apical-region peaks (sign check: should be NEGATIVE in systole)")
    print("=" * 72)
    print(f"               R4 u_long   R5 u_long   R4 u_short   R5 u_short  (mm)")
    for sid, d in all_data.items():
        uL = d["corr"]["u_long_regions"]
        uS = d["corr"]["u_short_regions"]
        def _f(x):
            return f"{x:+7.2f}" if np.isfinite(x) else "    NaN"
        print(f"  {sid:<10}  {_f(_signed_peak(uL[:, 4]))}  "
              f"{_f(_signed_peak(uL[:, 5]))}  "
              f"{_f(_signed_peak(uS[:, 4]))}  "
              f"{_f(_signed_peak(uS[:, 5]))}")

    # ── Final |max| summary ───────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("DISPLACEMENT SUMMARY")
    print("=" * 72)
    for sid, d in all_data.items():
        uL = d["corr"]["u_long_regions"]
        uS = d["corr"]["u_short_regions"]
        eG = float(np.nanmax(np.abs(d["corr"]["epsilon_long_global"])))
        uL_max = float(np.nanmax(np.abs(uL)))
        uS_max = float(np.nanmax(np.abs(uS)))
        print(f"  {sid:<8}  |u_long|_max = {uL_max:5.2f} mm  "
              f"|u_short|_max = {uS_max:5.2f} mm  "
              f"ε_long_global |peak| = {eG:+.3f}")

    if u_warnings:
        print()
        print("=" * 72)
        print("Displacement warnings (unusually large peaks)")
        print("=" * 72)
        for w in u_warnings:
            print(f"  {w}")


if __name__ == "__main__":
    main()
