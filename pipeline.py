"""
pipeline.py — v5  (true 3D speckle tracking, global seed placement)
====================================================================
Seeds are placed **directly in the full 3D myocardial volume** — not one
z-slice at a time.  Every seed is tracked in (x, y, z) through all frames.
The result is one global cardiac twist curve and a 3D trajectory visualisation.

Architecture
────────────
  select_seeds_3d_global  → (N, 3) seeds distributed in full 3D mask
  track_3d_direct         → (nt, N, 3) trajectories  +  (nt, N) NCC peaks
  compute_validity_3d     → (nt, N) validity mask
  compute_twist_3d        → (nt,) twist  +  (nt,) twist_std

Outputs
───────
  result/trajectories_3d.png   — 3D trajectory visualisation
  result/twist_global.png      — global twist curve
  result/twist_3d.npz          — numerical results

Usage
─────
  cd /scratch/xg110/BME543
  python -m v5_3d_tracking.code.pipeline
"""

from __future__ import annotations

import os
import time as _time

import numpy as np
import nrrd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401  (registers 3d projection)
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.cm as cm

from scipy.signal import savgol_filter as _savgol

from .segment_3d      import load_mask_3d
from .select_seeds_3d import select_seeds_3d_global
from .track_3d        import track_3d_direct, compute_validity_3d, TEMPLATE_HALF, SEARCH_HALF
from .twist_3d        import compute_twist_3d

# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════
VOL_NPZ    = "us4d.npz"
SEG_NRRD   = "Segmentation.seg.nrrd"
OUT_DIR    = "v5_3d_tracking/result"

FRAME_RATE = 15.0          # volumes / second

N_SEEDS_PER_GROUP = 6      # target seeds per anatomical group (base / apex)
MIN_VALID_FRAC    = 0.50   # drop seeds with < this fraction of valid frames
MIN_DIST_MM  = 12.0        # minimum 3D seed spacing (mm)
N_Z_LEVELS   = 4           # z-depth bands within each group
N_SECTORS    = 8           # angular sectors per z-band
USE_PSR      = False       # True: PSR scoring (better seeds, ~2× slower)

T_HALF        = TEMPLATE_HALF    # (4,4,2) → 9×9×5 template
S_HALF_APEX   = SEARCH_HALF      # (15,15,8) → 31×31×17 — apex: larger motion range
S_HALF_BASE   = (8, 8, 4)        # (8,8,4)  → 17×17×9  — base: smaller/thinner wall

NCC_THRESH      = 0.25   # global validity threshold
NCC_THRESH_BASE = 0.35   # stricter threshold for base (thinner wall, more ambiguous)
RING_TOL_PX     = 5.0

# Anatomical split fractions (same as twist_3d.py)
BASE_FRAC = (0.0, 1/3)
APEX_FRAC = (2/3, 1.0)

# Group colors used across all visualisations
_BASE_COLOR = '#4472C4'   # blue
_APEX_COLOR = '#C0504D'   # red
# ═══════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
# 3D Trajectory visualisation
# ─────────────────────────────────────────────────────────────────────────────

def _viz_trajectories_3d(
    traj      : np.ndarray,    # (nt, N, 3)
    valid     : np.ndarray,    # (nt, N) bool
    seeds     : np.ndarray,    # (N, 3)  int
    ncc_peaks : np.ndarray,    # (nt, N) float
    mask_info,
    scale     : np.ndarray,
    time_axis : np.ndarray,
    out_dir   : str,
    base_mask : np.ndarray | None = None,
    apex_mask : np.ndarray | None = None,
):
    """
    3-panel trajectory figure:
      Left   — compact 3D side view (anatomical context, 3 landmark rings only)
      Right top    — axial (top-down XY) projection, Base group
      Right bottom — axial (top-down XY) projection, Apex group

    The two axial panels directly show the in-plane rotation that twist measures.
    Trajectories are time-colored (plasma colormap: dark=early → bright=late) so
    the direction and magnitude of rotation are immediately visible.
    """
    nt, N, _ = traj.shape

    # ── Smooth trajectories for display only ─────────────────────────────────
    wl = min(11, nt)
    if wl % 2 == 0:
        wl -= 1
    traj_v = traj.copy().astype(np.float64)
    if wl >= 3 and nt > wl:
        for i in range(N):
            if valid[:, i].sum() >= wl:
                for d in range(3):
                    traj_v[:, i, d] = _savgol(traj[:, i, d], wl, 3)

    n_base = int(base_mask.sum()) if base_mask is not None else 0
    n_apex = int(apex_mask.sum()) if apex_mask is not None else 0
    z_lo, z_hi = int(mask_info.z_lo), int(mask_info.z_hi)
    theta = np.linspace(0, 2 * np.pi, 120)
    t_cmap = cm.plasma

    # ── Layout — 3D view only ────────────────────────────────────────────────
    fig  = plt.figure(figsize=(7, 7), facecolor='white')
    ax3d = fig.add_subplot(111, projection='3d', facecolor='white')

    # ════════════════════════════════════════════════════════════
    # Panel 1 — compact 3D side view
    # ════════════════════════════════════════════════════════════
    for pane in (ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor('#e0e0e0')
    ax3d.grid(True, alpha=0.10, linestyle=':')

    # Only 3 landmark rings (base / mid / apex)
    lmark_fracs = {'Base': 0.10, 'Mid': 0.50, 'Apex': 0.90}
    for label, frac in lmark_fracs.items():
        z  = int(z_lo + (z_hi - z_lo) * frac)
        cx, cy = mask_info.centre_at(z)
        ri, ro = mask_info.radii_at(z)
        for r in (ri, ro):
            ax3d.plot(cx + r * np.cos(theta), cy + r * np.sin(theta),
                      np.full(120, float(z)),
                      color='#777777', alpha=0.40, lw=0.9, zorder=1)
        ax3d.text(cx + ro + 1.5, cy, float(z), label,
                  fontsize=7, color='#555', va='center', ha='left', alpha=0.8)

    # LV long-axis — solid line between top and bottom centroid points
    cx_lo, cy_lo = mask_info.centre_at(z_lo)
    cx_hi, cy_hi = mask_info.centre_at(z_hi)
    ax3d.plot([cx_lo, cx_hi], [cy_lo, cy_hi], [float(z_lo), float(z_hi)],
              color='#222', ls='-', lw=1.8, alpha=0.75, zorder=2)

    # Seeds: one color per group, time-fading line
    for i in range(N):
        c  = _BASE_COLOR if (base_mask is not None and base_mask[i]) else _APEX_COLOR
        tr = traj_v[:, i, :]
        # Draw trajectory as segments with increasing opacity
        for t in range(1, nt):
            if valid[t, i]:
                frac  = (t - 1) / max(nt - 2, 1)
                ax3d.plot(tr[t-1:t+1, 0], tr[t-1:t+1, 1], tr[t-1:t+1, 2],
                          color=c, alpha=0.30 + 0.55 * frac,
                          lw=0.8 + 1.0 * frac, solid_capstyle='round', zorder=3)
        ax3d.scatter(*tr[0].tolist(), color=c, s=18, marker='o',
                     edgecolors='#111', linewidths=0.4, depthshade=False,
                     alpha=0.95, zorder=6)

    ax3d.set_xlabel('x', labelpad=4, fontsize=8)
    ax3d.set_ylabel('y', labelpad=4, fontsize=8)
    ax3d.set_zlabel('z', labelpad=4, fontsize=8)
    ax3d.tick_params(labelsize=6)
    x_ext = float(np.ptp(traj_v[:, :, 0]))
    y_ext = float(np.ptp(traj_v[:, :, 1]))
    z_ext = float(np.ptp(traj_v[:, :, 2])) * (float(scale[2]) / float(scale[0]))
    ax3d.set_box_aspect([max(x_ext, 1), max(y_ext, 1), max(z_ext, 1)])
    ax3d.view_init(elev=20, azim=-55)
    ax3d.set_title(f'3D view  (N={N}, {nt} frames)', fontsize=9, pad=6)

    # Proxy handles for legend
    from matplotlib.lines import Line2D
    ax3d.legend(handles=[
        Line2D([0], [0], color=_BASE_COLOR, lw=2, label=f'Base  n={n_base}'),
        Line2D([0], [0], color=_APEX_COLOR, lw=2, label=f'Apex  n={n_apex}'),
    ], fontsize=8, loc='upper left', framealpha=0.7)

    fig.suptitle(
        f'Speckle Trajectories  ·  3D NCC  ·  N={N} seeds  ·  {nt} frames',
        fontsize=10, y=0.98,
    )

    out = f'{out_dir}/trajectories_3d.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {out}')


# ─────────────────────────────────────────────────────────────────────────────
# Twist curve visualisation
# ─────────────────────────────────────────────────────────────────────────────

def _viz_seed_selection(
    vol0      : np.ndarray,   # (nx, ny, nz) uint8
    seeds     : np.ndarray,   # (N, 3) int — (ix, iy, iz)
    mask_info,
    out_dir   : str,
    n_per_group : int = 5,    # unused (kept for API compat)
    base_mask : np.ndarray | None = None,
    apex_mask : np.ndarray | None = None,
):
    """
    Per-seed three-row figure showing three orthogonal views for each seed.

    Layout: 3 rows × N columns (one column per seed, base first then apex)
      Row 0 — XY axial   slice (z = iz) : full US image + mask overlay +
               yellow template box + cyan crosshair
      Row 1 — XZ coronal slice (y = iy) : same overlays in the XZ plane
      Row 2 — YZ sagittal slice (x = ix): same overlays in the YZ plane

    All slices are shown at full resolution (aspect='auto') so nothing is
    distorted.  Red mask overlay, yellow template box, cyan + for seed centre.
    Base seeds get a blue border; Apex seeds get a red border.
    """
    from .track_3d import TEMPLATE_HALF as _TH
    hx, hy, hz = _TH   # 4, 4, 2  →  9×9×5 template

    N = len(seeds)
    if N == 0:
        return
    nx, ny, nz = vol0.shape

    # Column order: base seeds first, then apex
    base_idx  = np.where(base_mask)[0] if base_mask is not None else np.array([], int)
    apex_idx  = np.where(apex_mask)[0] if apex_mask is not None else np.array([], int)
    col_order = np.concatenate([base_idx, apex_idx]).astype(int)
    n_cols    = len(col_order)
    n_base    = len(base_idx)

    # Rows 0-2: orthogonal slices; Row 3: intensity profiles (shorter)
    # height_ratios are chosen so each subplot matches its data's native aspect ratio,
    # preventing stretching when aspect='auto' fills the allocated space.
    #   Row 0 XY   slice: shape (ny, nx) → height/width = ny/nx
    #   Row 1 XZ   slice: shape (nz, nx) → height/width = nz/nx
    #   Row 2 YZ   slice: shape (nz, ny) → height/width = nz/ny
    hr_xy   = ny / nx
    hr_xz   = nz / nx
    hr_yz   = nz / ny
    hr_prof = 0.55 * hr_xz          # profile row: shorter than a slice row

    col_w   = 2.2                   # inches per column
    h_slices = col_w * (hr_xy + hr_xz + hr_yz)
    h_prof   = col_w * hr_prof
    fig, axes = plt.subplots(
        4, n_cols,
        figsize=(col_w * n_cols, h_slices + h_prof + 0.8),
        facecolor='white',
        gridspec_kw={'hspace': 0.08, 'wspace': 0.06,
                     'height_ratios': [hr_xy, hr_xz, hr_yz, hr_prof]},
    )
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    # Row labels (leftmost column only)
    axes[0, 0].set_ylabel('XY  axial\n(z = iz)', fontsize=7, labelpad=4)
    axes[1, 0].set_ylabel('XZ  coronal\n(y = iy)', fontsize=7, labelpad=4)
    axes[2, 0].set_ylabel('YZ  sagittal\n(x = ix)', fontsize=7, labelpad=4)
    axes[3, 0].set_ylabel('Intensity\nprofile', fontsize=7, labelpad=4)

    def _draw_slice(ax, sl, msk, seed_col, seed_row,
                    box_col0, box_row0, box_dcol, box_drow,
                    grp_color):
        """Render one full-slice panel with overlays."""
        ax.imshow(sl,  cmap='gray', origin='lower', aspect='auto',
                  vmin=0, vmax=255)
        if msk.any():
            ax.imshow(np.where(msk > 0, 1.0, np.nan),
                      cmap='Reds', vmin=0, vmax=1,
                      alpha=0.35, origin='lower', aspect='auto')
        rect = plt.Rectangle(
            (box_col0 - 0.5, box_row0 - 0.5), box_dcol, box_drow,
            edgecolor='yellow', facecolor='none', lw=1.1, zorder=4)
        ax.add_patch(rect)
        ax.plot(seed_col, seed_row, '+', color='cyan',
                ms=6, mew=1.4, zorder=5)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(grp_color); sp.set_linewidth(1.6)

    for col, si in enumerate(col_order):
        ix, iy, iz = int(seeds[si, 0]), int(seeds[si, 1]), int(seeds[si, 2])
        is_base    = col < n_base
        grp_color  = _BASE_COLOR if is_base else _APEX_COLOR
        grp_label  = 'B' if is_base else 'A'
        center_val = int(vol0[ix, iy, iz])

        # Template bounds (clamped)
        tx0 = max(ix - hx, 0); tx1 = min(ix + hx + 1, nx)
        ty0 = max(iy - hy, 0); ty1 = min(iy + hy + 1, ny)
        tz0 = max(iz - hz, 0); tz1 = min(iz + hz + 1, nz)

        # ── Row 0 : XY axial  (z = iz)  col=ix  row=iy ──────────────────────
        sl_xy  = vol0[:, :, iz].T          # (ny, nx)
        msk_xy = mask_info.mask_3d[:, :, iz].T.astype(np.float32)
        _draw_slice(axes[0, col], sl_xy, msk_xy,
                    seed_col=ix, seed_row=iy,
                    box_col0=tx0, box_row0=ty0,
                    box_dcol=tx1-tx0, box_drow=ty1-ty0,
                    grp_color=grp_color)
        axes[0, col].set_title(
            f'[{grp_label}] s{si+1}  iz={iz}  {center_val}/255',
            fontsize=6.5, color=grp_color, pad=2)

        # ── Row 1 : XZ coronal (y = iy)  col=ix  row=iz ─────────────────────
        sl_xz  = vol0[:, iy, :].T          # (nz, nx)
        msk_xz = mask_info.mask_3d[:, iy, :].T.astype(np.float32)
        _draw_slice(axes[1, col], sl_xz, msk_xz,
                    seed_col=ix, seed_row=iz,
                    box_col0=tx0, box_row0=tz0,
                    box_dcol=tx1-tx0, box_drow=tz1-tz0,
                    grp_color=grp_color)

        # ── Row 2 : YZ sagittal (x = ix)  col=iy  row=iz ────────────────────
        sl_yz  = vol0[ix, :, :].T          # (nz, ny)
        msk_yz = mask_info.mask_3d[ix, :, :].T.astype(np.float32)
        _draw_slice(axes[2, col], sl_yz, msk_yz,
                    seed_col=iy, seed_row=iz,
                    box_col0=ty0, box_row0=tz0,
                    box_dcol=ty1-ty0, box_drow=tz1-tz0,
                    grp_color=grp_color)

        # ── Row 3 : X/Y intensity profiles ──────────────────────────────────
        CROP   = 12
        x0 = max(ix - CROP, 0); x1 = min(ix + CROP + 1, nx)
        y0 = max(iy - CROP, 0); y1 = min(iy + CROP + 1, ny)

        x_rel  = np.arange(x0, x1) - ix
        x_norm = vol0[x0:x1, iy, iz].astype(np.float32) / 255.0
        y_rel  = np.arange(y0, y1) - iy
        y_norm = vol0[ix, y0:y1, iz].astype(np.float32) / 255.0

        ax3 = axes[3, col]
        ax3.plot(x_rel, x_norm, color='#2255AA', lw=1.4, label='X-cut')
        ax3.plot(y_rel, y_norm, color='#AA3322', lw=1.4, ls='--', label='Y-cut')
        ax3.axvline(0, color='#888', lw=0.7, ls=':')
        ax3.set_xlim(-CROP, CROP)
        ax3.set_ylim(-0.05, 1.05)
        ax3.set_xlabel('Δ px', fontsize=6, labelpad=1)
        ax3.tick_params(labelsize=5)
        ax3.set_yticks([0, 0.5, 1.0])
        ax3.set_yticklabels(['0', '.5', '1'], fontsize=5)
        ax3.grid(True, alpha=0.18, linestyle=':')
        ax3.set_facecolor('#f8f8f8')
        for sp in ax3.spines.values():
            sp.set_edgecolor(grp_color); sp.set_linewidth(1.6)
        if col == 0:
            ax3.legend(fontsize=5, loc='upper right', framealpha=0.7,
                       handlelength=1.2)

    fig.suptitle(
        'Seed selection  —  3 orthogonal views + intensity profiles  (frame 0)\n'
        'Red overlay = mask  ·  Yellow box = template  ·  Cyan + = seed centre  ·  '
        'Blue border = Base  ·  Red border = Apex',
        fontsize=9, y=1.01,
    )

    out = f'{out_dir}/seed_selection_scores.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {out}')


def _viz_seed_templates(
    vol0      : np.ndarray,   # (nx, ny, nz) uint8 — frame-0 volume
    seeds     : np.ndarray,   # (N, 3) int — (ix, iy, iz)
    mask_info,
    t_half    : tuple,        # (hx, hy, hz)
    out_dir   : str,
    n_show    : int = 12,
    base_mask : np.ndarray | None = None,
    apex_mask : np.ndarray | None = None,
):
    """
    For representative seeds, show three orthogonal cross-sections through
    the seed centre, with the actual segmentation mask overlaid.

    Coordinate convention
    ─────────────────────
    vol0 shape : (nx, ny, nz)   — indices (ix, iy, iz)
    After .T on a 2-D slice the array becomes (rows=iy, cols=ix), so
    imshow displays:  x-axis = ix direction,  y-axis = iy direction.
    Seed/patch coordinates are placed accordingly.

    Layout per row (one seed):
      col 0 : XY axial   slice (z=iz)   — US image + mask overlay + red box + cross
      col 1 : XZ coronal slice (y=iy)   — same
      col 2 : YZ sagittal slice (x=ix)  — same
      col 3 : XY template patch  (9×9 pixels)
      col 4 : XZ template patch  (9×5 pixels)
      col 5 : YZ template patch  (9×5 pixels)
    """
    hx, hy, hz = t_half
    N = len(seeds)
    nx, ny, nz = vol0.shape

    # Pick seeds spread evenly across z-depth
    z_order  = np.argsort(seeds[:, 2])
    idx_lin  = np.round(np.linspace(0, N - 1, min(n_show, N))).astype(int)
    show_idx = z_order[idx_lin].tolist()
    n_rows   = len(show_idx)

    fig, axes = plt.subplots(n_rows, 6,
                             figsize=(13, 2.1 * n_rows),
                             facecolor='white')
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    col_titles = ['XY (axial, z=iz)', 'XZ (coronal, y=iy)', 'YZ (sagittal, x=ix)',
                  'XY patch', 'XZ patch', 'YZ patch']
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=8, pad=4)

    for row, si in enumerate(show_idx):
        ix, iy, iz = int(seeds[si, 0]), int(seeds[si, 1]), int(seeds[si, 2])

        # Template patch bounds (clamped to volume)
        x0, x1 = max(ix - hx, 0), min(ix + hx + 1, nx)
        y0, y1 = max(iy - hy, 0), min(iy + hy + 1, ny)
        z0, z1 = max(iz - hz, 0), min(iz + hz + 1, nz)

        # ── Full cross-section slices ─────────────────────────────────────────
        # vol0[:, :, iz]  shape (nx, ny)  → .T → (ny, nx)  row=iy  col=ix
        sl_xy  = vol0[:, :, iz].T
        msk_xy = mask_info.mask_3d[:, :, iz].T.astype(np.float32)

        # vol0[:, iy, :]  shape (nx, nz)  → .T → (nz, nx)  row=iz  col=ix
        sl_xz  = vol0[:, iy, :].T
        msk_xz = mask_info.mask_3d[:, iy, :].T.astype(np.float32)

        # vol0[ix, :, :]  shape (ny, nz)  → .T → (nz, ny)  row=iz  col=iy
        sl_yz  = vol0[ix, :, :].T
        msk_yz = mask_info.mask_3d[ix, :, :].T.astype(np.float32)

        # ── Template patches (extracted directly) ─────────────────────────────
        patch_xy = vol0[x0:x1, y0:y1, iz].T       # (y-ext, x-ext)
        patch_xz = vol0[x0:x1, iy,   z0:z1].T     # (z-ext, x-ext)
        patch_yz = vol0[ix,   y0:y1, z0:z1].T     # (z-ext, y-ext)

        # ── Helper: draw one full-slice panel ─────────────────────────────────
        def _draw_ctx(ax, sl, msk, seed_col, seed_row,
                      rect_col0, rect_row0, rect_dcol, rect_drow):
            ax.imshow(sl, cmap='gray', origin='lower', aspect='auto',
                      vmin=0, vmax=255)
            # Actual segmentation mask — semi-transparent red overlay
            if msk.any():
                ax.imshow(np.where(msk > 0, 1.0, np.nan),
                          cmap='Reds', vmin=0, vmax=1,
                          alpha=0.35, origin='lower', aspect='auto')
            # Template bounding box (yellow)
            rect = plt.Rectangle((rect_col0 - 0.5, rect_row0 - 0.5),
                                  rect_dcol, rect_drow,
                                  edgecolor='yellow', facecolor='none',
                                  lw=1.2, zorder=4)
            ax.add_patch(rect)
            # Seed centre cross (cyan)
            ax.plot(seed_col, seed_row, '+', color='cyan',
                    ms=7, mew=1.5, zorder=5)
            ax.tick_params(labelsize=6)

        # XY: col=ix, row=iy
        _draw_ctx(axes[row, 0], sl_xy, msk_xy,
                  seed_col=ix, seed_row=iy,
                  rect_col0=x0, rect_row0=y0,
                  rect_dcol=x1-x0, rect_drow=y1-y0)
        if base_mask is not None and base_mask[si]:
            grp_lbl = 'B'; grp_clr = _BASE_COLOR
        elif apex_mask is not None and apex_mask[si]:
            grp_lbl = 'A'; grp_clr = _APEX_COLOR
        else:
            grp_lbl = '?'; grp_clr = '#888888'
        axes[row, 0].set_ylabel(
            f's{si+1} [{grp_lbl}]  iz={iz}', fontsize=7, color=grp_clr)

        # XZ: col=ix, row=iz
        _draw_ctx(axes[row, 1], sl_xz, msk_xz,
                  seed_col=ix, seed_row=iz,
                  rect_col0=x0, rect_row0=z0,
                  rect_dcol=x1-x0, rect_drow=z1-z0)

        # YZ: col=iy, row=iz
        _draw_ctx(axes[row, 2], sl_yz, msk_yz,
                  seed_col=iy, seed_row=iz,
                  rect_col0=y0, rect_row0=z0,
                  rect_dcol=y1-y0, rect_drow=z1-z0)

        # ── Template patches ─────────────────────────────────────────────────
        for j, (patch, lbl) in enumerate([
            (patch_xy, f'{patch_xy.shape[1]}×{patch_xy.shape[0]} px'),
            (patch_xz, f'{patch_xz.shape[1]}×{patch_xz.shape[0]} px'),
            (patch_yz, f'{patch_yz.shape[1]}×{patch_yz.shape[0]} px'),
        ]):
            ax = axes[row, 3 + j]
            if patch.size > 0:
                ax.imshow(patch, cmap='gray', origin='lower',
                          vmin=0, vmax=255, interpolation='nearest',
                          aspect='equal')
                ax.set_xlabel(lbl, fontsize=6, labelpad=2)
            else:
                ax.text(0.5, 0.5, '(edge)', ha='center', va='center',
                        fontsize=7, transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor('yellow'); sp.set_linewidth(1.0)

    fig.suptitle(
        f'Seed cross-sections & template patches  (frame 0,  template {2*hx+1}×{2*hy+1}×{2*hz+1} vx)\n'
        r'Red overlay = segmentation ROI   Yellow box = template   Cyan + = seed centre',
        fontsize=9, y=1.01,
    )
    plt.tight_layout()
    out = f'{out_dir}/seed_templates.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {out}')


def _viz_tracking_gif(
    vol4d     : np.ndarray,   # (nx, ny, nz, nt) uint8
    traj      : np.ndarray,   # (nt, N, 3)
    valid     : np.ndarray,   # (nt, N) bool
    seeds     : np.ndarray,   # (N, 3) int
    out_dir   : str,
    base_mask : np.ndarray | None = None,
    apex_mask : np.ndarray | None = None,
    fps       : float = 4.0,   # slow playback
    trail_len : int   = 5,     # past frames to show as fading trail
):
    """
    Generate one animated GIF per anatomical group (base / apex).

    Each frame shows:
      • Background : XY axial US slice at the group's representative z, playing
                     at `fps` frames/s (slow so motion is clearly visible).
      • Seed dots  : filled circle at the current tracked (ix, iy) position of
                     every seed in the group.
      • Trail      : fading line of the last `trail_len` positions so the
                     direction of motion is immediately readable.
      • Frame-0 diamond: fixed open marker showing where each seed started.

    Representative z is the median z-coordinate of the group's seeds at frame 0.
    """
    import matplotlib.animation as _anim

    nt, N, _ = traj.shape
    nx, ny, nz, _ = vol4d.shape

    groups = []
    if base_mask is not None and base_mask.any():
        groups.append(('base', np.where(base_mask)[0], _BASE_COLOR))
    if apex_mask is not None and apex_mask.any():
        groups.append(('apex', np.where(apex_mask)[0], _APEX_COLOR))

    for grp_name, idx, grp_color in groups:
        z_rep = int(np.median(seeds[idx, 2]))

        fig, ax = plt.subplots(figsize=(5, 5), facecolor='black')
        ax.set_facecolor('black')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

        # ── Static frame-0 reference markers ─────────────────────────────────
        for i in idx:
            ax.plot(seeds[i, 0], seeds[i, 1], 'D',
                    ms=5, mfc='none', mec=grp_color,
                    mew=1.0, alpha=0.6, zorder=3)

        # ── Animated objects ──────────────────────────────────────────────────
        sl0 = vol4d[:, :, z_rep, 0].T
        im  = ax.imshow(sl0, cmap='gray', origin='lower', aspect='auto',
                        vmin=0, vmax=255, animated=True)

        dot_objs   = []
        trail_objs = []
        for _ in idx:
            d, = ax.plot([], [], 'o', color=grp_color, ms=7,
                         markeredgecolor='white', markeredgewidth=0.6, zorder=5)
            tr, = ax.plot([], [], '-', color=grp_color, alpha=0.45, lw=1.4, zorder=4)
            dot_objs.append(d)
            trail_objs.append(tr)

        txt = ax.text(0.02, 0.97, '', transform=ax.transAxes,
                      fontsize=9, color='white', va='top',
                      fontfamily='monospace')
        title_txt = ax.set_title(
            f'{grp_name.capitalize()} group  (z≈{z_rep})  —  '
            f'N={len(idx)} seeds',
            fontsize=9, color=grp_color, pad=4)

        def _update(t, idx=idx):
            im.set_data(vol4d[:, :, z_rep, t].T)
            for k, i in enumerate(idx):
                # Current dot
                if valid[t, i]:
                    dot_objs[k].set_data([traj[t, i, 0]], [traj[t, i, 1]])
                else:
                    dot_objs[k].set_data([], [])
                # Trail: last trail_len valid positions
                t0 = max(0, t - trail_len)
                tx = [traj[tt, i, 0] for tt in range(t0, t + 1) if valid[tt, i]]
                ty = [traj[tt, i, 1] for tt in range(t0, t + 1) if valid[tt, i]]
                trail_objs[k].set_data(tx, ty)
            txt.set_text(f'frame {t:02d}/{nt-1:02d}')
            return [im] + dot_objs + trail_objs + [txt]

        ani = _anim.FuncAnimation(
            fig, _update, frames=nt,
            interval=int(1000 / fps), blit=True)

        out = f'{out_dir}/tracking_{grp_name}.gif'
        ani.save(out, writer=_anim.PillowWriter(fps=fps))
        plt.close(fig)
        print(f'Saved → {out}')


def _viz_twist(
    twist     : np.ndarray,
    twist_std : np.ndarray,
    rot_base  : np.ndarray,
    rot_apex  : np.ndarray,
    time_axis : np.ndarray,
    N         : int,
    out_dir   : str,
    n_base    : int = 0,
    n_apex    : int = 0,
):
    from scipy.signal import savgol_filter as _sg

    nt   = len(time_axis)
    pk_i = int(np.argmax(np.abs(twist)))

    # Extra-smooth versions just for display (wider window than computation)
    def _extra_smooth(arr, wl=13):
        wl = min(wl, nt)
        if wl % 2 == 0: wl -= 1
        if wl >= 5 and nt > wl:
            out = _sg(arr, wl, 3)
            out[0] = 0.0
            return out
        return arr.copy()

    rb_sm  = _extra_smooth(rot_base)
    ra_sm  = _extra_smooth(rot_apex)
    tw_sm  = _extra_smooth(twist)
    pk_sm  = int(np.argmax(np.abs(tw_sm)))

    fig, (ax_comp, ax_twist) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True,
        gridspec_kw={'hspace': 0.10})
    fig.patch.set_facecolor('white')

    # ── Top panel: base & apex rotations ─────────────────────────────────────
    # Raw frames as faint scatter; smoothed as bold line
    ax_comp.scatter(time_axis, rot_base, s=14, color=_BASE_COLOR, alpha=0.35, zorder=2)
    ax_comp.scatter(time_axis, rot_apex, s=14, color=_APEX_COLOR, alpha=0.35, zorder=2)
    ax_comp.plot(time_axis, rb_sm, '-', color=_BASE_COLOR, lw=2.2,
                 label=f'Base  (n={n_base})')
    ax_comp.plot(time_axis, ra_sm, '-', color=_APEX_COLOR, lw=2.2,
                 label=f'Apex  (n={n_apex})')
    ax_comp.axhline(0, color='#888', lw=0.8, ls='--')
    ax_comp.set_ylabel('Rotation  (°)', fontsize=10)
    ax_comp.set_title(
        f'Base & Apex rotations  ·  3D NCC  (N={N} seeds, {nt} frames)',
        fontsize=11, pad=8)
    ax_comp.legend(fontsize=9, framealpha=0.7)
    ax_comp.grid(True, alpha=0.18, linestyle=':')
    ax_comp.set_facecolor('#fafafa')

    # ── Bottom panel: twist = apex − base ────────────────────────────────────
    # Faint raw points, bold smooth line, shaded ±SD band
    ax_twist.scatter(time_axis, twist, s=14, color='mediumpurple',
                     alpha=0.35, zorder=2, label='Per-frame estimate')
    ax_twist.fill_between(time_axis,
                          tw_sm - twist_std, tw_sm + twist_std,
                          alpha=0.18, color='mediumpurple')
    ax_twist.plot(time_axis, tw_sm, '-', color='#5B2C8D', lw=2.8,
                  label='Smoothed twist')
    ax_twist.axhline(0, color='#888', lw=0.8, ls='--')

    # Peak marker
    ax_twist.axvline(time_axis[pk_sm], color='#C00', lw=1.2, ls=':')
    ax_twist.scatter([time_axis[pk_sm]], [tw_sm[pk_sm]],
                     color='#C00', s=80, zorder=6,
                     label=f'Peak  {tw_sm[pk_sm]:+.2f}°  @  {time_axis[pk_sm]:.2f} s')

    # Fill area under curve to emphasise U-shape
    ax_twist.fill_between(time_axis, 0, tw_sm,
                          where=(tw_sm > 0), alpha=0.12, color='mediumpurple')

    ax_twist.set_xlabel('Time  (s)', fontsize=10)
    ax_twist.set_ylabel('Twist  (°)', fontsize=10)
    ax_twist.set_title('Global Cardiac Twist  =  Apex − Base', fontsize=11, pad=8)
    ax_twist.legend(fontsize=9, framealpha=0.7)
    ax_twist.grid(True, alpha=0.18, linestyle=':')
    ax_twist.set_facecolor('#fafafa')

    plt.tight_layout()
    out = f'{out_dir}/twist_global.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {out}')


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Load volume ───────────────────────────────────────────────────────────
    print('Loading us4d.npz …')
    npz   = np.load(VOL_NPZ)
    vol4d = npz['data']        # (nx, ny, nz, nt) uint8
    scale = npz['scale']       # [dx, dy, dz] mm/voxel
    nt    = vol4d.shape[3]
    time_axis = np.arange(nt) / FRAME_RATE
    print(f'  vol shape : {vol4d.shape}   scale: {scale}')

    # ── Load segmentation ─────────────────────────────────────────────────────
    print('Loading segmentation …')
    mask_info = load_mask_3d(SEG_NRRD)

    vol0 = vol4d[..., 0]
    vol1 = vol4d[..., 1] if USE_PSR else None

    # ── Anatomical zone boundaries ────────────────────────────────────────────
    z_lo_m, z_hi_m = mask_info.z_lo, mask_info.z_hi
    lv_len_m       = z_hi_m - z_lo_m
    z_base_hi_m    = int(z_lo_m + lv_len_m * BASE_FRAC[1])
    z_apex_lo_m    = int(z_lo_m + lv_len_m * APEX_FRAC[0])

    print(f'\nCONFIG  n_seeds_per_group={N_SEEDS_PER_GROUP}  min_dist={MIN_DIST_MM} mm  '
          f'z_levels={N_Z_LEVELS}  sectors={N_SECTORS}  '
          f'template={tuple(2*h+1 for h in T_HALF)}  '
          f'search_base={tuple(2*h+1 for h in S_HALF_BASE)}  '
          f'search_apex={tuple(2*h+1 for h in S_HALF_APEX)}  '
          f'use_psr={USE_PSR}\n'
          f'  base z∈[{z_lo_m},{z_base_hi_m}]  '
          f'apex z∈[{z_apex_lo_m},{z_hi_m}]  '
          f'middle excluded\n')

    # ── 1. Anatomically targeted seed selection ───────────────────────────────
    # Seeds are selected separately from the basal and apical thirds so that
    # both groups are adequately sampled.  The middle third is excluded from
    # selection entirely — those seeds are never computed or visualised.
    print(f'Selecting {N_SEEDS_PER_GROUP} base seeds (z∈[{z_lo_m},{z_base_hi_m}]) …')
    t0 = _time.time()
    seeds_base = select_seeds_3d_global(
        vol0, mask_info, scale,
        n_seeds     = N_SEEDS_PER_GROUP,
        min_dist_mm = MIN_DIST_MM,
        vol1        = vol1,
        n_z_levels  = N_Z_LEVELS,
        n_sectors   = N_SECTORS,
        z_lo_limit  = z_lo_m,
        z_hi_limit  = z_base_hi_m,
    )
    print(f'  → {len(seeds_base)} base seeds  ({_time.time()-t0:.1f} s)')

    print(f'Selecting {N_SEEDS_PER_GROUP} apex seeds (z∈[{z_apex_lo_m},{z_hi_m}]) …')
    t0 = _time.time()
    seeds_apex = select_seeds_3d_global(
        vol0, mask_info, scale,
        n_seeds     = N_SEEDS_PER_GROUP,
        min_dist_mm = MIN_DIST_MM,
        vol1        = vol1,
        n_z_levels  = N_Z_LEVELS,
        n_sectors   = N_SECTORS,
        z_lo_limit  = z_apex_lo_m,
        z_hi_limit  = z_hi_m,
    )
    print(f'  → {len(seeds_apex)} apex seeds  ({_time.time()-t0:.1f} s)')

    seeds = np.concatenate([seeds_base, seeds_apex], axis=0)
    N     = len(seeds)
    # Group membership arrays (length N)
    base_mask_sel = np.zeros(N, dtype=bool)
    apex_mask_sel = np.zeros(N, dtype=bool)
    base_mask_sel[:len(seeds_base)] = True
    apex_mask_sel[len(seeds_base):] = True

    if N == 0:
        print('ERROR: no seeds found — check mask / parameters.')
        return

    # Print seed positions
    for i, (ix, iy, iz) in enumerate(seeds):
        cx, cy = mask_info.centre_at(iz)
        r = np.sqrt((ix - cx)**2 + (iy - cy)**2)
        print(f'  seed {i+1:2d}  (ix={ix:3d}, iy={iy:3d}, iz={iz:3d})  '
              f'r={r:.1f} px')

    # ── 2. 3D Direct NCC tracking ─────────────────────────────────────────────
    # Base and apex are tracked separately with different search windows:
    #   base  → smaller window (17×17×9) because the basal wall is thin and a
    #            large search region causes the tracker to jump to false matches.
    #   apex  → original larger window (31×31×17).
    n_base_seeds = int(base_mask_sel.sum())
    n_apex_seeds = int(apex_mask_sel.sum())

    print(f'\nTracking {n_base_seeds} base seeds  (s_half={S_HALF_BASE}) …')
    t0 = _time.time()
    traj_b, ncc_b = track_3d_direct(
        vol4d, seeds[base_mask_sel], t_half=T_HALF, s_half=S_HALF_BASE, verbose=True,
    )
    print(f'  done  ({_time.time()-t0:.1f} s)')

    print(f'Tracking {n_apex_seeds} apex seeds  (s_half={S_HALF_APEX}) …')
    t0 = _time.time()
    traj_a, ncc_a = track_3d_direct(
        vol4d, seeds[apex_mask_sel], t_half=T_HALF, s_half=S_HALF_APEX, verbose=True,
    )
    print(f'  done  ({_time.time()-t0:.1f} s)')

    # Re-assemble in original base-first order (base_mask_sel comes first)
    traj      = np.concatenate([traj_b, traj_a], axis=1)   # (nt, N, 3)
    ncc_peaks = np.concatenate([ncc_b,  ncc_a],  axis=1)   # (nt, N)

    # ── 3. Validity ───────────────────────────────────────────────────────────
    # Use a stricter NCC threshold for base seeds (thinner wall → more ambiguity)
    valid_b = compute_validity_3d(
        traj_b, ncc_b, mask_info,
        ncc_thresh=NCC_THRESH_BASE, ring_tol_px=RING_TOL_PX,
    )
    valid_a = compute_validity_3d(
        traj_a, ncc_a, mask_info,
        ncc_thresh=NCC_THRESH, ring_tol_px=RING_TOL_PX,
    )
    valid = np.concatenate([valid_b, valid_a], axis=1)   # (nt, N)

    valid_frac = valid[1:].mean()
    print(f'  overall validity : {valid_frac:.1%}')
    print(f'  mean NCC base (t>0) : {ncc_b[1:].mean():.3f}')
    print(f'  mean NCC apex (t>0) : {ncc_a[1:].mean():.3f}')

    # ── 3b. Drop seeds with poor tracking quality ─────────────────────────────
    # A seed that is invalid for the majority of frames contributes noise rather
    # than signal and distorts the twist curve.  Remove it entirely.
    per_seed_valid = valid[1:].mean(axis=0)   # (N,) fraction of valid frames
    keep = per_seed_valid >= MIN_VALID_FRAC
    n_drop = int((~keep).sum())
    if n_drop > 0:
        print(f'  dropped {n_drop} seeds with valid_frac < {MIN_VALID_FRAC:.0%}')
        seeds         = seeds[keep]
        traj          = traj[:, keep, :]
        ncc_peaks     = ncc_peaks[:, keep]
        valid         = valid[:, keep]
        base_mask_sel = base_mask_sel[keep]
        apex_mask_sel = apex_mask_sel[keep]
        N             = len(seeds)
    print(f'  seeds after quality filter : {N}  '
          f'(base={int(base_mask_sel.sum())}, apex={int(apex_mask_sel.sum())})')

    # ── 4. Global twist (base-apex method) ───────────────────────────────────
    print('\nComputing global twist …')
    twist, twist_std, rot_base, rot_apex = compute_twist_3d(traj, valid, mask_info)

    pk_i = int(np.argmax(np.abs(twist)))
    print(f'  peak twist : {twist[pk_i]:+.2f}°  @  t={time_axis[pk_i]:.2f} s')

    # ── 5. Save numerical results ─────────────────────────────────────────────
    out_npz = f'{OUT_DIR}/twist_3d.npz'
    np.savez_compressed(
        out_npz,
        time_axis  = time_axis,
        seeds      = seeds,
        trajectories = traj,
        ncc_peaks  = ncc_peaks,
        valid      = valid,
        twist      = twist,
        twist_std  = twist_std,
        rot_base   = rot_base,
        rot_apex   = rot_apex,
    )
    print(f'Saved → {out_npz}')

    # ── 6a. Visualise seed selection scoring ─────────────────────────────────
    print('\nRendering seed selection score figure …')
    _viz_seed_selection(vol0, seeds, mask_info, OUT_DIR,
                        base_mask=base_mask_sel, apex_mask=apex_mask_sel)

    # ── 6b. Visualise seed cross-sections & template patches ─────────────────
    print('\nRendering seed template figure …')
    _viz_seed_templates(vol0, seeds, mask_info, T_HALF, OUT_DIR, n_show=N,
                        base_mask=base_mask_sel, apex_mask=apex_mask_sel)

    # ── 7. Visualise 3D trajectories ──────────────────────────────────────────
    print('\nRendering 3D trajectory plot …')
    _viz_trajectories_3d(
        traj, valid, seeds, ncc_peaks,
        mask_info, scale, time_axis, OUT_DIR,
        base_mask=base_mask_sel, apex_mask=apex_mask_sel,
    )

    # ── 7b. Animated tracking GIFs ────────────────────────────────────────────
    print('\nRendering tracking GIFs …')
    _viz_tracking_gif(
        vol4d, traj, valid, seeds, OUT_DIR,
        base_mask=base_mask_sel, apex_mask=apex_mask_sel,
        fps=4.0, trail_len=5,
    )

    # ── 8. Visualise twist curve ──────────────────────────────────────────────
    _viz_twist(twist, twist_std, rot_base, rot_apex, time_axis, N, OUT_DIR,
               n_base=int(base_mask_sel.sum()), n_apex=int(apex_mask_sel.sum()))

    print('\nAll done.')


if __name__ == '__main__':
    main()
