"""
Speckle Tracking Twist Visualization
Shows the local block-matching algorithm step by step for cardiac twist measurement.
"""

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.signal import fftconvolve

# ─────────────────────────────────────────────
# Core algorithm
# ─────────────────────────────────────────────

def ncc_search(template, search_region):
    """Normalised cross-correlation between template and search region."""
    t = template - template.mean()
    s = search_region - search_region.mean()
    t_std = t.std()
    if t_std < 1e-6:
        return np.zeros((
            search_region.shape[0] - template.shape[0] + 1,
            search_region.shape[1] - template.shape[1] + 1,
        ))
    t_norm = t / (t_std * t.size)
    corr = fftconvolve(s, t_norm[::-1, ::-1], mode="valid")
    s_std = s.std()
    if s_std > 1e-6:
        corr /= s_std
    return corr


def track_points(frame_a, frame_b, points, patch_r=8, search_r=16):
    """
    Track a list of (x, y) seed points from frame_a to frame_b.

    Returns
    -------
    disp : (N, 2) array  dx, dy in pixels
    ncc_peak : (N,) array  peak NCC value (quality metric)
    """
    H, W = frame_a.shape
    disps, peaks = [], []
    for (px, py) in points:
        # patch in frame A
        x0, x1 = px - patch_r, px + patch_r + 1
        y0, y1 = py - patch_r, py + patch_r + 1
        if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
            disps.append((0.0, 0.0)); peaks.append(0.0)
            continue
        tmpl = frame_a[y0:y1, x0:x1].astype(float)

        # search window in frame B
        sx0 = max(0, x0 - search_r)
        sy0 = max(0, y0 - search_r)
        sx1 = min(W, x1 + search_r)
        sy1 = min(H, y1 + search_r)
        region = frame_b[sy0:sy1, sx0:sx1].astype(float)

        corr = ncc_search(tmpl, region)
        if corr.size == 0:
            disps.append((0.0, 0.0)); peaks.append(0.0)
            continue

        peak_idx = np.unravel_index(corr.argmax(), corr.shape)
        # displacement relative to original position
        dy = (sy0 + peak_idx[0]) - y0
        dx = (sx0 + peak_idx[1]) - x0
        disps.append((dx, dy))
        peaks.append(corr.max())

    return np.array(disps), np.array(peaks)


def compute_twist(points, disps, center):
    """
    Convert displacement vectors to tangential (twist) angle change.

    For each point P at angle θ from center, the tangential displacement
    component gives Δθ = arctan(tangential / radius).
    """
    cx, cy = center
    angles, radii, delta_theta = [], [], []
    for (px, py), (dx, dy) in zip(points, disps):
        rx, ry = px - cx, py - cy
        r = np.hypot(rx, ry)
        if r < 1e-3:
            angles.append(0); radii.append(0); delta_theta.append(0)
            continue
        # unit tangential vector (CCW)
        tx, ty = -ry / r, rx / r
        tangential = dx * tx + dy * ty      # pixels
        dtheta = np.degrees(np.arctan2(tangential, r))
        angles.append(np.degrees(np.arctan2(ry, rx)))
        radii.append(r)
        delta_theta.append(dtheta)
    return np.array(angles), np.array(radii), np.array(delta_theta)


# ─────────────────────────────────────────────
# Seed point grid (annular ring on myocardium)
# ─────────────────────────────────────────────

def make_annular_seeds(center, r_inner, r_outer, n_angles=24):
    """Evenly spaced points in an annulus around the LV center."""
    cx, cy = center
    pts = []
    for a in np.linspace(0, 2 * np.pi, n_angles, endpoint=False):
        for r in np.linspace(r_inner, r_outer, 3):
            x = int(round(cx + r * np.cos(a)))
            y = int(round(cy + r * np.sin(a)))
            pts.append((x, y))
    return pts


# ─────────────────────────────────────────────
# Main visualization
# ─────────────────────────────────────────────

def main():
    # ── load CLAHE data ──────────────────────────
    img  = nib.load("slice_z104_clahe.nii.gz")
    data = np.asarray(img.dataobj)[:, :, 0, :]   # (nx, ny, nt)
    dx_mm, dy_mm = float(img.header.get_zooms()[0]), float(img.header.get_zooms()[1])

    # Display as (row=y, col=x)
    fa = data[:, :, 0].T.astype(float)   # frame t=0
    fb = data[:, :, 1].T.astype(float)   # frame t=1
    H, W = fa.shape

    # ── LV center (estimated from image centroid of bright pixels) ──
    bright = fa > 160
    if bright.sum() > 0:
        ys, xs = np.where(bright)
        cx_auto, cy_auto = int(xs.mean()), int(ys.mean())
    else:
        cx_auto, cy_auto = W // 2, H // 2

    # Manual tweak: the bright ring centre for this slice
    # (blood pool is darker, myocardium is the bright ring)
    # find centroid of ring: exclude very bright pixels (specular) and very dim
    ring = (fa > 80) & (fa < 230)
    ys2, xs2 = np.where(ring)
    cx = int(xs2.mean()); cy = int(ys2.mean())
    print(f"LV centre estimate: ({cx}, {cy})")

    # ── seed points ─────────────────────────────
    # estimate inner/outer radius from ring extent
    dists = np.sqrt((xs2 - cx)**2 + (ys2 - cy)**2)
    r_inner = int(np.percentile(dists, 10))
    r_outer = int(np.percentile(dists, 60))
    print(f"Annulus radii: inner={r_inner}px, outer={r_outer}px")

    seeds = make_annular_seeds((cx, cy), r_inner, r_outer, n_angles=20)
    # filter out seeds outside FOV
    seeds = [(x, y) for (x, y) in seeds
             if 0 < x < W and 0 < y < H and fa[y, x] > 0]

    PATCH_R  = 8   # half-size of template patch
    SEARCH_R = 14  # extra search margin beyond patch

    # ── run tracking ────────────────────────────
    disps, ncc_vals = track_points(fa, fb, seeds,
                                   patch_r=PATCH_R, search_r=SEARCH_R)
    angles, radii, delta_theta = compute_twist(seeds, disps, (cx, cy))
    mean_twist = np.mean(delta_theta)
    print(f"Mean twist t0→t1: {mean_twist:.3f} deg")

    # ─────────────────────────────────────────────
    # Figure layout: 3 rows × 3 cols
    # ─────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#0e0e0e")
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35,
                          left=0.06, right=0.97, top=0.93, bottom=0.06)

    kw_img = dict(cmap="gray", origin="upper", interpolation="bilinear")
    kw_ax  = dict(facecolor="#0e0e0e")

    def style(ax, title):
        ax.set_title(title, color="white", fontsize=10, pad=5)
        ax.tick_params(colors="gray", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")

    # ── Panel 0: Frame A with seed grid ──────────
    ax0 = fig.add_subplot(gs[0, 0], **kw_ax)
    ax0.imshow(fa, vmin=0, vmax=255, **kw_img)
    for (px, py) in seeds:
        ax0.plot(px, py, ".", color="lime", ms=3, alpha=0.7)
    ax0.plot(cx, cy, "+", color="red", ms=10, mew=2)
    style(ax0, "Step 1 — Frame A + seed grid")

    # ── Panel 1: zoom on one patch + search window ──
    ex_idx = len(seeds) // 3      # pick a representative seed
    ex_x, ex_y = seeds[ex_idx]
    ax1 = fig.add_subplot(gs[0, 1], **kw_ax)
    pad = SEARCH_R + PATCH_R + 4
    zx0, zx1 = max(0, ex_x - pad), min(W, ex_x + pad)
    zy0, zy1 = max(0, ex_y - pad), min(H, ex_y + pad)
    ax1.imshow(fa[zy0:zy1, zx0:zx1], vmin=0, vmax=255,
               extent=[zx0, zx1, zy1, zy0], **kw_img)
    # search window
    sw = patches.Rectangle((ex_x - PATCH_R - SEARCH_R, ex_y - PATCH_R - SEARCH_R),
                            2*(PATCH_R+SEARCH_R), 2*(PATCH_R+SEARCH_R),
                            lw=1.5, edgecolor="yellow", facecolor="none", linestyle="--")
    # template patch
    tp = patches.Rectangle((ex_x - PATCH_R, ex_y - PATCH_R),
                            2*PATCH_R, 2*PATCH_R,
                            lw=2, edgecolor="cyan", facecolor="none")
    ax1.add_patch(sw); ax1.add_patch(tp)
    ax1.plot(ex_x, ex_y, "+", color="red", ms=10, mew=2)
    ax1.set_xlim(zx0, zx1); ax1.set_ylim(zy1, zy0)
    style(ax1, "Step 2 — Template (cyan) & search window (yellow)")

    # ── Panel 2: NCC correlation map ─────────────
    ax2 = fig.add_subplot(gs[0, 2], **kw_ax)
    x0, x1 = ex_x - PATCH_R, ex_x + PATCH_R + 1
    y0, y1 = ex_y - PATCH_R, ex_y + PATCH_R + 1
    sx0 = max(0, x0 - SEARCH_R); sy0 = max(0, y0 - SEARCH_R)
    sx1 = min(W, x1 + SEARCH_R); sy1 = min(H, y1 + SEARCH_R)
    tmpl_patch  = fa[y0:y1, x0:x1].astype(float)
    search_crop = fb[sy0:sy1, sx0:sx1].astype(float)
    corr_map = ncc_search(tmpl_patch, search_crop)
    im2 = ax2.imshow(corr_map, cmap="hot", origin="upper")
    peak_r, peak_c = np.unravel_index(corr_map.argmax(), corr_map.shape)
    ax2.plot(peak_c, peak_r, "c*", ms=12)
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04).ax.tick_params(colors="white")
    style(ax2, "Step 3 — NCC correlation map (peak = new position)")

    # ── Panel 3: Frame B with displacement vectors ──
    ax3 = fig.add_subplot(gs[1, 0:2], **kw_ax)
    ax3.imshow(fb, vmin=0, vmax=255, **kw_img)
    # colour vectors by NCC quality
    norm_ncc = Normalize(vmin=0, vmax=1)
    cmap_q   = plt.cm.plasma
    for i, ((px, py), (ddx, ddy)) in enumerate(zip(seeds, disps)):
        q = float(ncc_vals[i])
        col = cmap_q(norm_ncc(q))
        ax3.annotate("", xy=(px + ddx, py + ddy), xytext=(px, py),
                     arrowprops=dict(arrowstyle="->", color=col,
                                     lw=1.2, mutation_scale=8))
    ax3.plot(cx, cy, "+", color="red", ms=12, mew=2)
    sm = ScalarMappable(cmap=cmap_q, norm=norm_ncc)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax3, fraction=0.025, pad=0.02)
    cb.set_label("NCC quality", color="white", fontsize=8)
    cb.ax.tick_params(colors="white")
    style(ax3, "Step 4 — Displacement vectors (colour = NCC quality)")

    # ── Panel 4: Tangential decomposition (1 example) ──
    ax4 = fig.add_subplot(gs[1, 2], **kw_ax)
    ei_x, ei_y = seeds[ex_idx]
    ei_dx, ei_dy = disps[ex_idx]
    rx, ry = ei_x - cx, ei_y - cy
    r_len  = np.hypot(rx, ry)
    # unit radial & tangential
    rad_u  = np.array([rx, ry]) / r_len
    tan_u  = np.array([-ry, rx]) / r_len   # CCW tangential
    d_vec  = np.array([ei_dx, ei_dy])
    d_rad  = np.dot(d_vec, rad_u) * rad_u
    d_tan  = np.dot(d_vec, tan_u) * tan_u

    scale = 12   # arrow scale for visibility
    ax4.set_aspect("equal")
    ax4.set_facecolor("#0e0e0e")
    # draw vectors from origin
    o = np.array([0, 0])
    ax4.annotate("", xy=d_vec * scale, xytext=o,
                 arrowprops=dict(arrowstyle="->", color="white", lw=2))
    ax4.annotate("", xy=d_rad * scale, xytext=o,
                 arrowprops=dict(arrowstyle="->", color="tomato", lw=2, ls="--"))
    ax4.annotate("", xy=d_tan * scale, xytext=o,
                 arrowprops=dict(arrowstyle="->", color="cyan", lw=2, ls="--"))
    # draw arc for Δθ
    theta_start = np.degrees(np.arctan2(ry, rx))
    arc = patches.Arc((0, 0), r_len * 0.6 * scale / r_len,
                       r_len * 0.6 * scale / r_len,
                       angle=0, theta1=theta_start,
                       theta2=theta_start + delta_theta[ex_idx] * 5,
                       color="gold", lw=2)
    ax4.add_patch(arc)
    lim = max(abs(d_vec * scale).max(), 1) * 1.6
    ax4.set_xlim(-lim, lim); ax4.set_ylim(-lim, lim)
    ax4.axhline(0, color="#333"); ax4.axvline(0, color="#333")
    ax4.legend(
        [plt.Line2D([0],[0],color="white"), plt.Line2D([0],[0],color="tomato",ls="--"),
         plt.Line2D([0],[0],color="cyan",ls="--"), plt.Line2D([0],[0],color="gold")],
        ["displacement d", "radial d_r", "tangential d_t (→ twist)", "Δθ"],
        fontsize=8, labelcolor="white", facecolor="#1a1a1a", edgecolor="#444",
        loc="upper right"
    )
    style(ax4, "Step 5 — Radial / tangential decomposition")

    # ── Panel 5: Twist angle vs circumferential angle ──
    ax5 = fig.add_subplot(gs[2, 0:2], **kw_ax)
    sc = ax5.scatter(angles, delta_theta,
                     c=ncc_vals, cmap="plasma", vmin=0, vmax=1,
                     s=40, zorder=3, alpha=0.9)
    ax5.axhline(0, color="#444", lw=1)
    ax5.axhline(mean_twist, color="gold", lw=1.5, ls="--",
                label=f"Mean Δθ = {mean_twist:.3f}°")
    ax5.set_xlabel("Circumferential angle (°)", color="white", fontsize=9)
    ax5.set_ylabel("Local twist Δθ (°)", color="white", fontsize=9)
    ax5.tick_params(colors="white")
    ax5.legend(fontsize=9, labelcolor="white", facecolor="#1a1a1a", edgecolor="#444")
    cb5 = plt.colorbar(sc, ax=ax5, fraction=0.025, pad=0.02)
    cb5.set_label("NCC quality", color="white", fontsize=8)
    cb5.ax.tick_params(colors="white")
    style(ax5, "Step 6 — Local twist angle vs position (t0 → t1)")

    # ── Panel 6: Twist over all frames ──────────
    ax6 = fig.add_subplot(gs[2, 2], **kw_ax)
    twist_curve = []
    fa_ref = data[:, :, 0].T.astype(float)
    for t in range(1, data.shape[2]):
        fb_t = data[:, :, t].T.astype(float)
        d_t, ncc_t = track_points(fa_ref, fb_t, seeds,
                                   patch_r=PATCH_R, search_r=SEARCH_R)
        _, _, dt_arr = compute_twist(seeds, d_t, (cx, cy))
        twist_curve.append(dt_arr.mean())
    ax6.plot(range(1, data.shape[2]), twist_curve,
             color="gold", lw=2, marker="o", ms=4)
    ax6.axhline(0, color="#444", lw=1)
    ax6.fill_between(range(1, data.shape[2]), twist_curve,
                     alpha=0.25, color="gold")
    ax6.set_xlabel("Frame", color="white", fontsize=9)
    ax6.set_ylabel("Mean twist Δθ (°)", color="white", fontsize=9)
    ax6.tick_params(colors="white")
    style(ax6, "Step 7 — Cumulative twist curve (all frames)")

    # ── Title ────────────────────────────────────
    fig.suptitle(
        "Speckle Tracking for Cardiac Twist Measurement  —  Local NCC Block Matching",
        color="white", fontsize=13, fontweight="bold", y=0.97
    )

    out = "speckle_tracking_viz.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
