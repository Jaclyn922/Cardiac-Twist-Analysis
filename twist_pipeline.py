"""
Cardiac twist pipeline for an arbitrary z-slice.

Functions:
  segment_slice(vol, z)  → masks (nt, H, W), cx0, cy0, r_inner0, r_outer0
  track_slice(frames, masks, cx0, cy0) → trajectories, valid, twist, twist_std
  run_slice(vol, z) → dict with twist curve + metadata

Main: runs 3 representative slices (base/mid/apex) and plots comparison.
"""

import numpy as np
import nibabel as nib
from scipy import ndimage
from skimage.morphology import closing, disk
from skimage.feature import match_template
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# Shared parameters
# ─────────────────────────────────────────────
FRAME_RATE    = 15.0
THRESH_PCT    = 90
R_TOL         = 3
CXY_TOL       = 10
HALF_TEMPLATE = 8
HALF_SEARCH   = 20
GRID_SPACING  = 7

# Heart z-range and segments (z=45–165, ~40 slices each)
Z_BASE  = (45,  84)
Z_MID   = (85,  124)
Z_APEX  = (125, 165)
SEGMENTS = {"Base": Z_BASE, "Mid": Z_MID, "Apex": Z_APEX}
REP_Z    = {name: (lo + hi) // 2 for name, (lo, hi) in SEGMENTS.items()}

# ─────────────────────────────────────────────
# Segmentation helpers (same as segment_myocardium.py)
# ─────────────────────────────────────────────

def filled_circle(shape, cx, cy, radius):
    H, W = shape
    yy, xx = np.ogrid[:H, :W]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2


def segment_frame(frame_hw, thresh_pct=THRESH_PCT):
    H, W = frame_hw.shape
    thresh = np.percentile(frame_hw, thresh_pct)
    bright = frame_hw >= thresh
    bright = closing(bright, disk(2))
    labeled, n = ndimage.label(bright)
    if n == 0:
        cx, cy = W / 2, H / 2
        return cx, cy, 20.0, 40.0

    sizes = ndimage.sum(bright, labeled, range(1, n + 1))
    cc = labeled == (np.argmax(sizes) + 1)

    rows_all, cols_all = np.where(cc)
    row_thresh = rows_all.min() + 0.50 * (rows_all.max() - rows_all.min())
    mask50 = rows_all <= row_thresh
    rows50 = rows_all[mask50].astype(float)
    cols50 = cols_all[mask50].astype(float)

    A = np.column_stack([cols50, rows50, np.ones(len(cols50))])
    z = cols50 ** 2 + rows50 ** 2
    params, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
    cx = float(params[0] / 2.0)
    cy = float(params[1] / 2.0)

    dists50 = np.sqrt((cols50 - cx) ** 2 + (rows50 - cy) ** 2)
    r_inner = float(np.percentile(dists50, 10))
    r_outer = float(np.percentile(dists50, 90))
    return cx, cy, r_inner, r_outer


def segment_slice(vol, z):
    """
    Segment all nt frames of z-slice from vol (nx, ny, nz, nt).
    Returns masks (nt, H, W bool), and reference parameters from frame 0.
    """
    nx, ny, nz, nt = vol.shape
    frames = [vol[:, :, z, t].T.astype(np.uint8) for t in range(nt)]
    H, W = frames[0].shape

    cx0, cy0, r_inner0, r_outer0 = segment_frame(frames[0])

    masks = np.zeros((nt, H, W), dtype=bool)
    for t, frame in enumerate(frames):
        cx, cy, r_inner, r_outer = segment_frame(frame)
        cx      = float(np.clip(cx,      cx0      - CXY_TOL, cx0      + CXY_TOL))
        cy      = float(np.clip(cy,      cy0      - CXY_TOL, cy0      + CXY_TOL))
        r_inner = float(np.clip(r_inner, r_inner0 - R_TOL,   r_inner0 + R_TOL))
        r_outer = float(np.clip(r_outer, r_outer0 - R_TOL,   r_outer0 + R_TOL))
        epi  = filled_circle((H, W), cx, cy, r_outer)
        endo = filled_circle((H, W), cx, cy, r_inner)
        masks[t] = epi & ~endo

    return masks, frames, cx0, cy0, r_inner0, r_outer0


# ─────────────────────────────────────────────
# Speckle tracking helpers
# ─────────────────────────────────────────────

def track_direct(frame0, frame_t, seeds, H, W):
    ht = HALF_TEMPLATE
    hs = HALF_SEARCH
    new_pts = seeds.copy()
    for i, (px, py) in enumerate(seeds):
        x0, y0 = int(round(px)), int(round(py))
        r0, r1 = y0 - ht, y0 + ht + 1
        c0, c1 = x0 - ht, x0 + ht + 1
        if r0 < 0 or r1 > H or c0 < 0 or c1 > W:
            continue
        template = frame0[r0:r1, c0:c1]
        sr0 = max(0, y0 - hs);  sr1 = min(H, y0 + hs + 1)
        sc0 = max(0, x0 - hs);  sc1 = min(W, x0 + hs + 1)
        search = frame_t[sr0:sr1, sc0:sc1]
        if search.shape[0] < template.shape[0] or search.shape[1] < template.shape[1]:
            continue
        result   = match_template(search, template, pad_input=False)
        r_b, c_b = np.unravel_index(np.argmax(result), result.shape)
        new_pts[i] = [float(sc0 + c_b + ht), float(sr0 + r_b + ht)]
    return new_pts


def track_slice(frames_list, masks_np, cx0, cy0):
    """
    frames_list : list of nt float32 (H,W) arrays
    masks_np    : (nt, H, W) bool
    Returns: trajectories (nt,N,2), valid (nt,N), twist (nt,), twist_std (nt,)
    """
    nt = len(frames_list)
    H, W = frames_list[0].shape
    ht = HALF_TEMPLATE

    # seeds: grid inside frame-0 mask, interior only
    ys = np.arange(ht + 1, H - ht - 1, GRID_SPACING)
    xs = np.arange(ht + 1, W - ht - 1, GRID_SPACING)
    gx, gy = np.meshgrid(xs, ys)
    cands = np.column_stack([gx.ravel(), gy.ravel()])
    in_m  = np.array([masks_np[0, int(y), int(x)] for x, y in cands])
    seeds = cands[in_m].astype(float)
    N = len(seeds)
    if N == 0:
        tw = np.zeros(nt)
        return None, None, tw, tw

    f0 = frames_list[0].astype(np.float32)
    trajectories = [seeds.copy()]
    for t in range(1, nt):
        ft = frames_list[t].astype(np.float32)
        trajectories.append(track_direct(f0, ft, seeds, H, W))
    trajectories = np.array(trajectories)   # (nt, N, 2)

    # validity
    valid = np.zeros((nt, N), dtype=bool)
    for t in range(nt):
        for i in range(N):
            x, y = trajectories[t, i]
            xi, yi = int(round(x)), int(round(y))
            if 0 <= yi < H and 0 <= xi < W:
                valid[t, i] = masks_np[t, yi, xi]

    # twist
    angles   = np.arctan2(trajectories[:, :, 1] - cy0,
                           trajectories[:, :, 0] - cx0)
    angles_uw = np.unwrap(angles, axis=0)
    delta     = angles_uw - angles_uw[0:1, :]

    twist     = np.array([np.degrees(delta[t, valid[t]]).mean()
                           if valid[t].any() else 0.0 for t in range(nt)])
    twist_std = np.array([np.degrees(delta[t, valid[t]]).std()
                           if valid[t].sum() > 1 else 0.0 for t in range(nt)])
    return trajectories, valid, twist, twist_std


# ─────────────────────────────────────────────
# Combined: segment + track one z-slice
# ─────────────────────────────────────────────

def run_slice(vol, z, verbose=False):
    if verbose:
        print(f"  z={z}: segmenting …", end=" ", flush=True)
    masks_np, frames_list, cx0, cy0, r_inner0, r_outer0 = segment_slice(vol, z)
    if verbose:
        print("tracking …", end=" ", flush=True)
    trajectories, valid, twist, twist_std = track_slice(
        frames_list, masks_np, cx0, cy0)
    if verbose:
        n_seeds = valid[0].sum() if valid is not None else 0
        print(f"done  (seeds={n_seeds}, peak={twist[np.argmax(np.abs(twist))]:+.2f}°)")
    return {
        "z": z,
        "twist": twist,
        "twist_std": twist_std,
        "cx0": cx0, "cy0": cy0,
        "r_inner0": r_inner0, "r_outer0": r_outer0,
    }


# ─────────────────────────────────────────────
# Main: full run — all slices in each segment
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import time as _time

    print("Loading volume …")
    nii = nib.load("us4d.nii.gz")
    vol = np.asarray(nii.dataobj)   # (nx, ny, nz, nt)
    nt  = vol.shape[3]
    time_axis = np.arange(nt) / FRAME_RATE

    seg_colors = {"Base": "steelblue", "Mid": "forestgreen", "Apex": "crimson"}

    # ── Run all slices per segment ────────────────────────────────────────────
    all_results = {}   # seg_name → list of run_slice dicts
    for seg_name, (z_lo, z_hi) in SEGMENTS.items():
        z_range = list(range(z_lo, z_hi + 1))
        print(f"\n{'='*55}")
        print(f"[{seg_name}]  z={z_lo}–{z_hi}  ({len(z_range)} slices)")
        print(f"{'='*55}")
        seg_results = []
        t0 = _time.time()
        for idx, z in enumerate(z_range):
            print(f"  [{idx+1:3d}/{len(z_range)}] z={z}", end="  ")
            res = run_slice(vol, z, verbose=True)
            seg_results.append(res)
        elapsed = _time.time() - t0
        print(f"  Segment done in {elapsed:.1f}s")
        all_results[seg_name] = seg_results

    # ── Average twist curve per segment ──────────────────────────────────────
    seg_mean = {}   # seg_name → (mean_twist (nt,), std_twist (nt,))
    for seg_name, seg_results in all_results.items():
        twist_mat = np.array([r["twist"] for r in seg_results])  # (nz_seg, nt)
        seg_mean[seg_name] = (twist_mat.mean(axis=0), twist_mat.std(axis=0))

    # ── Save results ──────────────────────────────────────────────────────────
    save_dict = {"time_axis": time_axis}
    for seg_name, (mn, sd) in seg_mean.items():
        save_dict[f"{seg_name}_mean"] = mn
        save_dict[f"{seg_name}_std"]  = sd
        # also save per-slice twist matrix
        save_dict[f"{seg_name}_all"] = np.array(
            [r["twist"] for r in all_results[seg_name]])
    np.savez_compressed("twist_all_slices.npz", **save_dict)
    print("\nSaved → twist_all_slices.npz")

    # ── Figure 1: average twist per segment ──────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    for seg_name, (mn, sd) in seg_mean.items():
        c  = seg_colors[seg_name]
        pk = mn[np.argmax(np.abs(mn))]
        n  = len(all_results[seg_name])
        axes[0].plot(time_axis, mn, "-o", color=c, markersize=4, linewidth=2,
                     label=f"{seg_name} ({n} slices)  peak={pk:+.1f}°")
        axes[1].fill_between(time_axis, mn - sd, mn + sd, alpha=0.2, color=c)
        axes[1].plot(time_axis, mn, "-o", color=c, markersize=4, linewidth=2,
                     label=f"{seg_name}  peak={pk:+.1f}°")

    for ax in axes:
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_ylabel("Twist angle (°)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    axes[0].set_title("Mean twist per segment (averaged over all z-slices)")
    axes[1].set_title("Mean twist ± slice-to-slice SD")
    axes[1].set_xlabel("Time (s)")
    fig.suptitle("Cardiac Twist — Base / Mid / Apex  (full volume)", fontsize=12)
    plt.tight_layout()
    fig.savefig("twist_segments_full.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved → twist_segments_full.png")

    # ── Figure 2: heatmap — twist(z, t) for each segment ─────────────────────
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (seg_name, seg_results) in zip(axes2, all_results.items()):
        twist_mat = np.array([r["twist"] for r in seg_results])  # (nz_seg, nt)
        z_range   = list(range(*SEGMENTS[seg_name], 1))
        im = ax.imshow(twist_mat, aspect="auto", origin="lower",
                       extent=[time_axis[0], time_axis[-1],
                                SEGMENTS[seg_name][0], SEGMENTS[seg_name][1]],
                       cmap="RdBu_r", vmin=-15, vmax=15)
        plt.colorbar(im, ax=ax, label="Twist (°)")
        ax.set_title(f"{seg_name}  z={SEGMENTS[seg_name][0]}–{SEGMENTS[seg_name][1]}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("z slice")
    fig2.suptitle("Twist heatmap  twist(z, t)  — full volume", fontsize=12)
    plt.tight_layout()
    fig2.savefig("twist_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print("Saved → twist_heatmap.png")
