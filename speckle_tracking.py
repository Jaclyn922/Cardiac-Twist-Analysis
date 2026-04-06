"""
NCC speckle tracking within myocardial ring mask → cardiac twist.

Pipeline:
  1. Place seed points on a regular grid inside the frame-0 ring mask
  2. Track each seed frame-to-frame using NCC template matching (skimage)
  3. Compute angular displacement of each tracked point around the LV centre
  4. Mean angular displacement across seeds = twist angle (degrees)
  5. Visualise: trajectories on every-4th frame + twist-vs-time curve
"""

import numpy as np
import nibabel as nib
from skimage.feature import match_template
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────
HALF_TEMPLATE = 8    # template half-size in pixels  (template = 17×17)
HALF_SEARCH   = 20   # search window half-size — larger for direct tracking
GRID_SPACING  = 7    # seed grid spacing in pixels
FRAME_RATE    = 15.0 # volumes / second

# ─────────────────────────────────────────────
# Load raw frames and masks
# ─────────────────────────────────────────────
raw_nii  = nib.load("slice_z104_raw.nii.gz")
mask_nii = nib.load("slice_z104_mask.nii.gz")

raw_data  = np.asarray(raw_nii.dataobj)   # (nx, ny, 1, nt)
mask_data = np.asarray(mask_nii.dataobj)  # (nx, ny, 1, nt)

nx, ny, _, nt = raw_data.shape
frames = [raw_data[:, :, 0, t].T.astype(np.float32) for t in range(nt)]
masks  = [mask_data[:, :, 0, t].T.astype(bool)       for t in range(nt)]

H, W = frames[0].shape
ht   = HALF_TEMPLATE

# ─────────────────────────────────────────────
# Seed points: regular grid inside frame-0 mask
# (keep only interior points so template never clips)
# ─────────────────────────────────────────────
ys = np.arange(ht + 1, H - ht - 1, GRID_SPACING)
xs = np.arange(ht + 1, W - ht - 1, GRID_SPACING)
grid_x, grid_y = np.meshgrid(xs, ys)
candidates = np.column_stack([grid_x.ravel(), grid_y.ravel()])

in_mask = np.array([masks[0][int(y), int(x)] for x, y in candidates])
seeds = candidates[in_mask].astype(float)   # (N, 2)  [x, y]
N = len(seeds)
print(f"Seed points: {N}  |  Grid spacing: {GRID_SPACING} px")

# LV centre: centroid of frame-0 mask
rows0, cols0 = np.where(masks[0])
cx_ref = float(cols0.mean())
cy_ref = float(rows0.mean())
print(f"LV centre (frame 0): ({cx_ref:.1f}, {cy_ref:.1f})")

# ─────────────────────────────────────────────
# NCC tracking: frame-to-frame
# ─────────────────────────────────────────────
def track_direct(frame0, frame_t, seed_pts):
    """
    Track seed_pts from frame0 directly to frame_t (no accumulation).
    Templates are always extracted from frame0 at the original seed positions.
    seed_pts : (N, 2)  original [x, y] in frame0
    Returns   : (N, 2) matched positions in frame_t
    """
    new_pts = seed_pts.copy()
    hs = HALF_SEARCH

    for i, (px, py) in enumerate(seed_pts):
        x0, y0 = int(round(px)), int(round(py))

        # Template from frame0 at original seed position
        r0, r1 = y0 - ht, y0 + ht + 1
        c0, c1 = x0 - ht, x0 + ht + 1
        if r0 < 0 or r1 > H or c0 < 0 or c1 > W:
            continue
        template = frame0[r0:r1, c0:c1]        # always same template

        # Search window in frame_t centred on original seed position
        sr0 = max(0, y0 - hs);  sr1 = min(H, y0 + hs + 1)
        sc0 = max(0, x0 - hs);  sc1 = min(W, x0 + hs + 1)
        search = frame_t[sr0:sr1, sc0:sc1]

        if search.shape[0] < template.shape[0] or search.shape[1] < template.shape[1]:
            continue

        result   = match_template(search, template, pad_input=False)
        r_b, c_b = np.unravel_index(np.argmax(result), result.shape)

        new_pts[i] = [float(sc0 + c_b + ht), float(sr0 + r_b + ht)]

    return new_pts


print("Tracking (direct: frame-0 template → each frame) …")
trajectories = [seeds.copy()]   # frame 0: seeds at original positions

for t in range(1, nt):
    pts_t = track_direct(frames[0], frames[t], seeds)
    trajectories.append(pts_t)
    print(f"  frame 00 → {t:02d}  done", end="\r")

print(f"\nTracking complete. {nt} frames, {N} seeds.")
trajectories = np.array(trajectories)   # (nt, N, 2)

# ─────────────────────────────────────────────
# Validity mask: keep only points inside the ring mask at each frame
# ─────────────────────────────────────────────
valid = np.zeros((nt, N), dtype=bool)
for t in range(nt):
    for i in range(N):
        x, y = trajectories[t, i]
        xi, yi = int(round(x)), int(round(y))
        if 0 <= yi < H and 0 <= xi < W:
            valid[t, i] = masks[t][yi, xi]

n_valid = valid.sum(axis=1)
print(f"Valid points per frame: min={n_valid.min()}  max={n_valid.max()}  mean={n_valid.mean():.1f}")

# ─────────────────────────────────────────────
# Cardiac twist (only valid points)
# ─────────────────────────────────────────────
# Angle of each point around LV centre
angles = np.arctan2(trajectories[:, :, 1] - cy_ref,
                    trajectories[:, :, 0] - cx_ref)  # (nt, N)

# Unwrap each seed's angle time-series
angles_uw = np.unwrap(angles, axis=0)

# Angular displacement from frame 0
delta = angles_uw - angles_uw[0:1, :]               # (nt, N)

# Mean twist per frame using only valid (in-mask) points
twist     = np.array([np.degrees(delta[t, valid[t]]).mean() if valid[t].any() else 0.0
                       for t in range(nt)])
twist_std = np.array([np.degrees(delta[t, valid[t]]).std()  if valid[t].sum() > 1 else 0.0
                       for t in range(nt)])

time_axis = np.arange(nt) / FRAME_RATE

peak_idx = int(np.argmax(np.abs(twist)))
print(f"Peak twist : {twist[peak_idx]:+.2f}° at t={time_axis[peak_idx]:.3f} s "
      f"(frame {peak_idx})")

# ─────────────────────────────────────────────
# Figure 1: trajectories on every-4th frame
# ─────────────────────────────────────────────
show_t = list(range(0, nt, 4))
colors = plt.cm.hsv(np.linspace(0, 1, N, endpoint=False))

fig1, axes = plt.subplots(1, len(show_t), figsize=(3.5 * len(show_t), 4.5))

for ax, t in zip(axes, show_t):
    ax.imshow(frames[t], cmap="gray", origin="upper", vmin=0, vmax=255)
    for i in range(N):
        # Only draw trajectory segments where both endpoints are in-mask
        for t0 in range(t):
            if valid[t0, i] and valid[t0 + 1, i]:
                ax.plot([trajectories[t0, i, 0], trajectories[t0 + 1, i, 0]],
                        [trajectories[t0, i, 1], trajectories[t0 + 1, i, 1]],
                        "-", color=colors[i], linewidth=0.7, alpha=0.7)
        # Current position dot: only if valid at this frame
        if valid[t, i]:
            ax.plot(trajectories[t, i, 0], trajectories[t, i, 1],
                    "o", color=colors[i], markersize=2.5)
    ax.plot(cx_ref, cy_ref, "r+", markersize=10, markeredgewidth=1.5)
    ax.set_title(f"t={t}  ({t/FRAME_RATE:.2f}s)", fontsize=8)
    ax.axis("off")

fig1.suptitle(f"Speckle trajectories — NCC  ({N} seeds)", fontsize=11)
plt.tight_layout()
fig1.savefig("speckle_trajectories.png", dpi=150, bbox_inches="tight")
plt.close(fig1)
print("Saved → speckle_trajectories.png")

# ─────────────────────────────────────────────
# Figure 2: individual seed angles + mean twist
# ─────────────────────────────────────────────
fig2, axes2 = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

# Top: individual seed angular displacements (thin, transparent)
ax_top = axes2[0]
for i in range(N):
    ax_top.plot(time_axis, np.degrees(delta[:, i]),
                color="steelblue", linewidth=0.5, alpha=0.3)
ax_top.plot(time_axis, twist, "k-", linewidth=2, label="Mean twist")
ax_top.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax_top.set_ylabel("Angular displacement (°)")
ax_top.set_title("Individual seed trajectories (blue) + mean twist (black)")
ax_top.legend(fontsize=9)
ax_top.grid(True, alpha=0.3)

# Bottom: mean twist ± 1 SD
ax_bot = axes2[1]
ax_bot.fill_between(time_axis,
                    twist - twist_std, twist + twist_std,
                    alpha=0.3, color="steelblue", label="±1 SD")
ax_bot.plot(time_axis, twist, "o-", color="steelblue",
            linewidth=2, markersize=4, label="Mean twist")
ax_bot.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax_bot.axvline(time_axis[peak_idx], color="red", linewidth=1,
               linestyle=":", label=f"Peak {twist[peak_idx]:+.1f}°")
ax_bot.set_xlabel("Time (s)")
ax_bot.set_ylabel("Twist angle (°)")
ax_bot.set_title("Cardiac twist — z=104 slice")
ax_bot.legend(fontsize=9)
ax_bot.grid(True, alpha=0.3)

fig2.suptitle("Cardiac Twist from NCC Speckle Tracking", fontsize=12)
plt.tight_layout()
fig2.savefig("cardiac_twist.png", dpi=150, bbox_inches="tight")
plt.close(fig2)
print("Saved → cardiac_twist.png")
