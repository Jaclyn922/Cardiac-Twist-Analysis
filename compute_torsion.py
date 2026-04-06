"""
Cardiac torsion = Apex_twist - Base_twist

Two sign-flip boundaries in the data (from pipeline results):
  Base : z=85–99   (positive twist region, Mid_all indices 0-14)
  Apex : z=100–160 (negative twist region, Mid_all[15:] + Apex_all[:36])

Data in npz:
  Mid_all  : z=85–124 (indices 0..14 → z=85–99, indices 15..39 → z=100–124)
  Apex_all : z=125–165 (indices 0..35 → z=125–160)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Load and split by twist-direction zones ───────────────────────────────────
data      = np.load("twist_all_slices.npz")
time_axis = data["time_axis"]
nt        = len(time_axis)

# Base: z=85-99 (positive twist zone)
base_all = data["Mid_all"][:15]   # 15 slices

# Apex: z=100-160 (negative twist zone)
apex_all = np.vstack([
    data["Mid_all"][15:],    # z=100–124 (25 slices)
    data["Apex_all"][:36],   # z=125–160 (36 slices)
])  # shape (61, nt)

print(f"Base region (z=85–99)  : {len(base_all)} slices")
print(f"Apex region (z=100–160): {len(apex_all)} slices")

# ── Filter: keep slices consistent with the majority direction at mid-cycle ───
mid_frame = nt // 2

def filter_by_direction(arr, mid):
    vals = arr[:, mid]
    majority_neg = (vals < 0).sum() > len(vals) / 2
    mask = vals < 0 if majority_neg else vals > 0
    kept = arr[mask]
    return kept, mask

base_sel, base_mask = filter_by_direction(base_all, mid_frame)
apex_sel, apex_mask = filter_by_direction(apex_all, mid_frame)

print(f"Base kept : {base_mask.sum()}/{len(base_mask)}")
print(f"Apex kept : {apex_mask.sum()}/{len(apex_mask)}")

base_mean = base_sel.mean(axis=0)
apex_mean = apex_sel.mean(axis=0)
base_std  = base_sel.std(axis=0)
apex_std  = apex_sel.std(axis=0)

torsion     = apex_mean - base_mean
torsion_sem = np.sqrt(base_std**2 / len(base_sel) +
                      apex_std**2  / len(apex_sel))

pk_b = int(np.argmax(np.abs(base_mean)))
pk_a = int(np.argmax(np.abs(apex_mean)))
pk_t = int(np.argmax(np.abs(torsion)))

print(f"\nBase  peak : {base_mean[pk_b]:+.2f}°  at t={time_axis[pk_b]:.2f}s")
print(f"Apex  peak : {apex_mean[pk_a]:+.2f}°  at t={time_axis[pk_a]:.2f}s")
print(f"Torsion peak: {torsion[pk_t]:+.2f}°  at t={time_axis[pk_t]:.2f}s")

print("\nframe  time   base    apex   torsion")
for t in range(nt):
    print(f"  {t:2d}  {time_axis[t]:.3f}  {base_mean[t]:+.2f}  {apex_mean[t]:+.2f}  {torsion[t]:+.2f}")

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax = axes[0]
ax.fill_between(time_axis, base_mean - base_std, base_mean + base_std,
                alpha=0.15, color="steelblue")
ax.fill_between(time_axis, apex_mean - apex_std, apex_mean + apex_std,
                alpha=0.15, color="crimson")
ax.plot(time_axis, base_mean, "-o", color="steelblue", markersize=4, linewidth=2,
        label=f"Base  z=85–99   (n={len(base_sel)})  peak={base_mean[pk_b]:+.1f}°")
ax.plot(time_axis, apex_mean, "-o", color="crimson",   markersize=4, linewidth=2,
        label=f"Apex  z=100–160 (n={len(apex_sel)})  peak={apex_mean[pk_a]:+.1f}°")
ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax.set_ylabel("Twist angle (°)")
ax.set_title("Segment mean twist  —  Base (z=85–99) vs Apex (z=100–160)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.fill_between(time_axis, torsion - torsion_sem, torsion + torsion_sem,
                 alpha=0.25, color="darkorange", label="±SEM")
ax2.plot(time_axis, torsion, "-o", color="darkorange", markersize=5, linewidth=2.5,
         label="Torsion (Apex − Base)")
ax2.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax2.axvline(time_axis[pk_t], color="red", linewidth=1, linestyle=":",
            label=f"Peak {torsion[pk_t]:+.1f}°  @ {time_axis[pk_t]:.2f}s")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Torsion (°)")
ax2.set_title("Cardiac Torsion = Apex twist − Base twist")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

fig.suptitle("Cardiac Torsion Analysis  (Base z=85–99, Apex z=100–160)", fontsize=13)
plt.tight_layout()
fig.savefig("torsion.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("\nSaved → torsion.png")

# ── Heatmap: twist(z, t) for Base and Apex ───────────────────────────────────
# z-labels for each row
base_z = np.arange(85, 100)          # 15 slices → indices 0..14 of base_all
apex_z = np.arange(100, 161)         # 61 slices → Mid_all[15:] + Apex_all[:36]

vmax = max(np.abs(base_all).max(), np.abs(apex_all).max())
vmax = np.ceil(vmax)

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

# Base heatmap (z=85–99)
ax_b = axes2[0]
im_b = ax_b.imshow(base_all, aspect="auto", origin="lower",
                   extent=[time_axis[0], time_axis[-1], base_z[0] - 0.5, base_z[-1] + 0.5],
                   cmap="RdBu_r", vmin=-vmax, vmax=vmax)
ax_b.axhline(y=99.5, color="white", linewidth=1.0, linestyle="--", alpha=0.6)
plt.colorbar(im_b, ax=ax_b, label="Twist (°)")
ax_b.set_title(f"Base  z=85–99  (n={len(base_all)} slices)", fontsize=11)
ax_b.set_xlabel("Time (s)")
ax_b.set_ylabel("z slice")
# mark kept slices
for i, z in enumerate(base_z):
    if base_mask[i]:
        ax_b.add_patch(plt.Rectangle((time_axis[-1] + 0.01, z - 0.5),
                                      0.03, 1.0,
                                      color="green", clip_on=False, transform=ax_b.transData))

# Apex heatmap (z=100–160)
ax_a = axes2[1]
im_a = ax_a.imshow(apex_all, aspect="auto", origin="lower",
                   extent=[time_axis[0], time_axis[-1], apex_z[0] - 0.5, apex_z[-1] + 0.5],
                   cmap="RdBu_r", vmin=-vmax, vmax=vmax)
plt.colorbar(im_a, ax=ax_a, label="Twist (°)")
ax_a.set_title(f"Apex  z=100–160  (n={len(apex_all)} slices)", fontsize=11)
ax_a.set_xlabel("Time (s)")
ax_a.set_ylabel("z slice")
for i, z in enumerate(apex_z):
    if apex_mask[i]:
        ax_a.add_patch(plt.Rectangle((time_axis[-1] + 0.01, z - 0.5),
                                      0.03, 1.0,
                                      color="green", clip_on=False, transform=ax_a.transData))

fig2.suptitle("Twist heatmap  twist(z, t)  —  green strip = slice used in torsion", fontsize=12)
plt.tight_layout()
fig2.savefig("torsion_heatmap.png", dpi=150, bbox_inches="tight")
plt.close(fig2)
print("Saved → torsion_heatmap.png")
