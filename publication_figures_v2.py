"""
publication_figures_v2.py
=========================
Patched publication figures with:

- Smaller/tighter typography (titles 9 pt, ticks 7.5 pt, title ≤ titles+).
- Semi-transparent legends (alpha=0.55) everywhere — curves show through.
- Subplot labels (a, b, c, …) placed outside the axes frame.
- Figure 3 rebuilt as the three-metric comparison:
    row 1 twist vs ε_long, row 2 twist vs ε_short,
    cols Pearson raw | Pearson smoothed | DTW,
    shared Pearson colorbar spanning rows, separate DTW colorbar.
- Figure 2 per-region y-axes (no sharey) with 10% padding.
- Figure 6 smaller overlaid dots (size 25, alpha 0.8).
- Figure 7 annotation condensed to 2 lines, upper-right, alpha=0.55.
- SVG export for Figs 1, 3, 4, 6 in addition to PDF + PNG.
- Final verification printout.

Reads cached outputs only — no analysis recomputation.
"""

from __future__ import annotations

import csv
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter
from scipy.stats import spearmanr

import subject_metadata as smeta


# ─── Paths ───────────────────────────────────────────────────────────────────
DATA_ROOT = Path("/data/xinge/Cardiac-Twist-Analysis")
OUT_DIR   = DATA_ROOT / "figures_paper"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_SUBJECTS = ["002A", "03", "p009npa", "p009pa", "p066a",
                "p020_1a", "p030_lp2a"]
# FRAME_RATE is only the LAST-RESORT fallback.  The actual frame rate is
# resolved per-subject in subject_frame_rate(): DICOM FrameTime first, then
# us4d.npz (itself a placeholder unless overridden), then this constant.
FRAME_RATE   = 15.0
N_REGIONS    = 6


_frame_rate_cache: dict[str, tuple[float, str]] = {}


def subject_frame_rate(sid: str) -> tuple[float, str]:
    """Resolve a subject's acquisition frame rate in Hz.

    Preference order:
      1. DICOM tag (0018, 1063) FrameTime if the per-subject DICOM exists
         and has a non-zero value (real acquisition rate).
      2. The frame_rate field in us4d.npz (historically a placeholder of
         15.0 Hz from `preprocess_4d.load_4d(frame_rate=15.0)` — still
         returned so pipelines run, but flagged as such).
      3. Hardcoded 15 Hz fallback.
    """
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
            _frame_rate_cache[sid] = (fr, "us4d.npz placeholder (15 Hz default)")
            return _frame_rate_cache[sid]
        except Exception:
            pass
    _frame_rate_cache[sid] = (FRAME_RATE, "hardcoded fallback")
    return _frame_rate_cache[sid]


# ─── Global publication style (tighter typography) ──────────────────────────
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
    "legend.frameon":     True,
    "legend.framealpha":  0.55,
    "legend.edgecolor":   "0.7",
    "legend.fancybox":    False,
    "savefig.dpi":        300,
    "figure.dpi":         110,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
})

SUBJECT_COLORS = {
    "002A":     "#1f77b4",
    "03":       "#d62728",
    "p009npa":  "#2ca02c",
    "p009pa":   "#ff7f0e",
    "p066a":    "#9467bd",
    "p020_1a":  "#17becf",
    "p030_lp2a":"#8c564b",
}
REGION_CMAP         = "viridis"
COUPLING_DIVERGING  = "RdBu_r"
COUPLING_SEQUENTIAL = "YlGnBu"
MEAN_COLOR          = "#333333"
SD_COLOR            = "#CCCCCC"

SVG_FIGURES = {
    "Figure1_apex_base_gradient",
    "Figure3_coupling_comparison",
    "Figure4_xcd_subject03",
    "Figure6_feature_distributions",
}


# ─── Style helpers ───────────────────────────────────────────────────────────
def _enable_box_spines(ax) -> None:
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)


def _panel_label(ax, label: str,
                 *, dx: float = -0.07, dy: float = 1.06) -> None:
    ax.text(dx, dy, label, transform=ax.transAxes,
            fontweight="bold", fontsize=10, va="bottom", ha="left")


def _zero_line(ax) -> None:
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.6,
               alpha=0.7, zorder=1)


def _soft_ygrid(ax) -> None:
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)


def _legend_style(leg) -> None:
    if leg is None:
        return
    frame = leg.get_frame()
    frame.set_linewidth(0.5)
    frame.set_alpha(0.55)
    frame.set_edgecolor("0.7")
    frame.set_facecolor("white")


def _annotation_bbox() -> dict:
    return dict(boxstyle="round,pad=0.25",
                facecolor="white", edgecolor="0.75",
                linewidth=0.4, alpha=0.55)


def _save(fig, stem: str) -> tuple[Path, Path, Path | None]:
    pdf = OUT_DIR / f"{stem}.pdf"
    png = OUT_DIR / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.05)
    svg = None
    if stem in SVG_FIGURES:
        svg = OUT_DIR / f"{stem}.svg"
        fig.savefig(svg, format="svg", bbox_inches="tight",
                    pad_inches=0.05)
    return pdf, png, svg


def _spline_fit(t: np.ndarray, y: np.ndarray) -> UnivariateSpline | None:
    if np.std(y) < 1e-9:
        return None
    try:
        return UnivariateSpline(t, y, k=3,
                                s=len(t) * float(np.var(y)) * 0.5)
    except Exception:
        return None


PEAK_MIDDLE_FRAC = 0.80


def _middle_window_slice(n_dense: int, middle_frac: float = PEAK_MIDDLE_FRAC
                         ) -> tuple[int, int]:
    """Indices [start, end) that keep the middle `middle_frac` of the grid."""
    margin = int(n_dense * (1.0 - middle_frac) / 2.0)
    return margin, n_dense - margin


def _spline_peak_signed(
    t_arr: np.ndarray, y: np.ndarray, n_dense: int = 200,
    middle_frac: float = PEAK_MIDDLE_FRAC,
) -> tuple[float, float] | None:
    """Return (t_peak, y_peak_signed) picked inside the middle window of the
    spline's dense grid — avoids spline-edge artefacts where the fit is
    poorly constrained.  None if the region is flat.
    """
    if np.std(y) < 1e-9:
        return None
    spl = _spline_fit(t_arr, y)
    t_dense = np.linspace(t_arr[0], t_arr[-1], n_dense)
    y_dense = spl(t_dense) if spl is not None else np.interp(t_dense, t_arr, y)
    lo, hi = _middle_window_slice(n_dense, middle_frac)
    t_mid   = t_dense[lo:hi]
    y_mid   = y_dense[lo:hi]
    idx = int(np.argmax(np.abs(y_mid)))
    return float(t_mid[idx]), float(y_mid[idx])


def _subject_region_peaks(sub: dict) -> dict[int, tuple[float, float]]:
    """Per-region signed spline peaks across all 6 regions.

    Under the RBR-corrected twist definition (twist = rot − ⟨rot⟩) every
    region carries information, so R0 is no longer a forced-zero reference.
    """
    t_arr = sub["t_arr"]
    tw    = sub["twist_regions"]
    out: dict[int, tuple[float, float]] = {}
    for r in range(N_REGIONS):
        pk = _spline_peak_signed(t_arr, tw[:, r])
        if pk is not None:
            out[r] = pk
    return out


def _dominant_region(peaks: dict[int, tuple[float, float]]) -> int | None:
    """Apex-like dominant region — the most positive (CCW) signed peak.

    Under the RBR-corrected twist convention, the most positive region
    corresponds to the apical wringing component; in textbook normals
    this is R5.  Returns None if no region qualifies.
    """
    if not peaks:
        return None
    return max(peaks, key=lambda r: peaks[r][1])


def _minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    rng = float(x.max() - x.min())
    if rng < 1e-9:
        return np.zeros_like(x)
    return (x - x.min()) / rng


def _safe_num(s: str) -> float:
    try:
        return float(s)
    except ValueError:
        return float("nan")


# ─── Data loaders ────────────────────────────────────────────────────────────
def load_master_csv() -> list[dict]:
    rows = []
    with open(DATA_ROOT / "cohort_master_summary.csv") as f:
        for row in csv.DictReader(f):
            row["region"] = int(row["region"])
            for k in ("peak_amplitude", "t_peak", "systolic_slope",
                      "diastolic_slope", "AUC",
                      "pearson_long_raw",    "pearson_short_raw",
                      "pearson_long_smooth", "pearson_short_smooth",
                      "xcd_twist_long_s",    "xcd_twist_short_s",
                      "dtw_long",            "dtw_short"):
                row[k] = _safe_num(row[k])
            row["sign_flipped"] = int(row["sign_flipped"])
            rows.append(row)
    return rows


def load_subject(sid: str) -> dict | None:
    res = DATA_ROOT / sid / "result"
    need = ("twist_3d.npz", "strain_twist_analysis.npz",
            "xcd_analysis.npz", "dtw_analysis.npz")
    if not all((res / n).exists() for n in need):
        return None

    tw = np.load(res / "twist_3d.npz")
    an = np.load(res / "strain_twist_analysis.npz", allow_pickle=True)
    xc = np.load(res / "xcd_analysis.npz")
    dt = np.load(res / "dtw_analysis.npz")
    sign_flipped = bool(xc["sign_flipped"])

    twist_regions = an["twist_regions"].astype(np.float64).copy()
    if sign_flipped:
        twist_regions *= -1.0
    nt = twist_regions.shape[0]
    fr_hz, fr_src = subject_frame_rate(sid)
    t  = np.arange(nt) / fr_hz

    # Displacement-based per-region curves (mm).  Back-compat fallback to
    # old strain keys in case the NPZ was produced by an older pipeline run.
    if "u_long_regions" in an.files:
        u_long_raw  = an["u_long_regions"].astype(np.float64)
        u_short_raw = an["u_short_regions"].astype(np.float64)
    else:
        u_long_raw  = an["epsilon_long_regions"].astype(np.float64)
        u_short_raw = an["epsilon_short_regions"].astype(np.float64)

    window = min(5, nt // 3 * 2 - 1)
    if window >= 3:
        u_long_sm  = savgol_filter(u_long_raw,  window, 2, axis=0)
        u_short_sm = savgol_filter(u_short_raw, window, 2, axis=0)
    else:
        u_long_sm  = u_long_raw.copy()
        u_short_sm = u_short_raw.copy()

    return {
        "sid":                  sid,
        "nt":                   nt,
        "t_arr":                t,
        "frame_rate":           fr_hz,
        "frame_rate_source":    fr_src,
        "rot_regions":          tw["rot_regions"].astype(np.float64)
                                 * (-1.0 if sign_flipped else 1.0),
        "twist_regions":        twist_regions,
        "u_long_smooth":        u_long_sm,
        "u_short_smooth":       u_short_sm,
        "xcd_twist_matrix":     xc["xcd_twist_matrix"],
        "xcd_twist_long":       xc["xcd_twist_long"],
        "xcd_twist_short":      xc["xcd_twist_short"],
        "dtw_twist_long":       dt["dtw_twist_long"],
        "dtw_twist_short":      dt["dtw_twist_short"],
        "pearson_long_smooth":  dt["pearson_twist_long_smooth"],
        "pearson_short_smooth": dt["pearson_twist_short_smooth"],
        "sign_flipped":         sign_flipped,
    }


# ════════════════════════════════════════════════════════════════════════════
# Figure 1 — apex-to-base peak rotation gradient
# ════════════════════════════════════════════════════════════════════════════
def figure1_apex_base_gradient() -> tuple[Path, ...]:
    master   = load_master_csv()
    subjects = sorted({r["subject_id"] for r in master},
                      key=lambda s: ALL_SUBJECTS.index(s))
    regions  = [1, 2, 3, 4, 5]
    mat = np.full((len(subjects), len(regions)), np.nan)
    for row in master:
        if row["region"] in regions:
            mat[subjects.index(row["subject_id"]),
                regions.index(row["region"])] = row["peak_amplitude"]

    fig, ax = plt.subplots(figsize=(6.5, 4.0))

    for i, sid in enumerate(subjects):
        ax.plot(regions, mat[i], "-o",
                color=SUBJECT_COLORS[sid], lw=1.4, markersize=4.5,
                label=sid, zorder=3)
    mean = np.nanmean(mat, axis=0)
    sd   = np.nanstd(mat,  axis=0)
    ax.fill_between(regions, mean - sd, mean + sd,
                    color=SD_COLOR, alpha=0.5, zorder=1,
                    label="cohort ±SD")
    ax.plot(regions, mean, "-", color=MEAN_COLOR, lw=2.2,
            zorder=2, label="cohort mean")

    ax.set_xticks(regions)
    ax.set_xticklabels([f"R{r}" for r in regions])
    ax.set_xlabel("region (basal → apical)")
    ax.set_ylabel("peak twist (°)")
    ymax = float(np.nanmax(mat)) * 1.1
    ax.set_ylim(0, max(ymax, 1.0))
    _zero_line(ax)
    _soft_ygrid(ax)

    leg = ax.legend(loc="upper left")
    _legend_style(leg)

    return _save(fig, "Figure1_apex_base_gradient")


# ════════════════════════════════════════════════════════════════════════════
# Figure 2 — cohort twist overlay with per-region y-axes
# ════════════════════════════════════════════════════════════════════════════
def figure2_cohort_twist_overlay() -> tuple[Path, ...]:
    subs = [s for s in (load_subject(sid) for sid in ALL_SUBJECTS) if s]
    if not subs:
        raise RuntimeError("No subjects available")

    # Each subject's t_arr already uses its own DICOM-derived frame rate,
    # so take the shortest true duration as the common cutoff.
    common_T  = min(s["t_arr"][-1] for s in subs)
    max_T     = max(s["t_arr"][-1] for s in subs)
    t_common  = np.linspace(0.0, max_T, 200)
    in_common = t_common <= common_T

    curves = {}
    for sub in subs:
        sid = sub["sid"]
        curves[sid] = np.full((N_REGIONS, len(t_common)), np.nan)
        for r in range(N_REGIONS):
            if r == 0:
                curves[sid][r, :] = 0.0
                continue
            spl = _spline_fit(sub["t_arr"], sub["twist_regions"][:, r])
            if spl is None:
                curves[sid][r, :] = np.interp(
                    t_common, sub["t_arr"],
                    sub["twist_regions"][:, r],
                    left=np.nan, right=np.nan,
                )
            else:
                y = spl(np.clip(t_common, sub["t_arr"][0],
                                          sub["t_arr"][-1]))
                y[t_common > sub["t_arr"][-1]] = np.nan
                curves[sid][r, :] = y

    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.2),
                             sharex=True, sharey=False)
    panel_letters = list("abcdef")
    for r in range(N_REGIONS):
        ax = axes[r // 3, r % 3]
        row = []
        for sub in subs:
            sid = sub["sid"]
            y   = curves[sid][r]
            ax.plot(t_common[in_common], y[in_common],
                    color=SUBJECT_COLORS[sid], lw=1.2,
                    alpha=0.95, zorder=3, label=sid)
            if (~in_common).any():
                ax.plot(t_common[~in_common], y[~in_common],
                        color=SUBJECT_COLORS[sid], lw=0.9,
                        alpha=0.35, zorder=2)
            row.append(y)

        row_mat = np.array(row)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ymean = np.nanmean(row_mat[:, in_common], axis=0)
            ysd   = np.nanstd(row_mat[:, in_common],  axis=0)
        ax.fill_between(t_common[in_common], ymean - ysd, ymean + ysd,
                        color=SD_COLOR, alpha=0.6, zorder=1)
        ax.plot(t_common[in_common], ymean,
                color=MEAN_COLOR, lw=2.0, zorder=4)

        ax.axvline(common_T, color="gray", ls=":", lw=0.6, alpha=0.7)
        _zero_line(ax)

        if r == 0:
            ax.set_ylim(-0.5, 0.5)
            ax.set_title("R0 (reference)", fontsize=9, pad=4)
        else:
            # Per-panel ylim with 10% pad
            finite = row_mat[:, in_common]
            finite = finite[np.isfinite(finite)]
            if finite.size:
                lo, hi = float(finite.min()), float(finite.max())
                span = max(hi - lo, 1e-6)
                pad  = 0.1 * span
                ax.set_ylim(lo - pad, hi + pad)
            ax.set_title(f"R{r}", fontsize=9, pad=4)

        _panel_label(ax, panel_letters[r])

        if r // 3 == 1:
            ax.set_xlabel("time (s)")
        if r % 3 == 0:
            ax.set_ylabel("twist (°)")

    # Single figure-level legend below the grid
    legend_handles = [
        Line2D([0], [0], color=SUBJECT_COLORS[s["sid"]], lw=1.5,
               label=s["sid"]) for s in subs
    ] + [
        Line2D([0], [0], color=MEAN_COLOR, lw=2.0, label="cohort mean"),
        Patch(facecolor=SD_COLOR, alpha=0.6, label="±SD"),
    ]
    leg = fig.legend(
        handles=legend_handles, loc="lower center",
        bbox_to_anchor=(0.5, -0.02), ncol=len(legend_handles),
        handlelength=1.4, handletextpad=0.4,
    )
    _legend_style(leg)

    plt.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.14,
                        wspace=0.30, hspace=0.45)
    return _save(fig, "Figure2_cohort_twist_overlay")


# ════════════════════════════════════════════════════════════════════════════
# Figure 3 — three-metric coupling (central novelty figure)
# ════════════════════════════════════════════════════════════════════════════
def figure3_coupling_comparison() -> tuple[Path, ...]:
    master   = load_master_csv()
    subjects = sorted({r["subject_id"] for r in master},
                      key=lambda s: ALL_SUBJECTS.index(s))
    regions  = [1, 2, 3, 4, 5]

    def _mat(key):
        m = np.full((len(subjects), len(regions)), np.nan)
        for row in master:
            if row["region"] in regions:
                m[subjects.index(row["subject_id"]),
                  regions.index(row["region"])] = row[key]
        return m

    PR_long_raw  = _mat("pearson_long_raw")
    PR_long_sm   = _mat("pearson_long_smooth")
    DT_long      = _mat("dtw_long")
    PR_short_raw = _mat("pearson_short_raw")
    PR_short_sm  = _mat("pearson_short_smooth")
    DT_short     = _mat("dtw_short")

    dtw_vmax = float(np.nanmax(np.concatenate([DT_long.ravel(),
                                               DT_short.ravel()])))
    dtw_vmax = max(dtw_vmax, 1e-6)

    fig = plt.figure(figsize=(8.5, 5.0))
    gs  = GridSpec(
        2, 5, figure=fig,
        width_ratios=[1.0, 1.0, 0.06, 1.0, 0.06],
        hspace=0.40, wspace=0.30,
    )

    ax_pr_l = fig.add_subplot(gs[0, 0])
    ax_ps_l = fig.add_subplot(gs[0, 1])
    ax_dt_l = fig.add_subplot(gs[0, 3])
    ax_pr_s = fig.add_subplot(gs[1, 0])
    ax_ps_s = fig.add_subplot(gs[1, 1])
    ax_dt_s = fig.add_subplot(gs[1, 3])
    cax_pear = fig.add_subplot(gs[:, 2])
    cax_dtw  = fig.add_subplot(gs[:, 4])

    leftmost = {ax_pr_l, ax_pr_s}

    def _draw_heatmap(ax, data, cmap, vmin, vmax, *,
                      fmt_val, col_threshold, show_yticks):
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                if np.isnan(v):
                    txt, col = "nan", "black"
                else:
                    txt = fmt_val(v)
                    col = "white" if col_threshold(v) else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=7, color=col)
        ax.set_xticks(range(len(regions)))
        ax.set_xticklabels([f"R{r}" for r in regions])
        ax.set_yticks(range(len(subjects)))
        if show_yticks:
            ax.set_yticklabels(subjects)
        else:
            ax.set_yticklabels([""] * len(subjects))
        _enable_box_spines(ax)
        return im

    pear_im = None
    for ax in (ax_pr_l, ax_ps_l, ax_pr_s, ax_ps_s):
        data = {ax_pr_l: PR_long_raw,  ax_ps_l: PR_long_sm,
                ax_pr_s: PR_short_raw, ax_ps_s: PR_short_sm}[ax]
        pear_im = _draw_heatmap(
            ax, data, COUPLING_DIVERGING, -1, 1,
            fmt_val=lambda v: f"{v:+.2f}",
            col_threshold=lambda v: abs(v) > 0.6,
            show_yticks=(ax in leftmost),
        )

    dtw_im = None
    for ax in (ax_dt_l, ax_dt_s):
        data = DT_long if ax is ax_dt_l else DT_short
        dtw_im = _draw_heatmap(
            ax, data, COUPLING_SEQUENTIAL, 0, dtw_vmax,
            fmt_val=lambda v: f"{v:.2f}",
            col_threshold=lambda v: v > 0.55 * dtw_vmax,
            show_yticks=False,
        )

    ax_pr_l.set_title("Pearson ρ (raw)",      fontsize=9, pad=4)
    ax_ps_l.set_title("Pearson ρ (smoothed)", fontsize=9, pad=4)
    ax_dt_l.set_title("DTW distance",         fontsize=9, pad=4)

    ax_pr_l.set_ylabel("twist vs ε_long",  fontsize=8.5)
    ax_pr_s.set_ylabel("twist vs ε_short", fontsize=8.5)

    cb_p = fig.colorbar(pear_im, cax=cax_pear)
    cb_p.set_label("Pearson ρ", fontsize=8)
    cb_p.ax.tick_params(labelsize=7)
    cb_d = fig.colorbar(dtw_im, cax=cax_dtw)
    cb_d.set_label("DTW distance (lower = better)", fontsize=8)
    cb_d.ax.tick_params(labelsize=7)

    for ax, lbl in zip(
        [ax_pr_l, ax_ps_l, ax_dt_l, ax_pr_s, ax_ps_s, ax_dt_s],
        list("abcdef"),
    ):
        _panel_label(ax, lbl, dx=-0.10, dy=1.08)

    fig.text(
        0.5, -0.02,
        "DTW yields more consistent cross-subject coupling than Pearson "
        "correlation under low-frame-rate, ECG-free 3D speckle tracking "
        "(≤15 Hz). Pearson ρ fluctuates in sign and magnitude across "
        "subjects whereas DTW distance remains in a narrow range.",
        ha="center", fontsize=7.5, style="italic", color="0.3", wrap=True,
    )

    plt.subplots_adjust(left=0.08, right=0.94, top=0.92, bottom=0.10,
                        wspace=0.30, hspace=0.40)
    return _save(fig, "Figure3_coupling_comparison")


# ════════════════════════════════════════════════════════════════════════════
# Figure 4 — XCD map (subject 03)
# ════════════════════════════════════════════════════════════════════════════
def figure4_xcd_subject03() -> tuple[Path, ...]:
    sub = load_subject("03")
    if sub is None:
        raise FileNotFoundError("Subject 03 result files missing")

    mat_ms = sub["xcd_twist_matrix"] * 1000.0
    mat_ms = mat_ms.copy()
    np.fill_diagonal(mat_ms, 0.0)
    vmax = max(float(np.nanmax(np.abs(mat_ms))), 1.0)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(mat_ms, cmap=COUPLING_DIVERGING,
                   vmin=-vmax, vmax=vmax, aspect="equal")
    for i in range(N_REGIONS):
        for j in range(N_REGIONS):
            v = mat_ms[i, j]
            if np.isnan(v):
                txt, col = "nan", "black"
            else:
                txt = f"{int(round(v))}"
                col = "white" if abs(v) > 0.6 * vmax else "black"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=8, color=col)
    ax.set_xticks(range(N_REGIONS))
    ax.set_yticks(range(N_REGIONS))
    ax.set_xticklabels([f"R{r}" for r in range(N_REGIONS)])
    ax.set_yticklabels([f"R{r}" for r in range(N_REGIONS)])
    ax.set_xlabel("column region")
    ax.set_ylabel("row region")
    _enable_box_spines(ax)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("delay (ms)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    ax.text(0.5, -0.15, "positive = row leads column",
            transform=ax.transAxes, ha="center", va="top",
            style="italic", fontsize=8)
    ax.text(0.5, -0.24,
            "Subject 03: apex regions (R4–R5) lead basal regions (R1–R2) "
            "by ~300–500 ms,\nconsistent with apex-first rotation onset.",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.5)

    plt.subplots_adjust(left=0.12, right=0.94, top=0.95, bottom=0.18)
    return _save(fig, "Figure4_xcd_subject03")


# ════════════════════════════════════════════════════════════════════════════
# Figure 5 — per-subject twist curves with features
# ════════════════════════════════════════════════════════════════════════════
def figure5_twist_features(subject_id: str,
                           master_rows: list[dict]) -> tuple[Path, ...] | None:
    sub = load_subject(subject_id)
    if sub is None:
        return None

    t_arr = sub["t_arr"]
    tw    = sub["twist_regions"]
    region_cmap   = plt.get_cmap(REGION_CMAP)
    region_colors = [region_cmap(i / max(N_REGIONS - 1, 1))
                     for i in range(N_REGIONS)]

    spline_peaks = _subject_region_peaks(sub)

    # AUC still comes from the CSV (pipeline feature); peak / t_peak are
    # recomputed from the spline so the star lies ON the curve.
    feats_by_region = {
        int(r["region"]): r
        for r in master_rows if r["subject_id"] == subject_id
    }

    fig, axes = plt.subplots(2, 3, figsize=(7.5, 5.0), sharex=True)
    panel_letters = list("abcdef")

    assertion_records: list[tuple[int, float, float, float]] = []

    for r in range(N_REGIONS):
        ax = axes[r // 3, r % 3]
        y  = tw[:, r]

        ax.scatter(t_arr, y, s=12, color=region_colors[r], alpha=0.55,
                   edgecolors="none", zorder=3)

        if np.std(y) < 1e-9:
            ax.plot(t_arr, np.zeros_like(t_arr),
                    color=region_colors[r], lw=1.5, zorder=4)
            ax.set_title(f"R{r} (flat)", fontsize=9, pad=4)
            ax.set_ylim(-0.5, 0.5)
        else:
            spl = _spline_fit(t_arr, y)
            t_dense = np.linspace(t_arr[0], t_arr[-1], 200)
            y_dense = spl(t_dense) if spl is not None else np.interp(
                t_dense, t_arr, y)

            lo, hi = _middle_window_slice(len(t_dense))
            tpk, peak = spline_peaks.get(r, (
                float(t_dense[lo + int(np.argmax(np.abs(y_dense[lo:hi])))]),
                float(y_dense[lo + int(np.argmax(np.abs(y_dense[lo:hi])))]),
            ))
            feat = feats_by_region.get(r)
            auc  = (feat["AUC"] if feat
                    else float(np.trapezoid(y_dense, t_dense)))

            ax.axvspan(t_dense[0], tpk, color="#d62728",
                       alpha=0.08, zorder=1)
            ax.axvspan(tpk, t_dense[-1], color="#1f77b4",
                       alpha=0.08, zorder=1)

            ax.plot(t_dense, y_dense, color=region_colors[r],
                    lw=1.6, zorder=4)
            ax.scatter([tpk], [peak], marker="*", s=80,
                       color="#d62728", edgecolors="black",
                       linewidths=0.5, zorder=6)

            # Record for end-of-function assertion: star's y must equal the
            # drawn curve's y at the star's x.
            y_curve_at_tpk = (float(spl(tpk)) if spl is not None
                              else float(np.interp(tpk, t_arr, y)))
            assertion_records.append((r, tpk, peak, y_curve_at_tpk))

            ann = (f"peak = {peak:+.2f}°\n"
                   f"t_pk = {tpk:.2f} s\n"
                   f"AUC = {auc:+.2f}")
            ax.text(0.98, 0.98, ann,
                    transform=ax.transAxes,
                    fontsize=7, va="top", ha="right", family="monospace",
                    bbox=_annotation_bbox())

            ax.set_title(f"R{r}", fontsize=9, pad=4)

        _zero_line(ax)
        _panel_label(ax, panel_letters[r])
        if r % 3 == 0:
            ax.set_ylabel("twist (°)")
        if r // 3 == 1:
            ax.set_xlabel("time (s)")

    flip_tag = "  [rotation sign flipped]" if sub["sign_flipped"] else ""
    fig.text(0.99, 0.01, f"Subject {subject_id}{flip_tag}",
             ha="right", va="bottom", fontsize=7.5, color="#555",
             style="italic")

    plt.subplots_adjust(left=0.08, right=0.96, top=0.93, bottom=0.10,
                        wspace=0.28, hspace=0.40)

    # ── Sanity checks ─────────────────────────────────────────────────────
    # (1) star y-value must equal drawn curve y at star x.
    # (2) star t must lie inside the middle-window used for peak detection.
    t_lo = t_arr[0]
    t_hi = t_arr[-1]
    margin = (t_hi - t_lo) * (1.0 - PEAK_MIDDLE_FRAC) / 2.0
    win_lo = t_lo + margin
    win_hi = t_hi - margin
    for r, tpk, peak, y_curve in assertion_records:
        if abs(peak - y_curve) >= 0.01:
            raise AssertionError(
                f"[Figure 5 / {subject_id}] R{r}: red star y={peak:.4f}°"
                f" but spline y({tpk:.4f}s)={y_curve:.4f}° — star is off "
                "the curve."
            )
        if not (win_lo - 1e-6 <= tpk <= win_hi + 1e-6):
            raise AssertionError(
                f"[Figure 5 / {subject_id}] R{r}: peak time {tpk:.3f}s is "
                f"outside the middle-{int(PEAK_MIDDLE_FRAC*100)}% window "
                f"[{win_lo:.3f}, {win_hi:.3f}]s — spline-edge artefact."
            )

    return _save(fig, f"Figure5_twist_features_{subject_id}")


# ════════════════════════════════════════════════════════════════════════════
# Figure 6 — feature distributions (1×3)
# ════════════════════════════════════════════════════════════════════════════
def figure6_feature_distributions() -> tuple[Path, ...]:
    master   = load_master_csv()
    subjects = sorted({r["subject_id"] for r in master},
                      key=lambda s: ALL_SUBJECTS.index(s))
    regions  = [1, 2, 3, 4, 5]
    keys     = ("peak_amplitude", "t_peak", "AUC")
    titles   = {"peak_amplitude": "peak twist (°)",
                "t_peak":         "time to peak (s)",
                "AUC":            "AUC (°·s)"}

    per_feat: dict[str, list[list[float]]] = {
        k: [[] for _ in regions] for k in keys
    }
    subj_rows: dict[str, dict[int, dict]] = {s: {} for s in subjects}
    for row in master:
        if row["region"] in regions:
            subj_rows[row["subject_id"]][row["region"]] = row
            for k in keys:
                if not np.isnan(row[k]):
                    per_feat[k][regions.index(row["region"])].append(row[k])

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.5))
    rng = np.random.default_rng(42)
    for ax, key in zip(axes, keys):
        data = per_feat[key]
        bp = ax.boxplot(
            data, positions=regions, widths=0.55,
            patch_artist=True, showfliers=False,
            medianprops=dict(color="black", linewidth=1.4),
            whiskerprops=dict(color="#555", linewidth=0.8),
            capprops=dict(color="#555", linewidth=0.8),
            boxprops=dict(edgecolor="#555", linewidth=0.8),
        )
        for box in bp["boxes"]:
            box.set(facecolor="#e0e0e0")

        for s in subjects:
            xs, ys = [], []
            for r in regions:
                if r in subj_rows[s]:
                    xs.append(r + rng.uniform(-0.12, 0.12))
                    ys.append(subj_rows[s][r][key])
            ax.scatter(xs, ys, s=25, color=SUBJECT_COLORS[s],
                       edgecolors="black", linewidths=0.4, zorder=5,
                       alpha=0.8)

        ax.set_xticks(regions)
        ax.set_xticklabels([f"R{r}" for r in regions])
        ax.set_xlabel("region")
        ax.set_ylabel(titles[key])
        _soft_ygrid(ax)
        if key in ("peak_amplitude", "AUC"):
            _zero_line(ax)

    for ax, lbl in zip(axes, "abc"):
        _panel_label(ax, lbl, dx=-0.18, dy=1.06)

    handles = [
        Line2D([0], [0], marker="o", linestyle="",
               color=SUBJECT_COLORS[s], markersize=5,
               markeredgecolor="black", markeredgewidth=0.4, label=s)
        for s in subjects
    ]
    leg = fig.legend(handles=handles, loc="upper center",
                     bbox_to_anchor=(0.5, 1.02),
                     ncol=len(subjects))
    _legend_style(leg)

    plt.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.14,
                        wspace=0.35)
    return _save(fig, "Figure6_feature_distributions")


# ════════════════════════════════════════════════════════════════════════════
# Figure 7 — twist–strain phase view (subject 03)
# ════════════════════════════════════════════════════════════════════════════
def figure7_phase_view_subject03() -> tuple[Path, ...]:
    sub = load_subject("03")
    if sub is None:
        raise FileNotFoundError("Subject 03 result files missing")

    t_arr = sub["t_arr"]
    tw    = sub["twist_regions"]
    uLS   = sub["u_long_smooth"]    # displacement-long, mm
    uSS   = sub["u_short_smooth"]   # displacement-short, mm
    p_l   = sub["pearson_long_smooth"]
    p_s   = sub["pearson_short_smooth"]
    d_l   = sub["dtw_twist_long"]
    d_s   = sub["dtw_twist_short"]

    color_t = "steelblue"
    color_l = "firebrick"
    color_s = "darkseagreen"

    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.2), sharex=True)
    panel_letters = list("abcdef")
    for r in range(N_REGIONS):
        ax = axes[r // 3, r % 3]

        y_t = _minmax(tw[:, r]) if np.std(tw[:, r]) > 1e-9 \
              else np.zeros_like(tw[:, r])
        y_l = _minmax(uLS[:, r])
        y_s = _minmax(uSS[:, r])

        ax.plot(t_arr, y_t, "-",  color=color_t, lw=1.4, zorder=4)
        ax.plot(t_arr, y_l, "-",  color=color_l, lw=1.2, zorder=3)
        ax.plot(t_arr, y_s, "--", color=color_s, lw=1.2, zorder=3)

        ax.set_ylim(-0.08, 1.08)
        ax.set_title(f"R{r}" + ("  (reference)" if r == 0 else ""),
                     fontsize=9, pad=4)

        if r == 0:
            ann = "reference\ntwist ≡ 0"
        else:
            i = r - 1
            ann = (f"ρ={p_l[i]:+.2f}  DTW={d_l[i]:.2f}  [long]\n"
                   f"ρ={p_s[i]:+.2f}  DTW={d_s[i]:.2f}  [short]")
        ax.text(0.98, 0.98, ann, transform=ax.transAxes,
                fontsize=7, va="top", ha="right", family="monospace",
                bbox=_annotation_bbox())

        _panel_label(ax, panel_letters[r])
        if r // 3 == 1:
            ax.set_xlabel("time (s)")
        if r % 3 == 0:
            ax.set_ylabel("normalized amplitude [0,1]")

    legend_handles = [
        Line2D([0], [0], color=color_t, lw=1.4, label="twist"),
        Line2D([0], [0], color=color_l, lw=1.2, label="u_long (mm)"),
        Line2D([0], [0], color=color_s, lw=1.2, ls="--",
               label="u_short (mm)"),
    ]
    leg = fig.legend(handles=legend_handles, loc="lower center",
                     bbox_to_anchor=(0.5, -0.03), ncol=3)
    _legend_style(leg)

    plt.subplots_adjust(left=0.08, right=0.97, top=0.92, bottom=0.14,
                        wspace=0.28, hspace=0.45)
    return _save(fig, "Figure7_phase_view_subject03")


# ════════════════════════════════════════════════════════════════════════════
# Figure 9 — cohort angle-change dominance summary
# ════════════════════════════════════════════════════════════════════════════
def _collect_cohort_spline_peaks(
) -> tuple[list[str], list[int], np.ndarray, dict[str, dict[int, tuple[float, float]]]]:
    """Matrix of signed spline peaks per subject × region (R1..R5).

    Returns (subjects, regions, peak_mat_signed, per_subject_peaks).
    Subjects missing NPZ files are dropped.
    """
    regions = [0, 1, 2, 3, 4, 5]
    subjects: list[str] = []
    per_subject: dict[str, dict[int, tuple[float, float]]] = {}
    rows: list[np.ndarray] = []
    for sid in ALL_SUBJECTS:
        sub = load_subject(sid)
        if sub is None:
            continue
        peaks = _subject_region_peaks(sub)
        per_subject[sid] = peaks
        subjects.append(sid)
        rows.append(np.array([
            peaks[r][1] if r in peaks else np.nan for r in regions
        ], dtype=float))
    peak_mat = np.vstack(rows) if rows else np.zeros((0, len(regions)))
    return subjects, regions, peak_mat, per_subject


def _collect_cohort_spline_peaks_with_times(
) -> tuple[list[str], list[int], np.ndarray, np.ndarray,
           dict[str, dict[int, tuple[float, float]]]]:
    """Signed-peak matrix *and* peak-time matrix (per subject × region).

    Includes R0 since the RBR-corrected twist makes R0 a meaningful
    quantity (the most-basal region's deviation from the LV mean rotation).
    """
    regions = [0, 1, 2, 3, 4, 5]
    subjects: list[str] = []
    per_subject: dict[str, dict[int, tuple[float, float]]] = {}
    peak_rows: list[np.ndarray] = []
    time_rows: list[np.ndarray] = []
    for sid in ALL_SUBJECTS:
        sub = load_subject(sid)
        if sub is None:
            continue
        peaks = _subject_region_peaks(sub)
        per_subject[sid] = peaks
        subjects.append(sid)
        peak_rows.append(np.array([
            peaks[r][1] if r in peaks else np.nan for r in regions
        ], dtype=float))
        time_rows.append(np.array([
            peaks[r][0] if r in peaks else np.nan for r in regions
        ], dtype=float))
    peak_mat = np.vstack(peak_rows) if peak_rows else np.zeros((0, 5))
    time_mat = np.vstack(time_rows) if time_rows else np.zeros((0, 5))
    return subjects, regions, peak_mat, time_mat, per_subject


def figure9_angle_dominance_summary() -> tuple[Path, ...]:
    """Two-panel summary of the peak-only view of regional twist.

    (a) Signed RBR-corrected peak per region, one line per subject, color-
        coded by clinical class.  Shows that normal subjects do not share
        a textbook profile under peak-only analysis and overlap with the
        abnormal cohort.
    (b) Apical-dominant region tally (region with the most positive signed
        peak per subject), stacked by class.  Shows that R5 is not the
        modal apical-dominant region across the cohort.
    """
    subjects, regions, peak_mat, _time_mat, _ = \
        _collect_cohort_spline_peaks_with_times()
    cls = np.array([smeta.diagnosis_class(s) for s in subjects])
    is_norm = (cls == "normal")
    is_abn  = (cls == "abnormal")

    dom_col = np.full(len(subjects), -1, dtype=int)
    for i in range(len(subjects)):
        if np.any(np.isfinite(peak_mat[i])):
            dom_col[i] = int(np.nanargmax(peak_mat[i]))

    NORM_COLOR = "#1b7837"
    ABN_COLOR  = "#b2182b"

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))

    # ── Panel (a): per-subject signed-peak profile, colored by class ─────
    axA = axes[0]
    for i, sid in enumerate(subjects):
        color = NORM_COLOR if is_norm[i] else ABN_COLOR
        axA.plot(regions, peak_mat[i], "-o",
                 color=color, lw=1.4, markersize=4.5,
                 alpha=0.85, zorder=3)
        # subject-id label at apex end (right side) to avoid a legend
        x_end = regions[-1]
        y_end = peak_mat[i, -1]
        if np.isfinite(y_end):
            axA.annotate(sid, xy=(x_end, y_end),
                         xytext=(4, 0), textcoords="offset points",
                         fontsize=6.5, color=color, va="center",
                         alpha=0.9)
    axA.axhline(0, color="0.5", ls="--", lw=0.5, alpha=0.7, zorder=1)
    axA.set_xticks(regions)
    axA.set_xticklabels([f"R{r}" for r in regions])
    axA.set_xlabel("region  (basal → apical)")
    axA.set_ylabel("peak twist  (°, signed)")
    axA.set_title("(a)  Per-region signed peak", fontsize=10, pad=4)
    # class-key legend (just two entries — no per-subject clutter)
    handles = [plt.Line2D([0], [0], color=NORM_COLOR, lw=1.6, marker="o",
                          markersize=4.5,
                          label=f"normal (n={int(is_norm.sum())})"),
               plt.Line2D([0], [0], color=ABN_COLOR,  lw=1.6, marker="o",
                          markersize=4.5,
                          label=f"abnormal (n={int(is_abn.sum())})")]
    leg = axA.legend(handles=handles, loc="upper left", fontsize=7.5,
                     frameon=True)
    _legend_style(leg)
    _panel_label(axA, "a")
    _soft_ygrid(axA)

    # ── Panel (b): apical-dominant region tally, stacked by class ────────
    axB = axes[1]
    tally_norm = np.zeros(len(regions), dtype=int)
    tally_abn  = np.zeros(len(regions), dtype=int)
    for i in range(len(subjects)):
        c = dom_col[i]
        if c < 0:
            continue
        if is_norm[i]:
            tally_norm[c] += 1
        elif is_abn[i]:
            tally_abn[c]  += 1
    axB.bar(regions, tally_norm, color=NORM_COLOR,
            edgecolor="black", linewidth=0.6,
            label=f"normal (n={int(is_norm.sum())})")
    axB.bar(regions, tally_abn, bottom=tally_norm,
            color=ABN_COLOR, edgecolor="black", linewidth=0.6,
            label=f"abnormal (n={int(is_abn.sum())})")
    total = tally_norm + tally_abn
    for r, h in zip(regions, total):
        if h > 0:
            axB.text(r, h + 0.08, str(int(h)),
                     ha="center", va="bottom", fontsize=9, fontweight="bold")
    axB.set_xticks(regions)
    axB.set_xticklabels([f"R{r}" for r in regions])
    axB.set_xlabel("region with largest signed peak")
    axB.set_ylabel("subjects")
    axB.set_title("(b)  Apical-dominant region tally", fontsize=10, pad=4)
    axB.set_ylim(0, max(int(total.max()) + 1, 2))
    leg = axB.legend(loc="upper right", fontsize=7.5)
    _legend_style(leg)
    _panel_label(axB, "b")

    plt.subplots_adjust(left=0.08, right=0.97, top=0.90,
                        bottom=0.16, wspace=0.32)
    return _save(fig, "Figure9_angle_dominance")


def write_cohort_angle_dominance_csv() -> Path:
    """Compact paper-style table — one row per subject.

    Columns: subject_id, diagnosis, R0..R5 (signed peak in degrees),
    apical_dom (region label of argmax signed peak),
    basal_dom  (region label of argmin signed peak).
    """
    subjects, regions, peak_mat, _time_mat, _ = \
        _collect_cohort_spline_peaks_with_times()
    out = DATA_ROOT / "cohort_angle_dominance.csv"
    header = (["subject_id", "diagnosis"]
              + [f"R{r}" for r in regions]
              + ["apical_dom", "basal_dom"])
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i, sid in enumerate(subjects):
            row_pk = peak_mat[i]
            if np.any(np.isfinite(row_pk)):
                ap_r = f"R{regions[int(np.nanargmax(row_pk))]}"
                bs_r = f"R{regions[int(np.nanargmin(row_pk))]}"
            else:
                ap_r = bs_r = "—"
            diag = smeta.diagnosis_class(sid)
            w.writerow(
                [sid, diag]
                + [f"{v:+.2f}" if np.isfinite(v) else "nan" for v in row_pk]
                + [ap_r, bs_r]
            )
    return out


# ════════════════════════════════════════════════════════════════════════════
# Figure 10 — twist lifecycle (apical / basal mean + net twist strip)
# ════════════════════════════════════════════════════════════════════════════
F10_LAYOUT = [
    ["002A",   "03",       "p009npa",   "p009pa"],
    ["p066a",  "p020_1a",  "p030_lp2a", None],   # None = legend slot
]
F10_APICAL_COLOR = "#d35400"   # warm orange
F10_BASAL_COLOR  = "#2c7fb8"   # cool blue
F10_NET_COLOR    = "#000000"   # black


def _f10_draw_legend_panel(ax, region_colors) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.05, 0.96, "Reading guide",
            fontsize=9.5, fontweight="bold", color="0.15",
            transform=ax.transAxes, va="top", ha="left")

    # Six-region backdrop key
    y0 = 0.84
    ax.text(0.05, y0 + 0.03, "Backdrop — six region curves (faded)",
            fontsize=7.5, color="0.25",
            transform=ax.transAxes, va="bottom", ha="left")
    for r, c in enumerate(region_colors):
        ax.plot([0.07, 0.22], [y0 - 0.035 * r, y0 - 0.035 * r],
                color=c, lw=1.0, alpha=0.55,
                transform=ax.transAxes, clip_on=False)
        ax.text(0.24, y0 - 0.035 * r, f"R{r}",
                fontsize=7, color="0.30",
                transform=ax.transAxes, va="center", ha="left")
    ax.text(0.42, y0 - 0.09,
            "viridis colormap:\nR0 = base → R5 = apex",
            fontsize=6.8, color="0.35", style="italic",
            transform=ax.transAxes, va="center", ha="left")

    # Foreground curves (upper subplot)
    y1 = 0.44
    ax.text(0.05, y1 + 0.05, "Upper subplot — derived means",
            fontsize=7.5, color="0.25",
            transform=ax.transAxes, va="bottom", ha="left")
    upper = [
        (F10_BASAL_COLOR,  1.9, "basal mean   ⟨R0:R2⟩"),
        (F10_APICAL_COLOR, 1.9, "apical mean  ⟨R3:R5⟩"),
    ]
    for i, (c, lw, txt) in enumerate(upper):
        y = y1 - 0.07 * i
        ax.plot([0.07, 0.22], [y, y], color=c, lw=lw,
                transform=ax.transAxes, clip_on=False)
        ax.text(0.24, y, txt, fontsize=7.2, color="0.20",
                transform=ax.transAxes, va="center", ha="left")

    # Lower subplot — net twist (difference)
    y2 = 0.22
    ax.text(0.05, y2 + 0.05,
            "Lower subplot — net twist  (apical − basal)",
            fontsize=7.5, color="0.25",
            transform=ax.transAxes, va="bottom", ha="left")
    ax.plot([0.07, 0.22], [y2, y2],
            color=F10_NET_COLOR, lw=2.4,
            transform=ax.transAxes, clip_on=False)
    ax.text(0.24, y2, "net twist", fontsize=7.2, color="0.20",
            transform=ax.transAxes, va="center", ha="left")

    # Peak-twist (≈ AVC) marker
    y3 = 0.08
    ax.plot([0.13, 0.13], [y3 - 0.035, y3 + 0.035],
            color="0.4", ls="--", lw=1.0,
            transform=ax.transAxes, clip_on=False)
    ax.text(0.18, y3,
            "peak-twist time  (≈ AVC, surrogate)",
            fontsize=7.0, color="0.25",
            transform=ax.transAxes, va="center", ha="left")


def figure10_twist_lifecycle() -> tuple[Path, ...]:
    """Per-subject lifecycle of LV rotation.

    Each subject occupies a stacked pair of axes:
      • upper : six region curves (faded viridis backdrop) plus the
                basal-third mean ⟨R0:R2⟩ (blue) and apical-third mean
                ⟨R3:R5⟩ (orange).
      • lower : the net twist (apical − basal) drawn on its own y-axis
                so the difference is easy to read off without overlap
                with the apical / basal traces.
    A vertical dashed line marks the peak-twist time, used here as a
    surrogate for AVC because clinical valve-event timing is not
    available for this cohort.  Each panel uses its own subject's
    cycle length on the x-axis.
    """
    region_cmap   = plt.get_cmap(REGION_CMAP)
    region_colors = [region_cmap(i / max(N_REGIONS - 1, 1))
                     for i in range(N_REGIONS)]

    n_rows = len(F10_LAYOUT)
    n_cols = max(len(r) for r in F10_LAYOUT)

    subs: dict[str, dict] = {}
    for row in F10_LAYOUT:
        for sid in row:
            if sid is not None and sid not in subs:
                s = load_subject(sid)
                if s is not None:
                    subs[sid] = s

    # Shared rotation y-range (upper subplots) and net y-range (lower).
    all_rot = np.concatenate([s["rot_regions"].ravel() for s in subs.values()])
    rot_min, rot_max = float(np.nanmin(all_rot)), float(np.nanmax(all_rot))
    pad = (rot_max - rot_min) * 0.06
    rot_lim = (rot_min - pad, rot_max + pad)

    net_max_abs = 0.0
    for s in subs.values():
        b = s["rot_regions"][:, 0:3].mean(axis=1)
        a = s["rot_regions"][:, 3:6].mean(axis=1)
        net_max_abs = max(net_max_abs, float(np.nanmax(np.abs(a - b))))
    net_lim = (-net_max_abs * 1.10, net_max_abs * 1.10)

    fig = plt.figure(figsize=(14.0, 7.2))
    outer_gs = fig.add_gridspec(n_rows, n_cols,
                                hspace=0.42, wspace=0.30,
                                left=0.05, right=0.99,
                                top=0.96, bottom=0.07)

    for row_i, row_subs in enumerate(F10_LAYOUT):
        for col_i in range(n_cols):
            sid = row_subs[col_i] if col_i < len(row_subs) else None

            if sid is None:
                ax_leg = fig.add_subplot(outer_gs[row_i, col_i])
                _f10_draw_legend_panel(ax_leg, region_colors)
                continue

            sub = subs.get(sid)
            if sub is None:
                ax = fig.add_subplot(outer_gs[row_i, col_i])
                ax.axis("off")
                ax.text(0.5, 0.5, f"{sid}: missing",
                        ha="center", va="center",
                        fontsize=8, color="0.5",
                        transform=ax.transAxes)
                continue

            t = sub["t_arr"]
            R = sub["rot_regions"]
            basal  = R[:, 0:3].mean(axis=1)
            apical = R[:, 3:6].mean(axis=1)
            net    = apical - basal

            inner = outer_gs[row_i, col_i].subgridspec(
                2, 1, height_ratios=[3.2, 1.2], hspace=0.08)
            ax_up = fig.add_subplot(inner[0])
            ax_dn = fig.add_subplot(inner[1], sharex=ax_up)

            # Upper — 6 faded region curves + basal/apical means
            for r in range(N_REGIONS):
                ax_up.plot(t, R[:, r], "-", color=region_colors[r],
                           lw=0.6, alpha=0.28, zorder=2)
            ax_up.plot(t, basal,  "-", color=F10_BASAL_COLOR,
                       lw=1.9, zorder=4)
            ax_up.plot(t, apical, "-", color=F10_APICAL_COLOR,
                       lw=1.9, zorder=4)

            # Lower — net twist
            ax_dn.plot(t, net, "-", color=F10_NET_COLOR,
                       lw=1.9, zorder=4)
            ax_dn.fill_between(t, 0, net, color=F10_NET_COLOR,
                               alpha=0.10, zorder=2)

            # Peak-twist (≈ AVC) — both subplots
            spl_peak = _spline_peak_signed(t, net)
            if spl_peak is not None:
                t_pk, net_pk = spl_peak
                for ax in (ax_up, ax_dn):
                    ax.axvline(t_pk, color="0.4", ls="--",
                               lw=0.9, alpha=0.85, zorder=3)
                ax_up.text(0.97, 0.96,
                           f"net peak  {net_pk:+.1f}°\n"
                           f"t ≈ AVC*  {t_pk:.2f} s",
                           transform=ax_up.transAxes,
                           fontsize=6.4, color="0.20",
                           va="top", ha="right", family="monospace",
                           bbox=_annotation_bbox())

            for ax in (ax_up, ax_dn):
                ax.axhline(0, color="0.55", ls=":", lw=0.5,
                           alpha=0.7, zorder=1)
                ax.set_xlim(t[0], t[-1])
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
                ax.tick_params(labelsize=7.2)

            ax_up.set_ylim(rot_lim)
            ax_dn.set_ylim(net_lim)

            ax_up.set_title(sid, fontsize=9.0, pad=3, fontweight="bold")
            plt.setp(ax_up.get_xticklabels(), visible=False)
            ax_dn.set_xlabel("time (s)", fontsize=7.6)
            if col_i == 0:
                ax_up.set_ylabel("rotation (°)", fontsize=7.6)
                ax_dn.set_ylabel("net (°)",      fontsize=7.2)

    return _save(fig, "Figure10_twist_lifecycle")


# ════════════════════════════════════════════════════════════════════════════
# Figure 11 — region-wise case-study analysis (single subject)
# ════════════════════════════════════════════════════════════════════════════
#
# Methodological context
# ----------------------
# Most clinical LV-twist studies summarise the cycle with two endpoint
# rotations (apex / base) plus their difference (net twist), e.g.:
#   • Sengupta et al., JACC Cardiovasc Imaging 2008 / 2014 — wringing model
#   • Notomi et al.,   Circulation / JACC 2005      — STE baseline
#   • Helle-Valle et al., Circulation 2005          — 3D STE foundational
# A smaller body of work argues that rotation behaves as a wave that
# propagates along the helical band of the LV, so mid-LV regions carry
# information beyond the two endpoints (Buckberg, Circulation 2005;
# Russel, EHJ 2009; Stöhr, AJP-HCP 2015).  Our 6-region pipeline retains
# the four mid-LV regions (R1..R4) that 2-region summaries discard.
#
# Layout
# ------
#   (a) — six region twist curves (RBR-corrected), with peaks marked
#         and three scalar summaries displayed:
#             σ_tpeak       — peak-time spread across regions
#             ρ_sys(R0,R5)  — apex-base correlation in systole
#             W_R0R5        — apex-base work covariance (raw rotation)
#   (b) — fitted single twist curve obtained by per-frame linear
#         regression  twist(z, t) ≈ α(t)·z + β(t)  through the six
#         (z_r, twist_r) samples; we plot α(t)·L  where  L = z_R5 −
#         z_R0.  Using all six regions makes this fitted curve more
#         robust than a simple R5−R0 difference: any one noisy region
#         has limited influence on the slope.
# ════════════════════════════════════════════════════════════════════════════
def _f11_compute_z_mm(sid: str) -> np.ndarray:
    """Per-region mean long-axis position in mm (used for the linear
    twist gradient fit in panel (b))."""
    res = DATA_ROOT / sid / "result"
    tw  = np.load(res / "twist_3d.npz")
    an  = np.load(res / "strain_twist_analysis.npz", allow_pickle=True)
    us  = np.load(DATA_ROOT / sid / "us4d.npz")

    seeds      = tw["seeds"].astype(float)
    region_ids = tw["region_ids"].astype(int)
    d_long     = an["d_long"].astype(float)
    scale      = np.asarray(us["scale"]).astype(float)

    seeds_mm = seeds * scale[None, :]
    z_long   = seeds_mm @ d_long
    z_per_r  = np.full(N_REGIONS, np.nan)
    for r in range(N_REGIONS):
        mask = (region_ids == r)
        if mask.any():
            z_per_r[r] = float(z_long[mask].mean())
    return z_per_r


def figure11_region_analysis(subject_id: str = "002A"
                             ) -> tuple[Path, ...] | None:
    """Region-wise case-study figure for one subject."""
    sub = load_subject(subject_id)
    if sub is None:
        return None

    t      = sub["t_arr"]
    R      = sub["twist_regions"]                  # RBR-corrected, (nt, 6)
    R_raw  = sub["rot_regions"]                    # RAW rotation, (nt, 6)

    region_cmap   = plt.get_cmap(REGION_CMAP)
    region_colors = [region_cmap(i / max(N_REGIONS - 1, 1))
                     for i in range(N_REGIONS)]

    # Spline-smoothed dense traces for plotting + linear-gradient fit
    t_dense = np.linspace(t[0], t[-1], 240)
    R_dense = np.empty((N_REGIONS, t_dense.size))
    for r in range(N_REGIONS):
        spl        = _spline_fit(t, R[:, r])
        R_dense[r] = (spl(t_dense) if spl is not None
                      else np.interp(t_dense, t, R[:, r]))

    # Per-region peak (signed) via the middle-window spline rule
    peak_amps  = np.full(N_REGIONS, np.nan)
    peak_times = np.full(N_REGIONS, np.nan)
    for r in range(N_REGIONS):
        p = _spline_peak_signed(t, R[:, r])
        if p is not None:
            peak_times[r], peak_amps[r] = p

    # AVC* surrogate from apical/basal-third net twist
    basal_t  = R[:, 0:3].mean(axis=1)
    apical_t = R[:, 3:6].mean(axis=1)
    spl_net  = _spline_peak_signed(t, apical_t - basal_t)
    t_avc    = spl_net[0] if spl_net is not None else None

    # Three scalar metrics (matching the cohort table)
    finite_pt   = np.isfinite(peak_times)
    sigma_tpeak = (float(np.std(peak_times[finite_pt]) * 1000.0)
                   if int(finite_pt.sum()) >= 2 else float("nan"))
    if t_avc is not None:
        sys_mask = (t <= t_avc)
        if int(sys_mask.sum()) >= 3:
            a_sys, b_sys = R[sys_mask, 0], R[sys_mask, 5]
            rho_sys = (float(np.corrcoef(a_sys, b_sys)[0, 1])
                       if a_sys.std() > 1e-9 and b_sys.std() > 1e-9
                       else float("nan"))
        else:
            rho_sys = float("nan")
    else:
        rho_sys = float("nan")
    a_raw  = R_raw[:, 0] - R_raw[:, 0].mean()
    b_raw  = R_raw[:, 5] - R_raw[:, 5].mean()
    W_R0R5 = float(np.mean(a_raw * b_raw))

    # ── Linear-gradient fitted twist  α(t)·L  ────────────────────────────
    z_mm     = _f11_compute_z_mm(subject_id)
    valid_z  = ~np.isnan(z_mm)
    if int(valid_z.sum()) >= 3 and z_mm[valid_z].std() > 1e-9:
        z_v          = z_mm[valid_z]
        L_axis       = float(z_v.max() - z_v.min())
        alpha_dense  = np.empty(t_dense.size)
        for i in range(t_dense.size):
            y           = R_dense[valid_z, i]
            coefs       = np.polyfit(z_v, y, 1)
            alpha_dense[i] = float(coefs[0])
        fitted_twist = alpha_dense * L_axis
    else:
        fitted_twist = np.full(t_dense.size, np.nan)
        L_axis       = float("nan")

    fig = plt.figure(figsize=(8.0, 6.4))
    gs  = fig.add_gridspec(2, 1, height_ratios=[2.4, 1.2], hspace=0.16,
                           left=0.10, right=0.96,
                           top=0.92, bottom=0.10)

    # ── (a) Six region twist curves ──────────────────────────────────────
    axA = fig.add_subplot(gs[0])
    for r in range(N_REGIONS):
        axA.scatter(t, R[:, r], s=10, color=region_colors[r],
                    alpha=0.30, edgecolors="none", zorder=2)
        axA.plot(t_dense, R_dense[r], "-", color=region_colors[r],
                 lw=1.7, alpha=0.95, zorder=4, label=f"R{r}")
        if np.isfinite(peak_times[r]):
            axA.scatter([peak_times[r]], [peak_amps[r]],
                        marker="*", s=80,
                        color=region_colors[r],
                        edgecolors="black", linewidths=0.5, zorder=6)
    if t_avc is not None:
        axA.axvline(t_avc, color="0.4", ls="--", lw=0.9,
                    alpha=0.8, zorder=3)

    annot_a = (f"σ_tpeak       = {sigma_tpeak:.0f} ms\n"
               f"ρ_sys(R0,R5)  = {rho_sys:+.2f}\n"
               f"W_R0R5        = {W_R0R5:+.2f} °²")
    axA.text(0.02, 0.98, annot_a, transform=axA.transAxes,
             fontsize=7.6, va="top", ha="left", family="monospace",
             bbox=_annotation_bbox())
    axA.axhline(0, color="0.55", ls=":", lw=0.6)
    axA.set_xlim(t[0], t[-1])
    axA.set_ylabel("twist (°)")
    axA.set_title("(a)  Six region twist curves  (peaks = ★)",
                  fontsize=10, pad=4)
    leg = axA.legend(loc="lower right", fontsize=6.8, ncol=3,
                     frameon=True, columnspacing=0.6, handlelength=1.2)
    _legend_style(leg)
    plt.setp(axA.get_xticklabels(), visible=False)
    _panel_label(axA, "a")

    # ── (b) Linear-gradient fitted twist α(t)·L ─────────────────────────
    axB = fig.add_subplot(gs[1], sharex=axA)
    if np.any(np.isfinite(fitted_twist)):
        axB.plot(t_dense, fitted_twist, "-",
                 color="#000000", lw=2.4, zorder=4)
        axB.fill_between(t_dense, 0, fitted_twist,
                         where=fitted_twist >= 0,
                         color="#d62728", alpha=0.10, zorder=2)
        axB.fill_between(t_dense, 0, fitted_twist,
                         where=fitted_twist < 0,
                         color="#1f77b4", alpha=0.10, zorder=2)
        i_pk     = int(np.argmax(np.abs(fitted_twist)))
        v_pk     = fitted_twist[i_pk]
        t_pk_fit = t_dense[i_pk]
        axB.scatter([t_pk_fit], [v_pk], marker="*", s=110,
                    color="#d62728", edgecolors="black",
                    linewidths=0.6, zorder=6)
        axB.text(0.98, 0.95,
                 f"fitted peak = {v_pk:+.2f}°\n"
                 f"L = {L_axis:.1f} mm",
                 transform=axB.transAxes,
                 fontsize=7.4, va="top", ha="right", family="monospace",
                 bbox=_annotation_bbox())
    if t_avc is not None:
        axB.axvline(t_avc, color="0.4", ls="--", lw=0.9,
                    alpha=0.8, zorder=3)
    axB.axhline(0, color="0.55", ls=":", lw=0.6)
    axB.set_xlim(t[0], t[-1])
    axB.set_xlabel("time (s)")
    axB.set_ylabel("α·L (°)")
    axB.set_title("(b)  Linear-gradient fitted twist "
                  "(uses all 6 regions)", fontsize=10, pad=4)
    _panel_label(axB, "b")

    fig.suptitle(
        f"Figure 11.  Region-wise twist analysis — case study "
        f"{subject_id}",
        fontsize=11, fontweight="bold", y=0.985,
    )
    return _save(fig, f"Figure11_region_analysis_{subject_id}")


# ════════════════════════════════════════════════════════════════════════════
# Combined all_figures.pdf + captions
# ════════════════════════════════════════════════════════════════════════════
def build_combined_pdf(pdf_paths: list[Path]) -> Path:
    out = OUT_DIR / "all_figures.pdf"
    with PdfPages(out) as pp:
        for path in pdf_paths:
            png = path.with_suffix(".png")
            if not png.exists():
                continue
            img = plt.imread(png)
            fig = plt.figure(figsize=(img.shape[1] / 300,
                                      img.shape[0] / 300))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(img); ax.axis("off")
            pp.savefig(fig, dpi=300, bbox_inches="tight", pad_inches=0.0)
            plt.close(fig)
    return out


CAPTIONS = """\
**Figure 1.** Peak regional twist shows a monotonic base-to-apex gradient \
across the pilot cohort (n=5). Subject curves (colored) are overlaid with \
the cohort mean (black line) and ±1 SD envelope (gray band). Apical \
regions (R4–R5) exhibit rotation magnitudes approximately two to three \
times larger than basal regions (R1–R2), consistent with the established \
apex-dominant twist pattern.

**Figure 2.** Regional twist waveforms across the cohort. Each subject's \
spline-smoothed curve is drawn at full length (solid within the common \
observation window, faded beyond each subject's own cycle length). The \
cohort mean (thick black) and ±1 SD band (gray) are computed only over \
the common window (vertical dotted line at 0.87 s; common n_t = 14). \
Per-region y-axes are used to expose mid-cavity dynamics that are \
compressed when amplitudes are forced to share scale.

**Figure 3.** Three-metric coupling between regional twist and strain \
curves: Pearson ρ on raw curves, Pearson ρ on Savitzky–Golay-smoothed \
curves, and dynamic-time-warping distance on min-max-normalized curves. \
Rows separate long-axis (top) and short-axis (bottom) strain. DTW is \
phase-invariant and yields a more compact inter-subject distribution \
than Pearson, supporting its use for 3D-STE data lacking ECG \
synchronization.

**Figure 4.** Region-to-region twist cross-correlation delay (XCD) map \
for subject 03. Each cell reports the lag in milliseconds at which the \
column region's curve best matches the row region's, with positive \
values indicating that the row region leads. Apical regions (R4–R5) \
lead basal regions (R1–R2) by approximately 300–500 ms, consistent \
with an apex-first rotation onset during early systole.

**Figure 5.** Per-subject twist waveforms, one per region. Raw per-frame \
data points are overlaid with the cubic-smoothing-spline fit used for \
feature extraction; the peak is marked with a red star. The systolic \
(pre-peak) phase is shaded light red and the diastolic (post-peak) phase \
light blue. The inset reports peak amplitude, time-to-peak and area \
under the twist–time curve. One file is provided per subject.

**Figure 6.** Distribution of three twist-curve features across regions \
R1–R5. Boxes show the per-region interquartile range and median; \
individual subjects are overlaid as colored dots with small horizontal \
jitter. Systolic and diastolic slope features, which were unstable at \
15 vol/s, are omitted for clarity.

**Figure 7.** Twist and strain phase view for subject 03 across all six \
regions. Each curve is min-max normalized to [0, 1] so shape and phase \
can be compared independently of amplitude. Inset annotations list \
smoothed-Pearson and DTW coupling values; regions with low Pearson can \
still show small DTW when curves share a waveform but differ only by \
phase.

**Figure 10.** Per-subject lifecycle of LV rotation, inspired by — but \
not a copy of — the classical apex/base/net-twist plot. Each subject \
occupies a stacked pair of axes drawn on the subject's own cycle \
length. The upper axis carries six light viridis curves in the \
background, one per z-region (R0 = base → R5 = apex), with two thicker \
overlays: a basal-third mean ⟨R0:R2⟩ (blue) and an apical-third mean \
⟨R3:R5⟩ (orange). The lower axis carries the net twist, defined here \
as apical mean − basal mean (black, with a light fill toward zero), \
plotted on its own y-axis so the difference is easy to read off without \
overlapping the apical / basal traces above. This six-region partition \
into thirds is the definition adopted in the present work; it \
generalises the original two-curve apex/base summary by averaging \
across three regions per end to be less sensitive to noise from any \
single z-band. The vertical dashed line in both axes marks the \
peak-twist time (≈ AVC), used here as a surrogate for aortic-valve \
closure because clinical valve-event timing was unavailable for this \
cohort. Among the seven pilot subjects, only 03 displays the textbook \
signature (apical mean clearly above zero, basal mean clearly below, a \
large smooth net-twist arc); 002A and p066a show degraded apex/base \
separation; p020_1a (MI) and p030_lp2a (LVH / amyloid) show jagged or \
inverted gradients; and p009pa (LBBB, paced) recovers only a partial \
apical/basal separation relative to its pre-pacer counterpart p009npa \
— partial restoration, not normal physiology.
"""


def write_captions() -> Path:
    out = OUT_DIR / "figure_captions.txt"
    with open(out, "w") as f:
        f.write(CAPTIONS)
    return out


# ════════════════════════════════════════════════════════════════════════════
# Figure 12 — cohort metric landscape (Nature-style)
# ════════════════════════════════════════════════════════════════════════════
#
# Three region-wise metrics aggregated across the 7-subject pilot
# cohort, with literature-expected normal bands shaded for reference:
#
#   σ_tpeak      — std of per-region peak times (ms)
#                  expected normal: ~80–150 ms
#   ρ_sys(R0,R5) — apex-base correlation in systole
#                  expected normal: −0.9 to −0.5
#   W_R0R5       — apex-base work covariance (°²)
#                  expected normal: strongly negative
#
# Layout: three vertical strip plots (a–c) + one polar normality radar
# (d) where each subject is one polygon overlaid in green (normal) /
# red (abnormal); the closer a polygon hugs the outer ring the more
# textbook-normal that subject's three metrics jointly are.
# ════════════════════════════════════════════════════════════════════════════
def _f12_compute_cohort_metrics() -> list[dict]:
    """Return one row per subject with the three region-wise metrics."""
    rows: list[dict] = []
    for sid in ALL_SUBJECTS:
        sub = load_subject(sid)
        if sub is None:
            continue
        t       = sub["t_arr"]
        R       = sub["twist_regions"]
        R_raw   = sub["rot_regions"]

        # σ_tpeak  — std of per-region peak times (ms)
        peaks = []
        for r in range(N_REGIONS):
            p = _spline_peak_signed(t, R[:, r])
            if p is not None:
                peaks.append(float(p[0]))
        sigma_tpeak = (float(np.std(peaks) * 1000.0)
                       if len(peaks) >= 2 else float("nan"))

        # AVC* surrogate from apical/basal-third net twist
        spl_net = _spline_peak_signed(
            t, R[:, 3:6].mean(axis=1) - R[:, 0:3].mean(axis=1))
        t_avc   = float(spl_net[0]) if spl_net is not None else None

        # ρ_sys(R0, R5) — Pearson over t ≤ t_avc on RBR-corrected twist
        if t_avc is None:
            rho_sys = float("nan")
        else:
            mask = (t <= t_avc)
            a, b = R[mask, 0], R[mask, 5]
            rho_sys = (float(np.corrcoef(a, b)[0, 1])
                       if int(mask.sum()) >= 3
                       and a.std() > 1e-9 and b.std() > 1e-9
                       else float("nan"))

        # W_R0R5 — covariance on RAW rotation (°²)
        a_raw  = R_raw[:, 0] - R_raw[:, 0].mean()
        b_raw  = R_raw[:, 5] - R_raw[:, 5].mean()
        W_R0R5 = float(np.mean(a_raw * b_raw))

        rows.append({
            "sid":         sid,
            "diag":        smeta.diagnosis_class(sid),
            "sigma_tpeak": sigma_tpeak,
            "rho_sys":     rho_sys,
            "W_R0R5":      W_R0R5,
        })
    return rows


def _f12_normality_scores(r: dict) -> tuple[float, float, float]:
    """Per-metric score in [0, 1] where 1 = textbook normal."""
    s = r["sigma_tpeak"]
    if 80.0 <= s <= 150.0:
        score_sigma = 1.0
    elif s < 80.0:
        score_sigma = max(0.0, s / 80.0)
    else:
        score_sigma = max(0.0, 1.0 - (s - 150.0) / 200.0)

    rho = r["rho_sys"]
    score_rho = 0.0 if not np.isfinite(rho) else max(
        0.0, min(1.0, -rho / 0.7))

    W = r["W_R0R5"]
    if W >= 0:
        score_W = 0.0
    else:
        score_W = min(1.0, abs(W) / 5.0)

    return score_sigma, score_rho, score_W


def _f12_strip_plot(ax, rows: list[dict], key: str, label: str,
                    band_lo: float, band_hi: float,
                    norm_color: str, abn_color: str) -> None:
    band_color = "#bdbdbd"
    ax.axhspan(band_lo, band_hi, color=band_color, alpha=0.35, zorder=1)
    # Reference label
    y_text = band_hi + 0.04 * abs(band_hi - band_lo + 1e-9)
    ax.text(0.97, band_hi, "  expected\n  normal",
            transform=ax.get_yaxis_transform(),
            fontsize=6.4, color="0.45", ha="right", va="bottom")

    # Deterministic but not-too-tight horizontal jitter
    rng    = np.random.default_rng(seed=42)
    x_jit  = rng.uniform(-0.20, 0.20, size=len(rows))
    for i, r in enumerate(rows):
        c = norm_color if r["diag"] == "normal" else abn_color
        v = r[key]
        if not np.isfinite(v):
            continue
        ax.scatter(x_jit[i], v, s=140, color=c, edgecolors="black",
                   linewidth=0.8, alpha=0.90, zorder=4)
        ax.annotate(r["sid"], xy=(x_jit[i], v),
                    xytext=(11, 0), textcoords="offset points",
                    fontsize=7.5, va="center", color=c,
                    fontweight="bold")
    ax.set_xticks([])
    ax.set_xlim(-0.55, 0.85)
    ax.set_ylabel(label, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.30, zorder=0)


def figure12_cohort_metric_landscape() -> tuple[Path, ...]:
    rows = _f12_compute_cohort_metrics()

    NORM_COLOR = "#1b7837"
    ABN_COLOR  = "#b2182b"

    fig = plt.figure(figsize=(13.5, 4.6))
    gs  = fig.add_gridspec(1, 4, wspace=0.42,
                           left=0.045, right=0.985,
                           top=0.85, bottom=0.13)

    axA = fig.add_subplot(gs[0, 0])
    _f12_strip_plot(axA, rows, "sigma_tpeak", "σ_tpeak (ms)",
                    80.0, 150.0, NORM_COLOR, ABN_COLOR)
    axA.set_title("(a)  Peak-time spread", fontsize=10, pad=4)
    _panel_label(axA, "a")

    axB = fig.add_subplot(gs[0, 1])
    _f12_strip_plot(axB, rows, "rho_sys", "ρ_sys(R0, R5)",
                    -0.9, -0.5, NORM_COLOR, ABN_COLOR)
    axB.set_title("(b)  Apex-base correlation (systole)",
                  fontsize=10, pad=4)
    _panel_label(axB, "b")

    axC = fig.add_subplot(gs[0, 2])
    _f12_strip_plot(axC, rows, "W_R0R5", "W_R0R5 (°²)",
                    -10.0, -2.0, NORM_COLOR, ABN_COLOR)
    axC.set_title("(c)  Apex-base work covariance",
                  fontsize=10, pad=4)
    _panel_label(axC, "c")

    # ── (d) Polar normality radar ─────────────────────────────────────
    axD = fig.add_subplot(gs[0, 3], projection="polar")
    angles        = np.linspace(0.0, 2 * np.pi, 4)[:3]
    angles_closed = np.concatenate([angles, [angles[0]]])

    # Soft "perfect normal" reference ring
    ref_scores        = np.ones(3)
    ref_scores_closed = np.concatenate([ref_scores, [ref_scores[0]]])
    axD.plot(angles_closed, ref_scores_closed, "-",
             color="0.55", lw=0.9, alpha=0.7)

    for r in rows:
        s_sigma, s_rho, s_W = _f12_normality_scores(r)
        scores              = np.array([s_sigma, s_rho, s_W])
        scores_closed       = np.concatenate([scores, [scores[0]]])
        c = NORM_COLOR if r["diag"] == "normal" else ABN_COLOR
        axD.plot(angles_closed, scores_closed, "-",
                 color=c, lw=1.7, alpha=0.85, zorder=4)
        axD.fill(angles_closed, scores_closed, color=c,
                 alpha=0.12, zorder=2)
        # Subject id near outer score
        ang_max  = float(angles[int(np.argmax(scores))])
        score_max = float(np.max(scores))
        axD.text(ang_max, score_max + 0.07, r["sid"],
                 color=c, fontsize=6.6, ha="center", va="center",
                 fontweight="bold")

    axD.set_xticks(angles)
    axD.set_xticklabels(["σ_tpeak", "ρ_sys", "W_R0R5"], fontsize=8.5)
    axD.set_ylim(0, 1.15)
    axD.set_yticks([0.25, 0.5, 0.75, 1.0])
    axD.set_yticklabels([])
    axD.tick_params(axis="x", pad=8)
    axD.set_title("(d)  Normality radar  (1 = textbook normal)",
                  fontsize=9.5, pad=22)
    _panel_label(axD, "d")

    # Footer legend
    fig.text(0.045, 0.04,
             "● normal", color=NORM_COLOR, fontsize=9, fontweight="bold")
    fig.text(0.105, 0.04,
             "● abnormal", color=ABN_COLOR, fontsize=9, fontweight="bold")

    fig.suptitle(
        "Figure 12.  Region-wise metric landscape across the pilot cohort",
        fontsize=11, fontweight="bold", y=0.97)

    return _save(fig, "Figure12_cohort_metric_landscape")


# ════════════════════════════════════════════════════════════════════════════
# Figure 13 — Sengupta-style 4-panel phase view
# ════════════════════════════════════════════════════════════════════════════
#
# Composition deliberately replicates the classical "LV twist in health
# and disease" multi-panel phase plot (Sengupta 2008; Sengupta et al.,
# JACC Cardiovasc Imaging 2014):
#   • 2×2 grid of representative subjects (A, B, C, D)
#   • Each panel shows three rotation traces (apical mean / basal mean /
#     net twist) on a shared y-axis
#   • Four vertical phase markers + one dashed end-of-cycle line define
#     four labelled phases (1, 2, 3, 4)
#   • The third marker is the AVC* surrogate (peak of net twist)
#   • A bottom strip carries the net-twist rate (dθ/dt), which fills
#     the role of the ECG strip in the reference figure (we have no ECG)
# Phase boundaries are derived from the data, not from clinical events:
#     T1 = 0.30·t_avc       T2 = 0.75·t_avc
#     T3 = t_avc  (AVC*)    T4 = t_avc + 0.50·(t_end − t_avc)
# i.e. the cycle is split into four data-driven segments around the
# AVC* surrogate.  The pre-T1 region is unlabelled (analogous to the
# unlabelled atrial-systole / late-diastole interval in the reference).
# ════════════════════════════════════════════════════════════════════════════
F13_SUBJECTS = ["03", "p009npa", "p020_1a", "p030_lp2a"]
F13_LABELS   = {
    "03":         "03   (normal)",
    "p009npa":    "p009npa   (LBBB pre-pacer)",
    "p020_1a":    "p020_1a   (MI)",
    "p030_lp2a":  "p030_lp2a   (LVH / amyloid)",
}
F13_APEX_COLOR = "#e67e22"   # bright orange   — apical mean
F13_BASE_COLOR = "#f1c40f"   # warm yellow     — basal mean
F13_NET_COLOR  = "#5d4037"   # dark brown      — net twist
F13_PHASE_LINE = "#3498db"   # vivid blue      — phase markers
F13_END_LINE   = "0.30"      # dark gray       — dashed end-of-cycle
F13_RATE_COLOR = "#1f3a52"   # navy            — bottom-strip trace


def _f13_smooth(t: np.ndarray, y: np.ndarray, t_dense: np.ndarray
                ) -> np.ndarray:
    spl = _spline_fit(t, y)
    return spl(t_dense) if spl is not None else np.interp(t_dense, t, y)


def figure13_phase_view() -> tuple[Path, ...]:
    subs: dict[str, dict] = {}
    for sid in F13_SUBJECTS:
        s = load_subject(sid)
        if s is not None:
            subs[sid] = s
    if not subs:
        raise RuntimeError("Figure 13: no subjects loaded.")

    # Common y-range across all 4 panels (matches reference's shared scale)
    rot_min, rot_max = +np.inf, -np.inf
    for s in subs.values():
        R       = s["rot_regions"]
        basal_t = R[:, 0:3].mean(axis=1)
        apex_t  = R[:, 3:6].mean(axis=1)
        net_t   = apex_t - basal_t
        for arr in (basal_t, apex_t, net_t):
            rot_min = min(rot_min, float(np.nanmin(arr)))
            rot_max = max(rot_max, float(np.nanmax(arr)))
    pad     = (rot_max - rot_min) * 0.10
    rot_lim = (np.floor((rot_min - pad) / 3.0) * 3.0,
               np.ceil ((rot_max + pad) / 3.0) * 3.0)

    fig      = plt.figure(figsize=(13.0, 9.2))
    outer_gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.16,
                                left=0.06, right=0.985,
                                top=0.86, bottom=0.06)
    panel_letters = list("ABCD")

    for idx, sid in enumerate(F13_SUBJECTS):
        if sid not in subs:
            continue
        sub      = subs[sid]
        t        = sub["t_arr"]
        t_ms     = t * 1000.0
        R        = sub["rot_regions"]
        basal    = R[:, 0:3].mean(axis=1)
        apical   = R[:, 3:6].mean(axis=1)
        net      = apical - basal

        # AVC* and phase boundaries
        spl_net = _spline_peak_signed(t, net)
        t_avc   = (float(spl_net[0]) if spl_net is not None
                   else float(t[len(t) // 2]))
        t_end   = float(t[-1])
        T1, T2  = 0.30 * t_avc, 0.75 * t_avc
        T3, T4  = t_avc, t_avc + 0.50 * (t_end - t_avc)

        # Spline-smooth dense traces
        t_dense    = np.linspace(t[0], t_end, 240)
        t_dense_ms = t_dense * 1000.0
        apical_d   = _f13_smooth(t, apical, t_dense)
        basal_d    = _f13_smooth(t, basal,  t_dense)
        net_d      = _f13_smooth(t, net,    t_dense)

        # Inner gridspec — main curves on top, rate strip below
        inner   = outer_gs[idx // 2, idx % 2].subgridspec(
            2, 1, height_ratios=[3.6, 0.85], hspace=0.04)
        ax_main = fig.add_subplot(inner[0])
        ax_bot  = fig.add_subplot(inner[1], sharex=ax_main)

        # Background phase shading (very light) for readability
        for x_lo, x_hi, alpha in (
                (T1, T2, 0.04), (T2, T3, 0.06),
                (T3, T4, 0.04), (T4, t_end, 0.02)):
            ax_main.axvspan(x_lo * 1000, x_hi * 1000,
                            color="0.7", alpha=alpha, zorder=1)

        # Three rotation traces
        ax_main.plot(t_dense_ms, apical_d, "-",
                     color=F13_APEX_COLOR, lw=2.0, zorder=4)
        ax_main.plot(t_dense_ms, basal_d, "-",
                     color=F13_BASE_COLOR, lw=2.0, zorder=4)
        ax_main.plot(t_dense_ms, net_d, "-",
                     color=F13_NET_COLOR, lw=2.2, zorder=5)

        # Phase markers + dashed end
        for T_marker in (T1, T2, T3, T4):
            for ax in (ax_main, ax_bot):
                ax.axvline(T_marker * 1000, color=F13_PHASE_LINE,
                           lw=1.0, alpha=0.85, zorder=3)
        for ax in (ax_main, ax_bot):
            ax.axvline(t_end * 1000, color=F13_END_LINE,
                       ls="--", lw=0.9, alpha=0.85, zorder=3)

        # AVC* label above T3
        ax_main.text(T3 * 1000, rot_lim[1] - 0.5, "AVC*",
                     fontsize=8.5, color=F13_PHASE_LINE,
                     ha="center", va="top", fontweight="bold")

        # Phase numbers (1..4) between markers, near the bottom
        phase_centers = [
            ((T1 + T2)   / 2.0) * 1000,
            ((T2 + T3)   / 2.0) * 1000,
            ((T3 + T4)   / 2.0) * 1000,
            ((T4 + t_end) / 2.0) * 1000,
        ]
        y_phase_label = rot_lim[0] + 1.0
        for j, x_center in enumerate(phase_centers):
            ax_main.text(x_center, y_phase_label, str(j + 1),
                         fontsize=11.5, fontweight="bold",
                         color="0.30", ha="center", va="bottom")

        # Zero baseline
        ax_main.axhline(0, color="0.55", lw=0.5, alpha=0.6, zorder=2)

        # Y-axis
        ax_main.set_ylim(rot_lim)
        y_step  = 6.0
        y_ticks = np.arange(rot_lim[0], rot_lim[1] + 0.1, y_step)
        ax_main.set_yticks(y_ticks)
        ax_main.set_xlim(0, t_end * 1000)
        plt.setp(ax_main.get_xticklabels(), visible=False)
        ax_main.tick_params(labelsize=8)
        for s_name in ("top", "right"):
            ax_main.spines[s_name].set_visible(False)

        # Panel letter (A/B/C/D, bold)
        ax_main.text(-0.085, 1.06, panel_letters[idx],
                     transform=ax_main.transAxes,
                     fontsize=15, fontweight="bold", color="0.10")
        # Subject id + diagnosis (small subtitle, top-left of panel)
        ax_main.text(0.015, 1.02, F13_LABELS.get(sid, sid),
                     transform=ax_main.transAxes,
                     fontsize=8.5, color="0.30",
                     fontweight="bold", va="bottom")

        # Bottom strip — net-twist rate (dθ/dt), centred about zero
        dt_ms_step = (t_dense[1] - t_dense[0]) * 1000.0
        rate       = np.gradient(net_d, dt_ms_step)
        rate       = rate - float(np.mean(rate))
        ax_bot.plot(t_dense_ms, rate, "-",
                    color=F13_RATE_COLOR, lw=0.9, zorder=4)
        rmax = float(np.nanmax(np.abs(rate))) if rate.size else 1.0
        ax_bot.set_ylim(-rmax * 1.15, rmax * 1.15)
        ax_bot.set_yticks([])
        ax_bot.set_xlim(0, t_end * 1000)
        ax_bot.tick_params(labelsize=7.5)
        for s_name in ("top", "right", "left"):
            ax_bot.spines[s_name].set_visible(False)
        ax_bot.spines["bottom"].set_linewidth(0.6)
        ax_bot.set_xlabel("time (ms)", fontsize=8)

    # Curve-color legend at top
    handles = [
        Line2D([0], [0], color=F13_APEX_COLOR, lw=2.2,
               label="apical mean  ⟨R3:R5⟩"),
        Line2D([0], [0], color=F13_BASE_COLOR, lw=2.2,
               label="basal mean  ⟨R0:R2⟩"),
        Line2D([0], [0], color=F13_NET_COLOR,  lw=2.4,
               label="net twist  =  apical − basal"),
        Line2D([0], [0], color=F13_PHASE_LINE, lw=1.4,
               label="phase boundary  /  AVC*"),
        Line2D([0], [0], color=F13_RATE_COLOR, lw=0.9,
               label="dθ/dt  (rate strip)"),
    ]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.92), ncol=5,
               fontsize=8.5, frameon=True, framealpha=0.85,
               edgecolor="0.7")

    fig.suptitle(
        "Figure 13.  Sengupta-style four-panel phase view "
        "(structure replicated; data are ours)",
        fontsize=11.5, fontweight="bold", y=0.975)

    return _save(fig, "Figure13_phase_view")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════
def main() -> None:
    print(f"OUT_DIR = {OUT_DIR}")
    produced: list[tuple[str, Path, Path, Path | None]] = []

    print("\n[Figure 1]")
    pdf, png, svg = figure1_apex_base_gradient()
    produced.append(("Figure 1", pdf, png, svg)); print(f"  {pdf}")

    print("\n[Figure 2]")
    pdf, png, svg = figure2_cohort_twist_overlay()
    produced.append(("Figure 2", pdf, png, svg)); print(f"  {pdf}")

    print("\n[Figure 3]  (central novelty figure — 2×3 heatmap)")
    pdf, png, svg = figure3_coupling_comparison()
    produced.append(("Figure 3", pdf, png, svg)); print(f"  {pdf}")

    print("\n[Figure 4]")
    pdf, png, svg = figure4_xcd_subject03()
    produced.append(("Figure 4", pdf, png, svg)); print(f"  {pdf}")

    print("\n[Figure 5] (per subject)")
    master_rows = load_master_csv()
    for sid in ALL_SUBJECTS:
        res = figure5_twist_features(sid, master_rows)
        if res is None:
            print(f"  {sid}: missing — skipped")
            continue
        pdf, png, svg = res
        produced.append((f"Figure 5 ({sid})", pdf, png, svg)); print(f"  {pdf}")

    print("\n[Figure 6]")
    pdf, png, svg = figure6_feature_distributions()
    produced.append(("Figure 6", pdf, png, svg)); print(f"  {pdf}")

    print("\n[Figure 7]")
    pdf, png, svg = figure7_phase_view_subject03()
    produced.append(("Figure 7", pdf, png, svg)); print(f"  {pdf}")

    print("\n[Figure 9]  (cohort angle-change dominance summary)")
    pdf, png, svg = figure9_angle_dominance_summary()
    produced.append(("Figure 9", pdf, png, svg)); print(f"  {pdf}")

    print("\n[Figure 10]  (twist lifecycle — apical / basal / net)")
    pdf, png, svg = figure10_twist_lifecycle()
    produced.append(("Figure 10", pdf, png, svg)); print(f"  {pdf}")

    print("\n[Figure 11]  (region-wise case study, per subject)")
    for sid in ALL_SUBJECTS:
        res = figure11_region_analysis(sid)
        if res is None:
            print(f"  {sid}: missing — skipped")
            continue
        pdf, png, svg = res
        produced.append((f"Figure 11 ({sid})", pdf, png, svg))
        print(f"  {pdf}")

    print("\n[Figure 12]  (cohort metric landscape — Nature style)")
    pdf, png, svg = figure12_cohort_metric_landscape()
    produced.append(("Figure 12", pdf, png, svg)); print(f"  {pdf}")

    print("\n[Figure 13]  (Sengupta-style 4-panel phase view)")
    pdf, png, svg = figure13_phase_view()
    produced.append(("Figure 13", pdf, png, svg)); print(f"  {pdf}")

    # Per-subject dominant-region printout + CSV
    print("\n[Cohort angle dominance]  "
          f"(peak picked in middle {int(PEAK_MIDDLE_FRAC*100)}% of cycle)")
    dom_csv = write_cohort_angle_dominance_csv()
    print(f"  CSV → {dom_csv}")
    subjects_seen, regions_seen, peaks_mat, times_mat, per_subj = \
        _collect_cohort_spline_peaks_with_times()
    apical_tally = np.zeros(len(regions_seen), dtype=int)
    print(f"  {'subject':<10} {'diag':<10}  apical-dominant   "
          f"basal-dominant")
    for sid in subjects_seen:
        ps = per_subj[sid]
        if not ps:
            print(f"  {sid}: no valid region peaks")
            continue
        ap_r = max(ps, key=lambda r: ps[r][1])           # signed argmax
        bs_r = min(ps, key=lambda r: ps[r][1])           # signed argmin
        ap_v = ps[ap_r][1]
        bs_v = ps[bs_r][1]
        apical_tally[regions_seen.index(ap_r)] += 1
        diag = smeta.diagnosis_class(sid)
        print(f"  {sid:<10} {diag:<10}  R{ap_r} ({ap_v:+.2f}°)     "
              f"R{bs_r} ({bs_v:+.2f}°)")

    print("\n[Cohort apical-dominance tally] (signed argmax)")
    for r, c in zip(regions_seen, apical_tally):
        print(f"  R{r}: {int(c)}/{len(subjects_seen)} subjects "
              f"apex-like-dominant")
    apical_count = int(apical_tally[regions_seen.index(4)]
                       + apical_tally[regions_seen.index(5)])
    if apical_count == 0:
        print("  WARNING: no subject has its apex-like-dominant region in "
              "R4/R5 — unusual for LV twist; investigate data quality or "
              "tracking-sign handling.")

    print("\n[Combined PDF + captions]")
    combined = build_combined_pdf([p for _, p, _, _ in produced])
    print(f"  {combined}")
    caps = write_captions()
    print(f"  {caps}")

    # ── Verification ─────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("VERIFICATION")
    print("=" * 64)
    fig3_pdf = OUT_DIR / "Figure3_coupling_comparison.pdf"
    fig3_png = OUT_DIR / "Figure3_coupling_comparison.png"
    assert fig3_pdf.exists(), f"MISSING: {fig3_pdf}"
    assert fig3_png.exists(), f"MISSING: {fig3_png}"
    print(f"  [OK] {fig3_pdf.name} present "
          f"({fig3_pdf.stat().st_size/1024:.1f} KB)")
    print(f"  [OK] {fig3_png.name} present "
          f"({fig3_png.stat().st_size/1024:.1f} KB)")

    all_ok = True
    total = 0
    print("\n  file listing:")
    for name, pdf, png, svg in produced:
        sp, sg = pdf.stat().st_size, png.stat().st_size
        total += sp + sg
        oversized = (sp > 5 * 1024 * 1024) or (sg > 5 * 1024 * 1024)
        tag = "  OVERSIZED" if oversized else ""
        print(f"    {name:<22}  pdf={sp/1024:7.1f} KB  "
              f"png={sg/1024:7.1f} KB{tag}")
        if svg is not None:
            ss = svg.stat().st_size
            total += ss
            print(f"    {'':<22}  svg={ss/1024:7.1f} KB")
        if oversized:
            all_ok = False

    print(f"    all_figures.pdf          "
          f"size={combined.stat().st_size/1024:7.1f} KB")
    print(f"    figure_captions.txt      "
          f"size={caps.stat().st_size/1024:7.1f} KB")
    print(f"\n  TOTAL bundle   : {total/1024:.1f} KB")
    print(f"  All files <5 MB: {'YES' if all_ok else 'NO'}")
    print(f"  Typography     : "
          f"title {mpl.rcParams['axes.titlesize']} pt "
          f"> tick {mpl.rcParams['xtick.labelsize']} pt (expected)")
    print(f"  Legend alpha   : "
          f"{mpl.rcParams['legend.framealpha']} (target 0.55)")


if __name__ == "__main__":
    main()
