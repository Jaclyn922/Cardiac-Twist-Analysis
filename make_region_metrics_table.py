"""make_region_metrics_table.py — refined 3-metric region-wise summary.

Three metrics (each targeting a specific physiology / pathology
pattern):

    σ_tpeak       — Peak-time spread          (timing dyssynchrony vs RBR)
    ρ_sys(R0,R5)  — Apex-base correlation in systole
                                              (wringing preservation)
    W_R0R5        — Apex-base work covariance
                                              (amplitude-weighted wringing)

The systolic window is t ∈ [0, t_avc], where t_avc is the peak time of
the apical-third / basal-third net twist (used as an AVC* surrogate
elsewhere in the paper).

Data sources
------------
σ_tpeak       : sub["twist_regions"]  — RBR-corrected (preserved value).
ρ_sys         : sub["twist_regions"]  — RBR-corrected; isolates the
                                        wringing component, so the
                                        systolic apex-base correlation
                                        is not contaminated by the
                                        rigid-body rotation.
W_R0R5        : sub["rot_regions"]    — RAW rotation, sign-flipped;
                                        weights by raw amplitude (°²).

Outputs
-------
figures_paper/region_metrics_table.csv
figures_paper/region_metrics_table.md
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

import publication_figures_v2 as pf
import subject_metadata as smeta

DATA_ROOT = Path("/data/xinge/Cardiac-Twist-Analysis")
OUT_DIR   = DATA_ROOT / "figures_paper"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_REGIONS = 6


def _t_avc(R_twist: np.ndarray, t: np.ndarray) -> float | None:
    """Peak time of the apical-third / basal-third net twist (AVC*)."""
    basal_t  = R_twist[:, 0:3].mean(axis=1)
    apical_t = R_twist[:, 3:6].mean(axis=1)
    spl      = pf._spline_peak_signed(t, apical_t - basal_t)
    return None if spl is None else float(spl[0])


# ──────────────────────────────────────────────────────────────────────────
# σ_tpeak — std of per-region peak times (ms)
# ──────────────────────────────────────────────────────────────────────────
def metric_sigma_tpeak_ms(R_twist: np.ndarray, t: np.ndarray) -> float:
    times = []
    for r in range(N_REGIONS):
        p = pf._spline_peak_signed(t, R_twist[:, r])
        if p is not None:
            times.append(float(p[0]))
    return float(np.std(times) * 1000.0) if len(times) >= 2 else float("nan")


# ──────────────────────────────────────────────────────────────────────────
# ρ_sys(R0, R5) — Pearson ρ between R0 and R5 over the systolic window
# ──────────────────────────────────────────────────────────────────────────
def metric_rho_sys_R0_R5(R_twist: np.ndarray, t: np.ndarray,
                         t_avc: float | None) -> float:
    if t_avc is None:
        return float("nan")
    mask = (t <= t_avc)
    if int(mask.sum()) < 3:
        return float("nan")
    a = R_twist[mask, 0]
    b = R_twist[mask, 5]
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# ──────────────────────────────────────────────────────────────────────────
# W_R0R5 — covariance of raw R0 and R5 (°²), amplitude-weighted wringing
# ──────────────────────────────────────────────────────────────────────────
def metric_W_R0R5(R_raw: np.ndarray) -> float:
    a = R_raw[:, 0] - R_raw[:, 0].mean()
    b = R_raw[:, 5] - R_raw[:, 5].mean()
    return float(np.mean(a * b))


def compute_all_metrics(sid: str) -> dict | None:
    sub = pf.load_subject(sid)
    if sub is None:
        return None
    t       = sub["t_arr"]
    R_twist = sub["twist_regions"]   # RBR-corrected, sign-flipped
    R_raw   = sub["rot_regions"]     # RAW rotation,   sign-flipped
    t_avc   = _t_avc(R_twist, t)

    return {
        "subject_id":   sid,
        "diagnosis":    smeta.diagnosis_class(sid),
        "t_avc_s":      (float(t_avc) if t_avc is not None else float("nan")),
        "sigma_tpeak":  metric_sigma_tpeak_ms(R_twist, t),
        "rho_sys":      metric_rho_sys_R0_R5(R_twist, t, t_avc),
        "W_R0R5":       metric_W_R0R5(R_raw),
    }


def main() -> None:
    rows: list[dict] = []
    for sid in pf.ALL_SUBJECTS:
        m = compute_all_metrics(sid)
        if m is not None:
            rows.append(m)

    fields = ["subject_id", "diagnosis", "t_avc_s",
              "sigma_tpeak", "rho_sys", "W_R0R5"]

    # ── CSV ──────────────────────────────────────────────────────────────
    csv_path = OUT_DIR / "region_metrics_table.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            row_out = {}
            for k in fields:
                v = r[k]
                if isinstance(v, float):
                    row_out[k] = (f"{v:.4f}" if np.isfinite(v) else "nan")
                else:
                    row_out[k] = str(v)
            w.writerow(row_out)

    # ── Markdown ─────────────────────────────────────────────────────────
    md_path = OUT_DIR / "region_metrics_table.md"
    with open(md_path, "w") as f:
        f.write(
            "| Subject "
            "| σ_tpeak (ms) "
            "| ρ_sys(R0,R5) "
            "| W_R0R5 (°²) "
            "| t_avc (s) |\n")
        f.write("|---|---|---|---|---|\n")
        for r in rows:
            tag = f"{r['subject_id']} ({r['diagnosis']})"
            f.write(
                f"| {tag} "
                f"| {r['sigma_tpeak']:.0f} "
                f"| {r['rho_sys']:+.2f} "
                f"| {r['W_R0R5']:+.2f} "
                f"| {r['t_avc_s']:.2f} |\n"
            )
        f.write(
            "\n_Definitions:_\n"
            "- **σ_tpeak (ms)** — Peak-time spread.  Std of per-region "
            "peak times across R0..R5 (computed on RBR-corrected twist). "
            "_Captures timing dyssynchrony and RBR_.  Expected: "
            "**LBBB high (> 200 ms)**, amyloid / RBR low (< 80 ms), "
            "normal middle (~80–150 ms).\n"
            "- **ρ_sys(R0, R5)** — Apex-base correlation in systole.  "
            "Pearson ρ between R0 and R5 over t ∈ [0, t_avc], on RBR-"
            "corrected twist.  _Captures wringing preservation_.  "
            "Expected: **normal strongly negative**, RBR / loss of "
            "wringing → 0.\n"
            "- **W_R0R5 (°²)** — Apex-base work covariance.  "
            "Cov(rot_R0, rot_R5) over the whole cycle, on RAW rotation; "
            "units = degrees-squared because amplitudes are not "
            "normalised.  _Captures amplitude-weighted wringing_.  "
            "Expected: **normal strongly negative** (large counter-"
            "rotating amplitudes), abnormal → 0.\n"
            "- **t_avc (s)** — auxiliary column: peak time of the apical-"
            "third / basal-third net twist, used as the AVC* surrogate "
            "to define the systolic window for ρ_sys.\n"
        )

    # ── Console summary ──────────────────────────────────────────────────
    print(f"Saved → {csv_path}")
    print(f"Saved → {md_path}\n")

    print(f"{'subject':<11} {'diag':<9} "
          f"{'σ_tpk':>6} {'ρ_sys':>7} "
          f"{'W_R0R5':>8} {'t_avc':>6}")
    for r in rows:
        print(f"  {r['subject_id']:<10} {r['diagnosis']:<8} "
              f"{r['sigma_tpeak']:6.0f} "
              f"{r['rho_sys']:+7.2f} "
              f"{r['W_R0R5']:+8.2f} "
              f"{r['t_avc_s']:6.2f}")


if __name__ == "__main__":
    main()
