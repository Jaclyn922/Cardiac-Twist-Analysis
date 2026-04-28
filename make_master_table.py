"""make_master_table.py — emit cohort_master_table.{csv,md}

Aggregates per-(subject, region) metrics from the cached NPZ files into one
master table for direct use in the report.  All peak values use the same
middle-80% signed-peak rule as Figure 5 and Figure 9.

Reads only; no pipeline recomputation.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

import publication_figures_v2 as pf
import subject_metadata as smeta

DATA_ROOT    = Path("/data/xinge/Cardiac-Twist-Analysis")
OUT_DIR      = DATA_ROOT / "figures_paper"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ALL_SUBJECTS = ["002A", "03", "p009npa", "p009pa", "p066a",
                "p020_1a", "p030_lp2a"]
REGIONS      = [0, 1, 2, 3, 4, 5]


# ─── Data access ────────────────────────────────────────────────────────────
def _load_subject(sid: str) -> dict | None:
    res = DATA_ROOT / sid / "result"
    need = ("strain_twist_analysis.npz", "dtw_analysis.npz")
    if not all((res / n).exists() for n in need):
        return None
    an = np.load(res / "strain_twist_analysis.npz", allow_pickle=True)
    dt = np.load(res / "dtw_analysis.npz")
    sign_flipped = bool(dt["sign_flipped"])

    twist = an["twist_regions"].astype(np.float64).copy()
    if sign_flipped:
        twist *= -1.0
    nt = twist.shape[0]
    fr_hz, _ = pf.subject_frame_rate(sid)
    t  = np.arange(nt) / fr_hz

    return {
        "sid":                sid,
        "nt":                 nt,
        "t_arr":              t,
        "frame_rate":         fr_hz,
        "twist_regions":      twist,
        "u_long_regions":     an["u_long_regions"].astype(np.float64),
        "u_short_regions":    an["u_short_regions"].astype(np.float64),
        "rho_long_smooth":    dt["pearson_twist_long_smooth"]
                                .astype(np.float64),
        "rho_short_smooth":   dt["pearson_twist_short_smooth"]
                                .astype(np.float64),
        "dtw_twist_long":     dt["dtw_twist_long"].astype(np.float64),
        "dtw_twist_short":    dt["dtw_twist_short"].astype(np.float64),
        "sign_flipped":       sign_flipped,
    }


def _load_dominant_map() -> dict[str, int]:
    """Read apical-dominant region per subject from cohort_angle_dominance.csv.

    Accepts the compact schema (apical_dom = "R3") and the legacy schemas
    (apical_dominant_region / dominant_region as integer).
    """
    dom: dict[str, int] = {}
    with open(DATA_ROOT / "cohort_angle_dominance.csv") as f:
        for row in csv.DictReader(f):
            if "apical_dom" in row:
                v = row["apical_dom"].strip()
                dom[row["subject_id"]] = int(v.lstrip("R")) if v.startswith("R") else -1
            else:
                key = ("apical_dominant_region"
                       if "apical_dominant_region" in row
                       else "dominant_region")
                dom[row["subject_id"]] = int(row[key])
    return dom


def _mid_peak(t_arr: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Signed middle-window peak via the publication_figures_v2 helper."""
    p = pf._spline_peak_signed(t_arr, y)
    if p is None:
        return float("nan"), float("nan")
    return float(p[0]), float(p[1])


# ─── Row construction ──────────────────────────────────────────────────────
COLUMNS = [
    "subject_id", "diagnosis", "region",
    "nt", "duration_s",
    "twist_peak_deg",   "twist_peak_time_s",
    "u_long_peak_mm",   "u_long_peak_time_s",
    "u_short_peak_mm",  "u_short_peak_time_s",
    "pearson_twist_ulong", "pearson_twist_ushort",
    "dtw_twist_ulong",     "dtw_twist_ushort",
    "is_dominant", "notes",
]


def _note_str(sub: dict) -> str:
    notes = []
    if sub["sign_flipped"]:
        notes.append("sign flipped")
    if sub["nt"] < 18:
        notes.append("short cycle")
    return "; ".join(notes)


def _subject_region_row(sub: dict, region: int, dom_r: int) -> dict:
    t_arr = sub["t_arr"]
    tw_r  = sub["twist_regions"][:, region]
    uL_r  = sub["u_long_regions"][:, region]
    uS_r  = sub["u_short_regions"][:, region]

    tw_t, tw_v = _mid_peak(t_arr, tw_r)
    uL_t, uL_v = _mid_peak(t_arr, uL_r)
    uS_t, uS_v = _mid_peak(t_arr, uS_r)
    # Coupling arrays are still indexed by R1..R5 (length 5).  R0 has no
    # stored coupling values under the legacy schema, so leave them NaN.
    if region == 0:
        rho_L = rho_S = float("nan")
        dtw_L = dtw_S = float("nan")
    else:
        idx = region - 1
        rho_L = float(sub["rho_long_smooth"][idx])
        rho_S = float(sub["rho_short_smooth"][idx])
        dtw_L = float(sub["dtw_twist_long"][idx])
        dtw_S = float(sub["dtw_twist_short"][idx])

    return {
        "subject_id":            sub["sid"],
        "diagnosis":             smeta.diagnosis_class(sub["sid"]),
        "region":                f"R{region}",
        "nt":                    int(sub["nt"]),
        "duration_s":            float(sub["nt"]) / float(sub["frame_rate"]),
        "twist_peak_deg":        tw_v,
        "twist_peak_time_s":     tw_t,
        "u_long_peak_mm":        uL_v,
        "u_long_peak_time_s":    uL_t,
        "u_short_peak_mm":       uS_v,
        "u_short_peak_time_s":   uS_t,
        "pearson_twist_ulong":   rho_L,
        "pearson_twist_ushort":  rho_S,
        "dtw_twist_ulong":       dtw_L,
        "dtw_twist_ushort":      dtw_S,
        "is_dominant":           (region == dom_r),
        "notes":                 _note_str(sub),
    }


def _cohort_summary_row(region: int, rows: list[dict],
                        cohort_filter: str = "all") -> dict:
    """Mean ± SD across subjects for a single region (R1..R5).

    `cohort_filter`: "all", "normal", or "abnormal".
    """
    rg = f"R{region}"
    if cohort_filter == "all":
        sel = [r for r in rows if r["region"] == rg
               and r["subject_id"] != "cohort"]
    else:
        sel = [r for r in rows if r["region"] == rg
               and r["subject_id"] != "cohort"
               and r["diagnosis"] == cohort_filter]
    tw_pk  = np.array([r["twist_peak_deg"]  for r in sel], dtype=float)
    uL_pk  = np.array([r["u_long_peak_mm"]  for r in sel], dtype=float)
    uS_pk  = np.array([r["u_short_peak_mm"] for r in sel], dtype=float)

    def _ms(a: np.ndarray) -> str:
        if not np.any(np.isfinite(a)):
            return "nan"
        m = float(np.nanmean(a))
        s = float(np.nanstd(a))
        return f"{m:+.2f} ± {s:.2f}"

    label = {"all":      "cohort",
             "normal":   "cohort (normal)",
             "abnormal": "cohort (abnormal)"}[cohort_filter]

    return {
        "subject_id":            label,
        "diagnosis":             cohort_filter,
        "region":                rg,
        "nt":                    "",
        "duration_s":            "",
        "twist_peak_deg":        _ms(tw_pk),
        "twist_peak_time_s":     "",
        "u_long_peak_mm":        _ms(uL_pk),
        "u_long_peak_time_s":    "",
        "u_short_peak_mm":       _ms(uS_pk),
        "u_short_peak_time_s":   "",
        "pearson_twist_ulong":   "",
        "pearson_twist_ushort":  "",
        "dtw_twist_ulong":       "",
        "dtw_twist_ushort":      "",
        "is_dominant":           "",
        "notes":                 f"mean ± SD across {len(sel)} subjects",
    }


# ─── Formatting ────────────────────────────────────────────────────────────
NUMERIC_COLS = {
    "duration_s":           lambda v: f"{v:.2f}",
    "twist_peak_deg":       lambda v: f"{v:+.2f}",
    "twist_peak_time_s":    lambda v: f"{v:.2f}",
    "u_long_peak_mm":       lambda v: f"{v:+.2f}",
    "u_long_peak_time_s":   lambda v: f"{v:.2f}",
    "u_short_peak_mm":      lambda v: f"{v:+.2f}",
    "u_short_peak_time_s":  lambda v: f"{v:.2f}",
    "pearson_twist_ulong":  lambda v: f"{v:+.2f}",
    "pearson_twist_ushort": lambda v: f"{v:+.2f}",
    "dtw_twist_ulong":      lambda v: f"{v:.2f}",
    "dtw_twist_ushort":     lambda v: f"{v:.2f}",
}

RIGHT_ALIGN_COLS = {
    "nt", "duration_s",
    "twist_peak_deg",   "twist_peak_time_s",
    "u_long_peak_mm",   "u_long_peak_time_s",
    "u_short_peak_mm",  "u_short_peak_time_s",
    "pearson_twist_ulong", "pearson_twist_ushort",
    "dtw_twist_ulong",     "dtw_twist_ushort",
}


def _fmt_cell(col: str, val) -> str:
    if val is None:
        return ""
    if isinstance(val, bool):
        return "Y" if val else ""
    if isinstance(val, str):
        return val
    if isinstance(val, float) and not np.isfinite(val):
        return "nan"
    fmt = NUMERIC_COLS.get(col)
    if fmt is not None:
        return fmt(val)
    return str(val)


def _write_csv(rows: list[dict], out: Path) -> None:
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for row in rows:
            w.writerow([_fmt_cell(c, row.get(c, "")) for c in COLUMNS])


def _write_md(rows: list[dict], out: Path) -> None:
    cells = [[_fmt_cell(c, row.get(c, "")) for c in COLUMNS] for row in rows]
    widths = [len(h) for h in COLUMNS]
    for row in cells:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _pad(cell: str, width: int, right: bool) -> str:
        return cell.rjust(width) if right else cell.ljust(width)

    header = "| " + " | ".join(
        f"**{_pad(h, widths[i], False)}**" for i, h in enumerate(COLUMNS)
    ) + " |"
    # Separator row: use ---: for right-aligned numeric columns
    sep_cells = []
    for i, h in enumerate(COLUMNS):
        w = widths[i] + 4 if i == 0 else widths[i]  # header has '**..**' padding
        w = max(widths[i], 3)
        if h in RIGHT_ALIGN_COLS:
            sep_cells.append("-" * (w - 1) + ":")
        else:
            sep_cells.append("-" * w)
    sep = "| " + " | ".join(sep_cells) + " |"

    lines = [header, sep]
    last_subj = None
    for row_dict, row in zip(rows, cells):
        # Blank separator between subjects and before cohort summary block
        if row_dict["subject_id"] == "cohort" and last_subj != "cohort":
            lines.append("| " + " | ".join(
                _pad("", widths[i], COLUMNS[i] in RIGHT_ALIGN_COLS)
                for i in range(len(COLUMNS))
            ) + " |")
        last_subj = row_dict["subject_id"]
        cells_fmt = []
        for i, c in enumerate(row):
            cells_fmt.append(_pad(c, widths[i],
                                  COLUMNS[i] in RIGHT_ALIGN_COLS))
        lines.append("| " + " | ".join(cells_fmt) + " |")

    out.write_text("\n".join(lines) + "\n")


# ─── Main ──────────────────────────────────────────────────────────────────
def main() -> None:
    dom_map = _load_dominant_map()
    rows: list[dict] = []
    missing: list[str] = []

    for sid in ALL_SUBJECTS:
        sub = _load_subject(sid)
        if sub is None:
            missing.append(sid)
            continue
        dom_r = dom_map.get(sid, -1)
        for region in REGIONS:
            rows.append(_subject_region_row(sub, region, dom_r))

    for cf in ("all", "normal", "abnormal"):
        for region in (1, 2, 3, 4, 5):
            rows.append(_cohort_summary_row(region, rows, cohort_filter=cf))

    csv_path = OUT_DIR / "cohort_master_table.csv"
    md_path  = OUT_DIR / "cohort_master_table.md"
    _write_csv(rows, csv_path)
    _write_md(rows, md_path)

    print(f"Wrote {csv_path}  ({len(rows)} rows)")
    print(f"Wrote {md_path}")
    if missing:
        print(f"WARNING: missing subjects skipped: {missing}")

    # Markdown preview (first 12 lines so header + sep + blank + 5×R0..R4)
    print("\nMarkdown preview (first 12 lines):")
    print("-" * 72)
    for i, line in enumerate(md_path.read_text().splitlines()[:12]):
        print(line)
    print("-" * 72)


if __name__ == "__main__":
    main()
