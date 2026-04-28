"""make_dataset_tables.py — extract subject-level acquisition metadata and
objective image-quality metrics for Table 1 (dataset characteristics).

Reads only — does NOT recompute any pipeline quantity.  Missing DICOM tags
(most subjects have only the converted us4d.npz) are recorded as "N/A" and
reported at the end so we can decide whether to chase down the originals.

Outputs:
  figures_paper/dataset_table1.csv          machine-readable, full columns
  figures_paper/dataset_table1.md           IEEE single-column main table
  figures_paper/dataset_supplementary.md    extended metadata
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pydicom

import subject_metadata as smeta

DATA_ROOT    = Path("/data/xinge/Cardiac-Twist-Analysis")
OUT_DIR      = DATA_ROOT / "figures_paper"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_SUBJECTS = ["002A", "03", "p009npa", "p009pa", "p066a",
                "p020_1a", "p030_lp2a"]
FALLBACK_FRAME_RATE = 15.0
SUBJECT_META = smeta.SUBJECT_META


# ─── DICOM access ──────────────────────────────────────────────────────────
def _find_dicom(sid: str) -> Path | None:
    root = DATA_ROOT / sid
    candidates: list[Path] = []
    for sub in ("", "dcm", "raw"):
        d = root / sub if sub else root
        if d.is_dir():
            candidates.extend(d.glob("*.dcm"))
            candidates.extend(d.glob("*.DCM"))
    if not candidates:
        return None
    # Prefer the largest file (most likely the 3D+t volume)
    return max(candidates, key=lambda p: p.stat().st_size)


def _dcm_tag(ds: "pydicom.Dataset | None", group: int, elem: int) -> str | None:
    if ds is None:
        return None
    t = pydicom.tag.Tag(group, elem)
    if t not in ds:
        return None
    v = ds[t].value
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


# ─── Per-subject metadata ──────────────────────────────────────────────────
def _probe_from_image_comments(ds: "pydicom.Dataset | None") -> str | None:
    """Philips QLAB exports sometimes stash probe info in Image Comments."""
    if ds is None:
        return None
    c = _dcm_tag(ds, 0x0020, 0x4000)  # Image Comments
    if c and "FROM " in c:
        # e.g. "3D SPH (FROM IE33) WRITTEN AS 3DDCMXXX"
        start = c.find("(FROM ")
        end   = c.find(")", start)
        if 0 <= start < end:
            return c[start + 1:end].strip()
    return None


def _trajectory_step_cov(traj: np.ndarray) -> float:
    """Per-seed coefficient of variation of per-frame step length, cohort-
    averaged.  `traj` is (nt, N, 3) in voxel units."""
    steps = np.linalg.norm(np.diff(traj, axis=0), axis=2)        # (nt-1, N)
    m = steps.mean(axis=0)
    s = steps.std(axis=0)
    cov = np.full_like(m, np.nan, dtype=float)
    good = m > 1e-9
    cov[good] = s[good] / m[good]
    return float(np.nanmean(cov)) if np.any(good) else float("nan")


def collect_subject(sid: str) -> tuple[dict, list[str]]:
    missing: list[str] = []
    row: dict = {"subject_id": sid}

    # ── us4d volume (always present) ───────────────────────────────────────
    us = np.load(DATA_ROOT / sid / "us4d.npz")
    vol = us["data"]                                 # (H, W, D, T)
    scale = np.asarray(us["scale"]).astype(float)    # (sx, sy, sz) mm
    npz_fr = float(us["frame_rate"]) if "frame_rate" in us.files else None

    row["voxel_spacing_mm"] = (
        f"{scale[0]:.2f} × {scale[1]:.2f} × {scale[2]:.2f}"
    )
    row["volume_dimensions"] = (
        f"{vol.shape[0]}×{vol.shape[1]}×{vol.shape[2]}"
    )

    # ── pipeline NPZs ──────────────────────────────────────────────────────
    an = np.load(DATA_ROOT / sid / "result" / "strain_twist_analysis.npz",
                 allow_pickle=True)
    tw = np.load(DATA_ROOT / sid / "result" / "twist_3d.npz")
    dt = np.load(DATA_ROOT / sid / "result" / "dtw_analysis.npz")

    nt = int(an["twist_regions"].shape[0])
    row["frames_captured"]  = nt
    row["short_cycle_flag"] = bool(nt < 20)

    row["total_seeds"]       = int(tw["seeds"].shape[0])
    row["z_bands"]           = int(len(np.unique(tw["region_ids"])))
    per_region = np.bincount(tw["region_ids"].astype(int))
    row["seeds_per_region"]  = int(per_region.max()) if per_region.size else 0

    d_long = np.asarray(an["d_long"]).astype(float)
    row["long_axis_tilt_deg"] = float(
        np.degrees(np.arccos(np.clip(d_long[2], -1, 1)))
    )
    row["rotation_sign_flipped"] = bool(dt["sign_flipped"])

    # ── DICOM acquisition (optional) ───────────────────────────────────────
    dcm_path = _find_dicom(sid)
    ds = pydicom.dcmread(dcm_path, stop_before_pixels=True) if dcm_path else None

    row["scanner_manufacturer"] = _dcm_tag(ds, 0x0008, 0x0070) or "N/A"
    row["scanner_model"]        = _dcm_tag(ds, 0x0008, 0x1090) or "N/A"

    probe = (_dcm_tag(ds, 0x0018, 0x1030)
             or _dcm_tag(ds, 0x0018, 0x5010)
             or _dcm_tag(ds, 0x0018, 0x6031)
             or _probe_from_image_comments(ds)
             or "N/A")
    row["probe_transducer"]     = probe

    row["patient_age"]          = _dcm_tag(ds, 0x0010, 0x1010) or "N/A"
    row["patient_sex"]          = _dcm_tag(ds, 0x0010, 0x0040) or "N/A"

    # Frame rate: DICOM CineRate → FrameTime → us4d fallback
    fr_src = None
    fr_val: float | None = None
    cine_rate = _dcm_tag(ds, 0x0018, 0x0040)
    if cine_rate:
        try:
            fr_val = float(cine_rate); fr_src = "DICOM CineRate"
        except ValueError:
            pass
    if fr_val is None:
        ft = _dcm_tag(ds, 0x0018, 0x1063)
        if ft:
            try:
                ft_f = float(ft)
                if ft_f > 0:
                    fr_val = 1000.0 / ft_f
                    fr_src = "DICOM FrameTime"
            except ValueError:
                pass
    # If the DICOM is anonymized but xlsx supplied a clinician-provided rate
    # for this subject, prefer that over the 15 Hz fallback.
    xlsx_fr = SUBJECT_META.get(sid, {}).get("frame_rate_hz")
    if fr_val is None and xlsx_fr is not None:
        fr_val = float(xlsx_fr)
        fr_src = "xlsx (clinician metadata)"
    if fr_val is None:
        fr_val = FALLBACK_FRAME_RATE
        fr_src = "fallback (pipeline default 15 Hz)"

    row["frame_rate_hz"]        = float(fr_val)
    row["frame_rate_source"]    = fr_src
    row["duration_s"]           = float(nt) / float(fr_val)

    # Cardiac cycles (needs HR)
    hr_s = _dcm_tag(ds, 0x0018, 0x1088)
    if hr_s:
        try:
            hr = float(hr_s)
            if hr > 0:
                row["heart_rate_bpm"] = hr
                row["cardiac_cycles"] = row["duration_s"] * hr / 60.0
            else:
                row["heart_rate_bpm"], row["cardiac_cycles"] = "N/A", "N/A"
        except ValueError:
            row["heart_rate_bpm"], row["cardiac_cycles"] = "N/A", "N/A"
    else:
        row["heart_rate_bpm"], row["cardiac_cycles"] = "N/A", "N/A"

    # ── Objective image-quality metrics ────────────────────────────────────
    vol_f = vol.astype(np.float64)
    mean_I = float(vol_f.mean())
    std_I  = float(vol_f.std())
    row["mean_bmode_intensity"] = mean_I
    row["intensity_cov"]        = float(std_I / mean_I) if mean_I > 0 else float("nan")

    traj = tw["trajectories"].astype(np.float64)
    row["temporal_snr_seed_cov"] = _trajectory_step_cov(traj)

    # Tracking confidence from NCC peaks + valid mask (available in twist_3d)
    ncc = np.asarray(tw["ncc_peaks"]).astype(np.float64)
    valid = np.asarray(tw["valid"]).astype(bool)
    if valid.any():
        row["tracking_confidence_ncc"] = float(np.nanmean(ncc[valid]))
    else:
        row["tracking_confidence_ncc"] = float("nan")
    row["tracking_valid_fraction"] = float(valid.mean())

    # ── Missing-field bookkeeping ──────────────────────────────────────────
    for k in ("scanner_manufacturer", "scanner_model", "probe_transducer",
              "patient_age", "patient_sex", "heart_rate_bpm",
              "cardiac_cycles"):
        if row.get(k) == "N/A":
            missing.append(k)
    if fr_src.startswith("fallback"):
        missing.append("frame_rate_hz (used fallback 15 Hz)")

    # Diagnosis (clinician label from xlsx); QC tags moved to supplementary.
    row["diagnosis"] = SUBJECT_META.get(sid, {}).get("diagnosis", "—")
    qc_tags = []
    if row["short_cycle_flag"]:
        qc_tags.append("short cycle")
    if row["rotation_sign_flipped"]:
        qc_tags.append("sign flipped")
    if fr_src.startswith("fallback"):
        qc_tags.append("frame rate unrecoverable (anonymized DICOM)"
                       if dcm_path else "frame rate unrecoverable (no DICOM)")
    row["qc_notes"] = "; ".join(qc_tags) if qc_tags else "—"

    return row, missing


# ─── Output formatting ─────────────────────────────────────────────────────
def _fmt_numeric(v, fmt: str) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, str):
        return v
    if isinstance(v, float) and not np.isfinite(v):
        return "N/A"
    try:
        return fmt.format(v)
    except (ValueError, TypeError):
        return str(v)


CSV_COLUMNS = [
    "subject_id",
    "diagnosis",
    "scanner_manufacturer", "scanner_model", "probe_transducer",
    "voxel_spacing_mm", "volume_dimensions",
    "frame_rate_hz", "frame_rate_source",
    "frames_captured", "duration_s",
    "heart_rate_bpm", "cardiac_cycles",
    "patient_age", "patient_sex",
    "total_seeds", "z_bands", "seeds_per_region",
    "long_axis_tilt_deg", "rotation_sign_flipped",
    "mean_bmode_intensity", "intensity_cov",
    "temporal_snr_seed_cov",
    "tracking_confidence_ncc", "tracking_valid_fraction",
    "short_cycle_flag", "qc_notes",
]

CSV_FMT = {
    "frame_rate_hz":          "{:.2f}",
    "duration_s":             "{:.2f}",
    "cardiac_cycles":         "{:.2f}",
    "long_axis_tilt_deg":     "{:.2f}",
    "mean_bmode_intensity":   "{:.2f}",
    "intensity_cov":          "{:.3f}",
    "temporal_snr_seed_cov":  "{:.3f}",
    "tracking_confidence_ncc":"{:.3f}",
    "tracking_valid_fraction":"{:.3f}",
}


def write_csv(rows: list[dict], out: Path) -> None:
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for row in rows:
            cells = []
            for col in CSV_COLUMNS:
                v = row.get(col, "N/A")
                if isinstance(v, bool):
                    cells.append("Y" if v else "N")
                elif col in CSV_FMT:
                    cells.append(_fmt_numeric(v, CSV_FMT[col]))
                else:
                    cells.append(str(v) if v is not None else "N/A")
            w.writerow(cells)


# Markdown helpers
def _pad_row(cells: list[str], widths: list[int],
             right_cols: set[int]) -> str:
    pieces = []
    for i, c in enumerate(cells):
        pieces.append(c.rjust(widths[i]) if i in right_cols else
                      c.ljust(widths[i]))
    return "| " + " | ".join(pieces) + " |"


def _sep_row(widths: list[int], right_cols: set[int]) -> str:
    pieces = []
    for i, w in enumerate(widths):
        w2 = max(w, 3)
        pieces.append(("-" * (w2 - 1) + ":") if i in right_cols
                      else "-" * w2)
    return "| " + " | ".join(pieces) + " |"


def _md_table(header: list[str], body: list[list[str]],
              right_cols: set[int]) -> str:
    widths = [len(h) for h in header]
    for row in body:
        for i, c in enumerate(row):
            widths[i] = max(widths[i], len(c))
    lines = [
        _pad_row([f"**{h}**" for h in header],
                 [w + 4 for w in widths], right_cols),
        _sep_row(widths, right_cols),
    ]
    for row in body:
        lines.append(_pad_row(row, widths, right_cols))
    return "\n".join(lines) + "\n"


def write_main_md(rows: list[dict], out: Path) -> None:
    header = ["Subject", "Frames (n_t)", "Duration (s)", "Frame rate (Hz)",
              "Voxel (mm)", "Cardiac cycles", "Long-axis tilt (°)",
              "Mean B-mode I", "Diagnosis"]
    right = {1, 2, 3, 5, 6, 7}
    body: list[list[str]] = []
    for r in rows:
        body.append([
            r["subject_id"],
            str(r["frames_captured"]),
            _fmt_numeric(r["duration_s"], "{:.2f}"),
            _fmt_numeric(r["frame_rate_hz"], "{:.2f}"),
            r["voxel_spacing_mm"],
            _fmt_numeric(r["cardiac_cycles"], "{:.2f}"),
            _fmt_numeric(r["long_axis_tilt_deg"], "{:.1f}"),
            _fmt_numeric(r["mean_bmode_intensity"], "{:.1f}"),
            r["diagnosis"],
        ])
    out.write_text(_md_table(header, body, right))


def write_supp_md(rows: list[dict], out: Path) -> None:
    header = ["Subject", "Diagnosis", "Scanner manufacturer", "Model",
              "Probe", "Volume (vox)", "Age", "Sex",
              "Seeds", "Z-bands", "Seeds / region",
              "Sign flipped", "Intensity CoV",
              "Temporal seed CoV", "NCC mean", "Valid fraction",
              "Frame-rate source", "QC notes"]
    right = {5, 8, 9, 10, 12, 13, 14, 15}
    body: list[list[str]] = []
    for r in rows:
        body.append([
            r["subject_id"],
            r["diagnosis"],
            str(r["scanner_manufacturer"]),
            str(r["scanner_model"]),
            str(r["probe_transducer"]),
            r["volume_dimensions"],
            str(r["patient_age"]),
            str(r["patient_sex"]),
            str(r["total_seeds"]),
            str(r["z_bands"]),
            str(r["seeds_per_region"]),
            "Y" if r["rotation_sign_flipped"] else "N",
            _fmt_numeric(r["intensity_cov"], "{:.3f}"),
            _fmt_numeric(r["temporal_snr_seed_cov"], "{:.3f}"),
            _fmt_numeric(r["tracking_confidence_ncc"], "{:.3f}"),
            _fmt_numeric(r["tracking_valid_fraction"], "{:.3f}"),
            str(r["frame_rate_source"]),
            r["qc_notes"],
        ])
    out.write_text(_md_table(header, body, right))


# ─── Main ──────────────────────────────────────────────────────────────────
def main() -> None:
    rows: list[dict] = []
    missing_by_subject: dict[str, list[str]] = {}
    for sid in ALL_SUBJECTS:
        row, missing = collect_subject(sid)
        rows.append(row)
        missing_by_subject[sid] = missing

    csv_path  = OUT_DIR / "dataset_table1.csv"
    main_md   = OUT_DIR / "dataset_table1.md"
    supp_md   = OUT_DIR / "dataset_supplementary.md"

    write_csv(rows, csv_path)
    write_main_md(rows, main_md)
    write_supp_md(rows, supp_md)

    print(f"Wrote {csv_path}  ({len(rows)} rows)")
    print(f"Wrote {main_md}")
    print(f"Wrote {supp_md}")

    print("\nMissing fields summary:")
    for sid in ALL_SUBJECTS:
        miss = missing_by_subject[sid]
        if miss:
            print(f"  {sid}: {miss}")
        else:
            print(f"  {sid}: (none)")

    print("\ndataset_table1.md preview (first 2 rows):")
    print("-" * 72)
    for line in main_md.read_text().splitlines()[:4]:
        print(line)
    print("-" * 72)


if __name__ == "__main__":
    main()
