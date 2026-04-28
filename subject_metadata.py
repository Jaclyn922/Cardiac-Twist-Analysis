"""subject_metadata.py — single source of truth for clinician-supplied
metadata (transcribed from dataset.xlsx).

Used by:
  • make_dataset_tables.py   — Diagnosis column + frame-rate fallback
  • publication_figures_v2.py — subject_frame_rate() helper
  • fix_strain_final.py       — subject_frame_rate() helper
  • make_master_table.py      — diagnosis annotation
  • figure9 / cohort plots    — colour grouping by diagnosis category

`frame_rate_hz` is None when the DICOM is the authoritative source (002A:
real DICOM FrameTime exists; 03: same).  For subjects whose DICOM was
anonymized with zero metadata, the xlsx value below is used instead of
the 15 Hz placeholder.
"""

from __future__ import annotations

SUBJECT_META: dict[str, dict] = {
    "002A": {
        "xlsx_id":          "p002",
        "frame_rate_hz":    None,                  # real DICOM available
        "diagnosis":        "normal",
        "diagnosis_class":  "normal",
    },
    "03": {
        "xlsx_id":          "p003 (PLAX)",
        "frame_rate_hz":    28.0,                  # DICOM → 28.30, xlsx → 28
        "diagnosis":        "normal",
        "diagnosis_class":  "normal",
    },
    "p009npa": {
        "xlsx_id":          "p009 NP",
        "frame_rate_hz":    18.0,
        "diagnosis":        "LBBB / dyssynchrony, pre-pacer",
        "diagnosis_class":  "abnormal",
    },
    "p009pa": {
        "xlsx_id":          "p009 P",
        "frame_rate_hz":    17.0,
        "diagnosis":        "LBBB, paced",
        "diagnosis_class":  "abnormal",
    },
    "p066a": {
        "xlsx_id":          "p066",
        "frame_rate_hz":    30.0,
        "diagnosis":        "normal",
        "diagnosis_class":  "normal",
    },
    "p020_1a": {
        "xlsx_id":          "p020",
        "frame_rate_hz":    34.0,
        "diagnosis":        "MI, akinetic distal walls, hypokinetic proximal",
        "diagnosis_class":  "abnormal",
    },
    "p030_lp2a": {
        "xlsx_id":          "p030",
        "frame_rate_hz":    31.0,
        "diagnosis":        "LV hypertrophy, amyloidosis",
        "diagnosis_class":  "abnormal",
    },
}


def diagnosis_class(sid: str) -> str:
    return SUBJECT_META.get(sid, {}).get("diagnosis_class", "unknown")


def diagnosis_label(sid: str) -> str:
    return SUBJECT_META.get(sid, {}).get("diagnosis", "—")


def xlsx_frame_rate(sid: str) -> float | None:
    """Clinician-supplied frame rate from xlsx, or None if DICOM is
    authoritative (or subject not in metadata)."""
    v = SUBJECT_META.get(sid, {}).get("frame_rate_hz")
    return float(v) if v is not None else None
