# Region-Level 3D Cardiac Twist Analysis Pipeline

**Xinge Guo · Jacklyn Xu**  
Department of Biomedical Engineering, Duke University  
April 2026

---

## Table of Contents

1. [Overview](#overview)
2. [Background](#background)
3. [Repository Structure](#repository-structure)
4. [Dependencies](#dependencies)
5. [Dataset](#dataset)
6. [Pipeline Stages](#pipeline-stages)
   - [Stage 1: DICOM Parsing](#stage-1-dicom-parsing)
   - [Stage 2: LV Segmentation](#stage-2-lv-segmentation)
   - [Stage 3: Seed Initialisation](#stage-3-seed-initialisation)
   - [Stage 4: Anatomical Coordinate Frame](#stage-4-anatomical-coordinate-frame)
   - [Stage 5: 3D NCC Speckle Tracking](#stage-5-3d-ncc-speckle-tracking)
   - [Stage 6: Rotation and Twist Computation](#stage-6-rotation-and-twist-computation)
   - [Stage 7: Waveform Metrics](#stage-7-waveform-metrics)
7. [Key Methodological Decisions](#key-methodological-decisions)
8. [Output Files](#output-files)
9. [Running the Pipeline](#running-the-pipeline)
10. [Results Summary](#results-summary)
11. [Limitations](#limitations)
12. [Future Work](#future-work)
13. [Citation](#citation)
14. [References](#references)

---

## Overview

This repository provides an open-source Python pipeline for **region-level three-dimensional cardiac twist analysis** from Philips four-dimensional echocardiographic DICOM data, without reliance on proprietary software.

The pipeline parses Philips-specific private DICOM tags to reconstruct the 4D volume, initialises 36 myocardial seed points across six axial bands, tracks all seeds throughout the cardiac cycle via 3D normalised cross-correlation (NCC), applies rigid-body rotation (RBR) correction to derive per-region twist-time curves, and extracts three waveform-level metrics that characterise inter-regional dyssynchrony, wringing coherence, and amplitude-weighted base-apex coupling.

**Key contributions:**
- Open-source parsing of Philips 4D echocardiographic DICOM private tags
- Region-level 3D NCC speckle tracking with direct (non-sequential) template matching
- RBR correction for robust twist estimation robust to single-band tracking failures
- Three novel waveform-level metrics beyond commercial peak-only outputs:
  - `σ_tpeak` — inter-regional timing dyssynchrony
  - `ρ_sys(R0, R5)` — systolic apex-base wringing coherence
  - `W_R0R5` — amplitude-weighted apex-base coupling
- Linear-gradient fitted twist waveform summarising whole-LV torsion dynamics

---

## Background

The left ventricle contracts via a **wringing motion** driven by opposing helical muscle fibre layers (subepicardium: −60°, subendocardium: +60°). This produces counter-rotation between base and apex during systole, and rapid untwisting in early diastole that generates ventricular suction for diastolic filling.

Commercial 3D speckle tracking platforms (GE EchoPAC, Philips QLAB) extract only two scalar summaries from the twist-time waveform: **peak twist angle** and **time-to-peak**. The complete waveform — including shape, inter-regional phase relationships, and base-apex coupling dynamics — is displayed but not quantified.

This pipeline is designed to fill that gap by providing complete waveform-level characterisation of LV twist across six axial regions, enabling mechanistic discrimination between pathologies that share similar peak twist values but differ in their underlying failure mode.

---

## Repository Structure

```
cardiac_twist_pipeline/
│
├── preprocess_4d.py          # Stage 1: Philips DICOM parsing and volume reconstruction
├── segment_3d.py             # Stage 2: NRRD mask loading, per-slice radial bounds
├── select_seeds_3d.py        # Stage 3: Seed initialisation (brightness scoring + NMS)
├── ncc_3d.py                 # Stage 5: 3D NCC speckle tracking (FFT-accelerated)
├── track_3d.py               # Stage 5: Direct tracking with validity filtering
├── twist_3d.py               # Stage 4+6: Long-axis definition + rotation computation
├── pipeline.py               # Stage 6: RBR correction, twist curves, linear gradient fit
├── make_region_metrics_table.py  # Stage 7: σ_tpeak, ρ_sys, W_R0R5
├── make_master_table.py      # Cohort-level summary table generation
├── subject_metadata.py       # Frame rates, diagnoses, diagnosis_class per subject
├── fix_strain_final.py       # Displacement decomposition module (experimental)
├── publication_figures_v2.py # Figure generation for paper
│
├── data/
│   └── {subject_id}/
│       ├── {subject_id}.dcm      # Raw Philips 4D DICOM file
│       └── {subject_id}_mask.nrrd # 3D Slicer segmentation mask (frame 0)
│
├── results/
│   └── {subject_id}/
│       ├── trajectories.npz      # Seed trajectories across all frames
│       ├── twist_curves.npz      # Per-region twist-time curves (RBR-corrected)
│       ├── region_metrics.csv    # σ_tpeak, ρ_sys, W_R0R5
│       └── figures/              # Per-subject twist curve plots
│
└── README.md
```

---

## Dependencies

```bash
pip install numpy scipy matplotlib pydicom SimpleITK nibabel scikit-image
```

| Package | Version tested | Purpose |
|---------|---------------|---------|
| numpy | ≥ 1.24 | Array operations |
| scipy | ≥ 1.10 | FFT convolution, signal smoothing, Pearson correlation |
| matplotlib | ≥ 3.7 | Figure generation |
| pydicom | ≥ 2.4 | DICOM parsing |
| SimpleITK | ≥ 2.2 | NRRD mask loading |
| nibabel | ≥ 5.0 | NIfTI export for 3D Slicer |
| scikit-image | ≥ 0.21 | Image preprocessing utilities |

Python ≥ 3.10 required.

---

## Dataset

The pipeline was developed and tested on a pilot cohort of **N = 7 subjects** acquired at Duke University Medical Center using a Philips 4D echocardiographic system.

| Subject | Frames | Duration (s) | Frame rate (Hz) | Voxel spacing (mm) | LV tilt (°) | Diagnosis |
|---------|--------|-------------|----------------|-------------------|-------------|-----------|
| 002A | 25 | 0.72 | 34.77 | 0.84 × 0.84 × 0.63 | 8.1 | Normal |
| 03 | 17 | 0.60 | 28.30 | 1.09 × 1.09 × 0.82 | 22.7 | Normal |
| p066a | 21 | 0.70 | 30.00 | 0.84 × 0.84 × 0.63 | 17.6 | Normal |
| p009npa | 23 | 1.28 | 18.00 | 0.84 × 0.84 × 0.63 | 17.4 | LBBB, pre-pacer |
| p009pa | 14 | 0.82 | 17.00 | 0.84 × 0.84 × 0.63 | 2.7 | LBBB, paced |
| p020_1a | 28 | 0.82 | 34.00 | 0.84 × 0.84 × 0.63 | 27.5 | MI, akinetic distal walls |
| p030_lp2a | 27 | 0.87 | 31.00 | 0.84 × 0.84 × 0.63 | 15.2 | LV hypertrophy + amyloidosis |

**Note:** Frame rate was absent in the DICOM header for 3 subjects due to anonymisation; 15 Hz was assumed for those cases. This may affect temporal metrics (σ_tpeak, t_pk).

---

## Pipeline Stages

### Stage 1: DICOM Parsing

**File:** `preprocess_4d.py`

Philips 4D echocardiographic DICOM does not conform to the standard Enhanced US Volume SOP class. The pixel buffer is stored as a flat `(nt × nz, ny, nx)` array with spatial metadata encoded in private tags.

**Tags recovered:**

| Tag | Description |
|-----|-------------|
| `(3001, 1001)` | Number of z-slices (`nz`) — Philips private |
| `(3001, 1003)` | z-spacing in cm (`dz`) — Philips private |
| `(0018, 602C)` | Physical delta X in cm — standard |
| `(0018, 602E)` | Physical delta Y in cm — standard |

**Processing steps:**
1. Read raw `uint8` pixel buffer from DICOM
2. Reshape from `(nt × nz, ny, nx)` → `(nt, nz, ny, nx)` → transpose to `(nx, ny, nz, nt)`
3. Convert all spacings from cm → mm
4. Apply subject-specific overrides where `Rows`/`Columns` are absent in anonymised header
5. Export frame-0 volume as NIfTI (`.nii.gz`) for manual segmentation in 3D Slicer

---

### Stage 2: LV Segmentation

**File:** `segment_3d.py`

Manual LV myocardial segmentation was performed on the frame-0 volume using **3D Slicer v5.10.0**, exported as a binary NRRD mask.

For each z-slice:
- Identify all mask voxels
- Compute slice centroid `(cx, cy)`
- Compute distance from each mask voxel to centroid
- `r_inner(z)` = 10th percentile distance (endocardial boundary)
- `r_outer(z)` = 90th percentile distance (epicardial boundary)
- Empty slices are filled by linear interpolation from neighbouring slices

---

### Stage 3: Seed Initialisation

**File:** `select_seeds_3d.py`

Seeds are the speckle tracking anchor points placed within the LV myocardium.

**Layout:** 6 axial z-bands (R0 = most basal, R5 = most apical) × 6 angular sectors = **36 seeds total**.

Z-band boundaries are computed as:
```python
z_edges = np.linspace(z_lo, z_hi + 1, n_z_levels + 1)
```
where `z_lo` and `z_hi` are the minimum and maximum z-coordinates of valid candidate voxels from the mask.

**Seed selection per sector:**
1. Divide the myocardial annulus into 6 equal angular sectors
2. Score all candidate voxels by brightness: `I(s, 0)`
3. Select the brightest voxel in each sector
4. Apply greedy NMS: reject candidates within 12 mm of an already-selected seed
5. Exclude seeds with radial distance `R_s(0) < 5 mm` from the long axis (numerically unstable for displacement computation)

**Parameters:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| n_z_levels | 6 | Six axial bands R0–R5 |
| n_sectors | 6 | Six angular sectors per band |
| NMS min spacing | 12 mm | > NCC template footprint (~7.6 mm) |
| Min radial distance | 5 mm | Numerical stability for d_short |

---

### Stage 4: Anatomical Coordinate Frame

**File:** `twist_3d.py`

The LV long-axis direction `d_long` is required for transverse plane projection (rotation computation) and for displacement decomposition.

**Adopted method — Seed Geometry:**

```
c_base = mean position of 5 most-basal seeds at frame 0
c_apex = mean position of 5 most-apical seeds at frame 0
d_long = (c_apex - c_base) / ||c_apex - c_base||
```

**Why not PCA?**

Principal Component Analysis on the mask centroid curve `{(cx(z), cy(z), z)}` was the original implementation. PCA fails when LV tilt < 5° because the centroid curve has near-zero variance in x and y — the first principal component collapses into the imaging xy-plane rather than aligning with the anatomical long axis.

This failure was directly observed in subject **p009pa** (tilt = 2.7°). Seed geometry is robust at all orientations and was adopted for all 7 subjects.

**Short-axis direction (per seed):**
```
v_s = p_s(0) - [(p_s(0) - c_base) · d_long] * d_long
d_short,s = v_s / ||v_s||      (valid only if ||v_s|| >= 5 mm)
```

---

### Stage 5: 3D NCC Speckle Tracking

**File:** `ncc_3d.py`, `track_3d.py`

Seed positions are tracked across the cardiac cycle using **direct 3D normalised cross-correlation**.

**Strategy:** The template is fixed at frame 0 throughout the entire sequence. This avoids cumulative drift from sequential template updating at the cost of reduced robustness if the frame-0 speckle pattern differs substantially from later frames (e.g., due to large deformation or low frame rate).

**NCC computation:**
```
NCC(τ, t) = Σ_u T̃(u) · P̃(u+τ, t) / sqrt(Σ_u T̃²(u) · Σ_u P̃²(u+τ,t))
```
where `T̃` and `P̃` are zero-mean, unit-norm versions of the template and search patch respectively. Computed via FFT-accelerated convolution using `scipy.signal.fftconvolve`.

**Parameters:**

| Parameter | Value | Physical size |
|-----------|-------|---------------|
| Template size | 9 × 9 × 5 voxels | ≈ 7.6 × 7.6 × 8.2 mm (approx. isotropic) |
| Search volume | 31 × 31 × 17 voxels | ±12.6 × 12.6 × 13.2 mm |

**Validity filter (both conditions required):**
- `NCC*(s, t) ≥ 0.25`
- `r_inner(z) - 5 ≤ R_s(t) ≤ r_outer(z) + 5` voxels

Seeds failing either condition are marked invalid at that frame and excluded from regional rotation computation.

---

### Stage 6: Rotation and Twist Computation

**File:** `pipeline.py`

**Instantaneous rotation angle per seed:**
```
θ_s(t) = atan2(y_s(t) - y_c(t),  x_s(t) - x_c(t))
```
where `(x_c(t), y_c(t))` is the projection of the LV centroid onto the transverse plane at the seed's current z-level and frame t. Using a **time-varying centroid** removes the translational component introduced by LV longitudinal shortening during systole.

**Regional rotation:**
```
rot_r(t) = mean of θ_s(t) over all valid seeds s in band r at frame t
```

**RBR correction (adopted):**
```
<rot>(t) = (1/6) * Σ_{r=0}^{5} rot_r(t)
twist_r(t) = rot_r(t) - <rot>(t)
```

This subtracts the six-band mean rotation at each frame, removing the whole-heart rigid-body rotational component. Single-band tracking noise is diluted across all six bands rather than propagated. R0 is non-zero under this definition — it reflects the basal band's deviation from the mean whole-heart rotation. This formulation is consistent with the Sengupta 2008 torsion definition.

**Why not R0 subtraction?**

The original implementation used `twist_r(t) = rot_r(t) - rot_0(t)`, making `twist_R0(t) = 0` by construction. This was replaced because:
1. Any tracking failure in R0 propagates directly to all five other regions
2. Rigid-body rotation of the whole heart is not removed

**Post-processing:**
- Savitzky-Golay smoothing (window = 11, polynomial order = 3) applied to each twist-time curve
- Automatic sign-flip detection and correction for transducer orientation differences between subjects

**Linear-gradient fitted twist:**
```
α(t) = polyfit(z/L, {twist_r(t)}_{r=0}^{5}, degree=1)
Θ_fit(t) = α(t) · L
```
where `L` is the LV long-axis length in mm. This produces a single waveform summarising whole-LV torsion dynamics by fitting a linear spatial gradient to the six regional values at each frame.

---

### Stage 7: Waveform Metrics

**File:** `make_region_metrics_table.py`

Three scalar metrics are extracted from the per-region twist-time curves to quantify distinct mechanical failure modes.

#### σ_tpeak — Timing Failure

Standard deviation of regional peak times across 6 bands:
```
t_pk^r = argmax_t twist_r(t)
σ_tpeak = std({t_pk^r}_{r=0}^{5}) × 1000   [ms]
```
Large σ_tpeak indicates inter-regional timing dyssynchrony. **Normal range: [100, 200] ms.**

#### ρ_sys(R0, R5) — Directional Failure

Pearson correlation between R0 and R5 twist curves in systole:
```
ρ_sys = corr(twist_R0(t), twist_R5(t))   for t ∈ [0, t_avc]
```
where `t_avc` is estimated as the peak time of the net twist between apical-third and basal-third band averages. In normal wringing, base and apex deviate in opposite directions from the mean, so ρ_sys should be strongly negative. **Normal threshold: ρ_sys < −0.5.**

#### W_R0R5 — Amplitude Failure

Covariance of raw (non-RBR-corrected) rotation signals of R0 and R5 over the full cycle:
```
W_R0R5 = cov(rot_R0(t), rot_R5(t))   for t ∈ [0, T]
```
Unlike ρ_sys, W_R0R5 is amplitude-weighted and not normalised — a strongly negative value requires both large counter-rotating amplitudes and consistent anti-phase behaviour. **Normal threshold: W_R0R5 < −2 °².**

---

## Key Methodological Decisions

| Decision | Original | Adopted | Reason |
|----------|---------|---------|--------|
| Long-axis definition | PCA on mask centroid curve | Seed geometry (apex-base centroid vector) | PCA collapses into xy-plane at LV tilt < 5° (observed in p009pa, tilt = 2.7°) |
| Twist reference | R0 subtraction | RBR correction (6-band mean) | R0 noise propagates to all regions; rigid-body rotation not removed |
| Template update | Sequential (frame-to-frame) | Direct (frame-0 fixed) | Prevents cumulative drift from sequential tracking errors |
| Displacement | Not implemented | Implemented, not validated | NaN values in u_short due to seeds with R_s < 5 mm; excluded from main analysis |

---

## Output Files

For each subject, the pipeline produces:

| File | Description |
|------|-------------|
| `trajectories.npz` | Seed positions `p_s(t)` for all seeds and frames; validity flags |
| `twist_curves.npz` | `rot_r(t)`, `twist_r(t)` (RBR-corrected), `theta_fit(t)` per region |
| `region_metrics.csv` | σ_tpeak, ρ_sys, W_R0R5, dominant region, peak twist per band |
| `qc_metrics.csv` | Intensity CoV, temporal seed CoV, NCC mean per subject |
| `fig_twist_curves.png` | Per-region twist-time curves (6 subplots + overlay) |
| `fig_linear_gradient.png` | Linear-gradient fitted twist waveform Θ_fit(t) |
| `fig_trajectories_3d.png` | 3D seed trajectory visualisation |

---

## Running the Pipeline

### 1. Prepare data

Place each subject's DICOM file and 3D Slicer NRRD mask in:
```
data/{subject_id}/{subject_id}.dcm
data/{subject_id}/{subject_id}_mask.nrrd
```

### 2. Parse DICOM and export frame-0 volume

```bash
python preprocess_4d.py --subject 002A
```

This exports `data/002A/frame0.nii.gz` for segmentation.

### 3. Segment LV in 3D Slicer

- Open `frame0.nii.gz` in 3D Slicer v5.10.0
- Use the Segment Editor module to manually delineate LV myocardium
- Export as binary NRRD mask to `data/002A/002A_mask.nrrd`

### 4. Run full pipeline

```bash
python pipeline.py --subject 002A
```

Or run for all subjects:
```bash
for subj in 002A 03 p066a p009npa p009pa p020_1a p030_lp2a; do
    python pipeline.py --subject $subj
done
```

### 5. Generate cohort summary table

```bash
python make_master_table.py
```

---

## Results Summary

### Acquisition Quality and Tracking

| Subject | Intensity CoV | Temporal CoV | NCC mean |
|---------|--------------|--------------|----------|
| 002A | 1.916 | 0.903 | 0.763 |
| 03 | 1.851 | 0.833 | 0.806 |
| p066a | 1.634 | 0.769 | 0.847 |
| p009npa | 1.563 | 0.743 | 0.694 |
| p009pa | 2.625 | 0.622 | 0.709 |
| p020_1a | 1.613 | 0.788 | 0.789 |
| p030_lp2a | 1.936 | 0.992 | 0.821 |
| **Mean ± SD** | **1.877 ± 0.354** | **0.807 ± 0.117** | **0.776 ± 0.061** |

### Three-Metric Results

| Subject | Diagnosis | σ_tpeak (ms) | ρ_sys(R0,R5) | W_R0R5 (°²) | Failure mode |
|---------|-----------|-------------|--------------|-------------|--------------|
| 002A | Normal | 76 | −0.69 | −8.78 | None |
| 03 | Normal | 147 | −0.80 | −2.52 | None |
| p066a | Normal | 130 | −0.30 | +3.89 | Amplitude |
| p009npa | Abnormal | 285 | −0.08 | −7.44 | Timing + Direction |
| p009pa | Abnormal | 168 | +0.12 | −2.82 | Partial CRT recovery |
| p020_1a | Abnormal | 139 | +0.08 | +0.30 | All three |
| p030_lp2a | Abnormal | 73 | −0.25 | +2.80 | Amplitude |

Normal thresholds: σ_tpeak ∈ [100, 200] ms, ρ_sys < −0.5, W_R0R5 < −2 °²

Subject 03 is the only case where all three metrics simultaneously satisfy normal thresholds. Subject p066a (clinically normal) shows amplitude failure on two of three metrics, suggesting subclinical mechanical abnormality undetectable by EF or peak twist alone.

---

## Limitations

- **Small cohort (N = 7):** Results are illustrative and not statistically powered. No generalisable claims can be made about reference ranges.
- **No demographic metadata:** Age, sex, and detailed clinical parameters are unavailable, precluding covariate analysis.
- **Manual segmentation:** Inter-operator variability has not been quantified. Results may differ with different annotators.
- **Frame rate assumption:** Frame rate metadata was absent in 3 subjects due to anonymisation. 15 Hz was assumed; if incorrect, temporal metrics (σ_tpeak, t_pk, AUC) will be systematically scaled.
- **No ground-truth validation:** Twist estimates have not been compared against Philips QLAB, GE EchoPAC, or cardiac MR tagging. Absolute accuracy is unknown; relative patterns and waveform morphology are internally consistent.
- **Displacement decomposition:** Implemented but not validated due to NaN values in short-axis component (seeds with R_s(0) < 5 mm excluded). Not included in main analysis.

---

## Future Work

### Validation
- Bland-Altman agreement analysis against Philips QLAB and GE EchoPAC on peak twist values
- Validation against cardiac MR tagging as ground truth for myocardial motion

### Automated Segmentation
- U-Net architecture for LV myocardial segmentation from 4D echocardiographic volumes has been developed but not validated on this dataset. Once validated, this replaces manual 3D Slicer annotation and enables scaling to larger cohorts.

### Displacement Decomposition
- Implement stricter radial distance constraint at seed initialisation (enforce R_s(0) ≥ 5 mm for all seeds)
- Apply trajectory-level Savitzky-Golay smoothing before displacement projection
- Validate u_long and u_short against reference measurements
- Extract coupling metrics: ρ_long, ρ_short, DTW_long, DTW_short

### Larger and More Diverse Cohort
- Expand to dilated cardiomyopathy, hypertrophic cardiomyopathy, and additional amyloidosis cases available in the Duke dataset
- Establish reference ranges for σ_tpeak, ρ_sys, W_R0R5
- Statistical comparison between normal and pathological groups on full waveform feature set

---

