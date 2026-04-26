"""
4D Cardiac Ultrasound DICOM Preprocessor
Handles non-standard Philips 3D/4D DICOM (SOP 1.2.840.113543.6.6.1.3.10002)
"""

from __future__ import annotations
import numpy as np
import pydicom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class US4DData:
    """Metadata + 4-D array for one cardiac ultrasound volume sequence."""
    data: np.ndarray
    scale: list[float]
    dim: list[float]
    N: list[int]
    frame_rate: float = 15.0
    BScale: list[int] = field(default_factory=lambda: [0, 255])

    @property
    def nx(self): return self.N[0]
    @property
    def ny(self): return self.N[1]
    @property
    def nz(self): return self.N[2]
    @property
    def nt(self): return self.data.shape[3]


def explore_dicom(path: str | Path) -> None:
    ds = pydicom.dcmread(str(path), force=True)
    print("=" * 60)
    print("DICOM TAG DUMP")
    print("=" * 60)
    dim_keywords = {
        "rows", "columns", "frames", "spacing", "delta",
        "resolution", "dimension", "depth", "slice", "physical",
    }
    print("\n--- Dimension / spacing tags ---")
    for elem in ds:
        name_lower = elem.name.lower()
        if any(k in name_lower for k in dim_keywords) or elem.tag.group == 0x3001:
            val = ("[pixel data]" if elem.tag.group == 0x7FE0 else elem.value)
            print(f"  {elem.tag}  {elem.name}: {val}")

    print("\n--- All other tags ---")
    for elem in ds:
        if elem.tag.group in (0x7FE0, 0x3001):
            continue
        if any(k in elem.name.lower() for k in dim_keywords):
            continue
        try:
            print(f"  {elem.tag}  {elem.name}: {elem.value}")
        except Exception:
            print(f"  {elem.tag}  {elem.name}: [unreadable]")
    print("=" * 60)


# JacklynX changed
def _read_pixel_bytes(path: str | Path, ds) -> bytes:
    import struct
    if hasattr(ds, 'PixelData'):
        return bytes(ds.PixelData)
    with open(str(path), 'rb') as f:
        data = f.read()
    pos = data.find(b'\xe0\x7f\x10\x00')
    if pos == -1:
        raise ValueError(f"Cannot locate pixel data tag in {path}")
    vr = data[pos+4:pos+6]
    if vr in (b'OB', b'OW'):
        pixel_start = pos + 12
        length = struct.unpack('<I', data[pos+8:pos+12])[0]
    else:
        pixel_start = pos + 8
        length = struct.unpack('<I', data[pos+4:pos+8])[0]
    print(f"[load_4d] Raw pixel data found at offset {pixel_start}, {length/1e6:.1f} MB")
    return data[pixel_start:pixel_start + length]


def load_4d(path: str | Path, frame_rate: float = 15.0) -> US4DData:
    ds = pydicom.dcmread(str(path), force=True)

    # JacklynX changed
    DIM_OVERRIDE = {
        "p009npa": (224, 208, 208),
        "p009pa":  (224, 208, 224),
        "p066a":    (192, 160, 208),
        "p020_1a":  (288, 176, 208),
        "p030_lp2a":(288, 176, 208),
    }
    stem = Path(path).stem
    if stem in DIM_OVERRIDE:
        nx, ny, nz = DIM_OVERRIDE[stem]
        print(f"[load_4d] Using known dims for {stem}: nx={nx} ny={ny} nz={nz}")
    else:
        try:
            ny = int(ds.Rows)
            nx = int(ds.Columns)
        except AttributeError:
            nx, ny = 224, 208
            print(f"[load_4d] WARNING: missing Rows/Columns, assuming nx={nx} ny={ny}")
        try:
            nz = int(ds[0x3001, 0x1001].value)
        except KeyError:
            nz = 208
            print(f"[load_4d] WARNING: missing private nz tag, assuming nz={nz}")

    pixel_bytes = _read_pixel_bytes(path, ds)
    total_bytes = len(pixel_bytes)
    nt = total_bytes // (nx * ny * nz)
    assert nx * ny * nz * nt == total_bytes, (
        f"Pixel data size {total_bytes} not divisible by nx*ny*nz = {nx*ny*nz}")

    try:
        dx = float(ds[0x0018, 0x602C].value) * 10.0
        dy = float(ds[0x0018, 0x602E].value) * 10.0
        dz = float(ds[0x3001, 0x1003].value) * 10.0
    except KeyError:
        dx, dy, dz = 0.843, 0.836, 0.632
        print(f"[load_4d] WARNING: missing spacing tags, assuming dx={dx} dy={dy} dz={dz}")

    print(f"[load_4d] Spatial dims  : nx={nx}, ny={ny}, nz={nz}, nt={nt}")
    print(f"[load_4d] Voxel spacing : dx={dx:.4f} mm, dy={dy:.4f} mm, dz={dz:.4f} mm")

    raw = np.frombuffer(pixel_bytes, dtype=np.uint8)
    vol = raw.reshape((nt, nz, ny, nx))
    vol = vol.transpose(3, 2, 1, 0)

    scale = [dx, dy, dz]
    dim   = [nx * dx, ny * dy, nz * dz]
    print(f"[load_4d] Physical size : {dim[0]:.1f} × {dim[1]:.1f} × {dim[2]:.1f} mm³")
    return US4DData(data=vol, scale=scale, dim=dim, N=[nx, ny, nz], frame_rate=frame_rate)


def print_stats(us: US4DData) -> None:
    d = us.data
    nonzero_frac = np.count_nonzero(d) / d.size
    print("\n=== Data statistics ===")
    print(f"  shape      : {d.shape}  (nx, ny, nz, nt)")
    print(f"  dtype      : {d.dtype}")
    print(f"  min / max  : {d.min()} / {d.max()}")
    print(f"  mean±std   : {d.mean():.2f} ± {d.std():.2f}")
    print(f"  non-zero   : {nonzero_frac*100:.1f}%")
    print(f"  scale (mm) : x={us.scale[0]:.4f}  y={us.scale[1]:.4f}  z={us.scale[2]:.4f}")
    print(f"  FOV   (mm) : {us.dim[0]:.1f} × {us.dim[1]:.1f} × {us.dim[2]:.1f}")
    print(f"  frame_rate : {us.frame_rate} vol/s")


def plot_orthogonal_slices(us: US4DData, t: int = 0,
                           out_path: str | Path = "orthogonal_slices.png") -> None:
    vol = us.data[:, :, :, t]
    cx, cy, cz = us.nx // 2, us.ny // 2, us.nz // 2

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    titles = [
        f"YZ plane  (x={cx}, {us.scale[1]:.3f}×{us.scale[2]:.3f} mm)",
        f"XZ plane  (y={cy}, {us.scale[0]:.3f}×{us.scale[2]:.3f} mm)",
        f"XY plane  (z={cz}, {us.scale[0]:.3f}×{us.scale[1]:.3f} mm)",
    ]
    slices = [vol[cx, :, :].T, vol[:, cy, :].T, vol[:, :, cz].T]
    xlabels = ["y (pixel)", "x (pixel)", "x (pixel)"]
    ylabels = ["z (pixel)", "z (pixel)", "y (pixel)"]

    for ax, sl, title, xl, yl in zip(axes, slices, titles, xlabels, ylabels):
        ax.imshow(sl, cmap="gray", origin="lower", interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel(xl); ax.set_ylabel(yl)

    fig.suptitle(f"Orthogonal centre slices  (t={t})", fontsize=11)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"[plot] Orthogonal slices saved → {out_path}")


def save_time_animation(us: US4DData, axis: str = "z",
                        out_path: str | Path = "time_series.gif",
                        fps: float | None = None) -> None:
    fps = fps or us.frame_rate
    cx, cy, cz = us.nx // 2, us.ny // 2, us.nz // 2

    if axis == "x":
        frames = [us.data[cx, :, :, t].T for t in range(us.nt)]
        title_base = f"YZ centre (x={cx})"
    elif axis == "y":
        frames = [us.data[:, cy, :, t].T for t in range(us.nt)]
        title_base = f"XZ centre (y={cy})"
    else:
        frames = [us.data[:, :, cz, t].T for t in range(us.nt)]
        title_base = f"XY centre (z={cz})"

    vmin, vmax = us.BScale
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(frames[0], cmap="gray", vmin=vmin, vmax=vmax,
                   origin="lower", interpolation="nearest")
    ttl = ax.set_title(f"{title_base}  t=0", fontsize=10)
    ax.axis("off")
    plt.tight_layout()

    def _update(i):
        im.set_data(frames[i])
        ttl.set_text(f"{title_base}  t={i}")
        return [im, ttl]

    ani = animation.FuncAnimation(fig, _update, frames=us.nt,
                                  interval=1000 / fps, blit=True)

    out = Path(out_path)
    if out.suffix == ".gif":
        ani.save(str(out), writer="pillow", fps=fps)
    else:
        ani.save(str(out), writer="ffmpeg", fps=fps)
    plt.close(fig)
    print(f"[anim] Time series animation saved → {out_path}")


def save_npz(us: US4DData, out_path: str | Path = "us4d.npz") -> None:
    np.savez_compressed(
        str(out_path),
        data=us.data,
        scale=np.array(us.scale),
        dim=np.array(us.dim),
        N=np.array(us.N),
        frame_rate=np.array(us.frame_rate),
        BScale=np.array(us.BScale),
    )
    size_mb = Path(out_path).stat().st_size / 1e6
    print(f"[save] Saved → {out_path}  ({size_mb:.1f} MB)")


def load_npz(path: str | Path) -> US4DData:
    npz = np.load(str(path))
    return US4DData(
        data=npz["data"],
        scale=npz["scale"].tolist(),
        dim=npz["dim"].tolist(),
        N=npz["N"].tolist(),
        frame_rate=float(npz["frame_rate"]),
        BScale=npz["BScale"].tolist(),
    )


def save_nifti_slice(us: US4DData, z: int,
                     out_path: str | Path = "slice_z104.nii.gz") -> None:
    import nibabel as nib

    data = us.data[:, :, z, :].astype(np.uint8)[:, :, np.newaxis, :]
    dx, dy = us.scale[0], us.scale[1]
    dt = 1.0 / us.frame_rate

    affine = np.array([
        [dx,  0,  0,  0],
        [ 0, dy,  0,  0],
        [ 0,  0,  1,  0],
        [ 0,  0,  0,  1],
    ])

    img = nib.Nifti1Image(data, affine=affine)
    hdr = img.header
    hdr.set_xyzt_units(xyz="mm", t="sec")
    hdr["pixdim"][4] = dt
    hdr["cal_min"] = 0.0
    hdr["cal_max"] = 255.0
    hdr["descrip"] = f"z={z} slice, 4D US (nx,ny,1,nt)".encode()

    nib.save(img, str(out_path))
    size_mb = Path(out_path).stat().st_size / 1e6
    print(f"[save] Slice z={z} NIfTI → {out_path}  ({size_mb:.1f} MB)  shape={data.shape}")


def save_nifti(us: US4DData, out_path: str | Path = "us4d.nii.gz") -> None:
    import nibabel as nib

    data = us.data.astype(np.uint8)
    dx, dy, dz = us.scale
    dt = 1.0 / us.frame_rate

    affine = np.array([
        [dx,  0,  0,  0],
        [ 0, dy,  0,  0],
        [ 0,  0, dz,  0],
        [ 0,  0,  0,  1],
    ])

    img = nib.Nifti1Image(data, affine=affine)
    hdr = img.header
    hdr.set_xyzt_units(xyz="mm", t="sec")
    hdr["pixdim"][4] = dt
    hdr["dim_info"] = 0
    hdr["scl_slope"] = 1.0
    hdr["scl_inter"] = 0.0
    hdr["cal_min"]   = float(us.BScale[0])
    hdr["cal_max"]   = float(us.BScale[1])
    hdr["descrip"]   = b"4D cardiac US  (nx,ny,nz,nt)"

    nib.save(img, str(out_path))
    size_mb = Path(out_path).stat().st_size / 1e6
    print(f"[save] NIfTI saved → {out_path}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    import nibabel as nib

    # JacklynX changed
    DICOM_FILES = [
        Path("p020_1a.dcm"),
        Path("p030_lp2a.dcm"),
    ]

    for DICOM_PATH in DICOM_FILES:
        if not DICOM_PATH.exists():
            print(f"\n[SKIP] {DICOM_PATH} not found")
            continue

        OUT_DIR = Path(DICOM_PATH.stem)
        OUT_DIR.mkdir(exist_ok=True)
        print(f"\n{'='*55}")
        print(f"Processing: {DICOM_PATH}  →  {OUT_DIR}/")
        print(f"{'='*55}")

        us = load_4d(DICOM_PATH, frame_rate=15.0)
        print_stats(us)

        plot_orthogonal_slices(us, t=0,
                               out_path=OUT_DIR / "orthogonal_slices.png")
        save_time_animation(us, axis="z",
                            out_path=OUT_DIR / "time_series_z.gif")

        save_npz(us,   OUT_DIR / "us4d.npz")
        save_nifti(us, OUT_DIR / "us4d.nii.gz")

        data   = us.data[:, :, :, 0].astype(np.uint8)
        dx, dy, dz = us.scale
        affine = np.array([[dx,0,0,0],[0,dy,0,0],[0,0,dz,0],[0,0,0,1]])
        frame0 = nib.Nifti1Image(data, affine)
        nib.save(frame0, str(OUT_DIR / "frame0_3d.nii.gz"))
        print(f"[save] frame0_3d.nii.gz saved  shape={data.shape}")

        print(f"Done → {OUT_DIR}/")

    print("\nAll files processed.")