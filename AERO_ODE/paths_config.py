"""
Cross-platform path configuration for AERO_ODE (Windows / Ubuntu).

Design principles:
- Build paths with pathlib.Path; do not hand-write "/" or "\\" string paths.
- Anchor directories are resolved from __file__, independent of the process cwd.
- Pass os.fspath() to third-party libraries (xarray / h5py / os.walk / numpy).

Demo data layout (August 2024):
    Data/ERA5/2024/08/DD/YYYYMMDD.nc   ERA5 initial conditions
    Data/HRRR/2024/08/DD.h5            HRRR ground truth
    Data/HRRR/lats.npy, lons.npy, geo.h5
    Data/HRRR/stat/mean_crop.npy, std_crop.npy

Environment variables:
    AERO_ODE_DATA_ROOT  Override the default AERO_ODE/Data directory (code/data on separate disks).
"""
from __future__ import annotations

import os
from pathlib import Path


# Anchor directories
AERO_ODE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("AERO_ODE_DATA_ROOT", AERO_ODE_ROOT / "Data"))

# Default demo data year/month (August 2024)
DEMO_YEAR = 2024
DEMO_MONTH = 8


def ensure_exists(path: Path, label: str) -> Path:
    """Return path if it exists; otherwise raise a readable error."""
    if not path.exists():
        raise FileNotFoundError(
            f"[paths_config] {label} not found: {os.fspath(path)}\n"
            f"Ensure data/weights are in place, or set AERO_ODE_DATA_ROOT."
        )
    return path


# ERA5 (initial conditions)
def era5_root() -> Path:
    """ERA5 data root: Data/ERA5."""
    return DATA_ROOT / "ERA5"


def era5_sample_nc(year: int = DEMO_YEAR, month: int = DEMO_MONTH, day: int = 1) -> Path:
    """Single-sample ERA5 file: Data/ERA5/YYYY/MM/DD/YYYYMMDD.nc."""
    return (
        era5_root()
        / f"{year:04d}"
        / f"{month:02d}"
        / f"{day:02d}"
        / f"{year:04d}{month:02d}{day:02d}.nc"
    )


# HRRR (ground truth + static statistics)
def hrrr_stat_root() -> Path:
    """HRRR static/statistics root: Data/HRRR (lats.npy, lons.npy, geo.h5, stat/)."""
    return DATA_ROOT / "HRRR"


def hrrr_truth_root(year: int = DEMO_YEAR) -> Path:
    """HRRR truth root: Data/HRRR/YYYY.

    The path must contain the year string to satisfy Dataset._scan_hrrr_files
    `year_str in root` filtering (Windows backslash paths include the year as well).
    """
    return hrrr_stat_root() / f"{year:04d}"


def hrrr_truth_month_dir(year: int = DEMO_YEAR, month: int = DEMO_MONTH) -> Path:
    """HRRR truth month directory: Data/HRRR/YYYY/MM (contains DD.h5)."""
    return hrrr_truth_root(year) / f"{month:02d}"


# Model weights (paths relative to each project root)
def ngcm_checkpoint(project_root: Path) -> Path:
    """NeuralGCM checkpoint in project: <project_root>/NeuralGCM_Weights/*.pkl."""
    weights_dir = Path(project_root) / "NeuralGCM_Weights"
    pkls = sorted(weights_dir.glob("*.pkl"))
    if not pkls:
        raise FileNotFoundError(
            f"[paths_config] No NeuralGCM weights (*.pkl) found in {os.fspath(weights_dir)}."
        )
    return pkls[0]


def film_checkpoint_dir(project_root: Path, variant: str = "checkpoints_film") -> Path:
    """FiLM checkpoint directory in project (checkpoints_film or checkpoints_film_v2)."""
    return Path(project_root) / variant


def surface_static_data(project_root: Path) -> Path:
    """Surface static feature directory: <project_root>/Hrrr_rb/data."""
    return Path(project_root) / "Hrrr_rb" / "data"


# Outputs
def predicted_output_root(project_root: Path) -> Path:
    """Forecast output root: <project_root>/outputs/predicted_hrrr_rb."""
    return Path(project_root) / "outputs" / "predicted_hrrr_rb"


if __name__ == "__main__":
    print("AERO_ODE_ROOT :", os.fspath(AERO_ODE_ROOT))
    print("DATA_ROOT     :", os.fspath(DATA_ROOT))
    print("era5_root     :", os.fspath(era5_root()), era5_root().exists())
    print("hrrr_stat_root:", os.fspath(hrrr_stat_root()), hrrr_stat_root().exists())
    print("hrrr_truth    :", os.fspath(hrrr_truth_root()), hrrr_truth_root().exists())
    print("era5_sample   :", os.fspath(era5_sample_nc()), era5_sample_nc().exists())
