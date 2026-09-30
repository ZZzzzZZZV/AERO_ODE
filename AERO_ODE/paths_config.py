from __future__ import annotations

import os
import sys
from pathlib import Path

AERO_ODE_ROOT = Path(__file__).resolve().parent
if str(AERO_ODE_ROOT) not in sys.path:
    sys.path.insert(0, str(AERO_ODE_ROOT))

from common.paths import ROOT

AERO_ODE_ROOT = ROOT
DATA_ROOT = ROOT / "data"
AIR_ROOT = ROOT / "AERO_AIR_WD"
SURFACE_ROOT = ROOT / "AERO_Surface_WD"

DEMO_YEAR = 2024
DEMO_MONTH = 4
DEMO_DAY = 1

NGCM_WEIGHT = (
    ROOT
    / "shared_assets/NeuralGCM_Weights"
    / "neuralgcm_04_30_2024_neural_gcm_dynamic_forcing_deterministic_1_4_deg.pkl"
)
AIR_CKPT = AIR_ROOT / "checkpoints_film_25/model_ep5.pth"
SURFACE_CKPT = SURFACE_ROOT / "checkpoints_film_v2/model_ep5.pth"

def ensure_exists(path: Path, label: str) -> Path:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"[paths_config] missing {label}: {os.fspath(path)}"
        )
    return Path(path)

def era5_root() -> Path:
    return DATA_ROOT / "era5_test"

def era5_sample_nc(year: int = DEMO_YEAR, month: int = DEMO_MONTH, day: int = DEMO_DAY) -> Path:
    return (
        era5_root()
        / f"{year:04d}"
        / f"{month:02d}"
        / f"{day:02d}"
        / f"{year:04d}{month:02d}{day:02d}.nc"
    )

def hrrr_test_root() -> Path:
    return DATA_ROOT / "hrrr_test"

def hrrr_truth_month_dir(year: int = DEMO_YEAR, month: int = DEMO_MONTH) -> Path:
    dated = hrrr_test_root() / f"{year:04d}" / f"{month:02d}"
    if dated.is_dir():
        return dated
    return hrrr_test_root() / f"{month:02d}"

def hrrr_forecast_root() -> Path:
    return DATA_ROOT / "hrrr_forecast"

def hrrr_forecast_month_dir(year: int = DEMO_YEAR, month: int = DEMO_MONTH) -> Path:
    dated = hrrr_forecast_root() / f"{year:04d}" / f"{month:02d}"
    if dated.is_dir():
        return dated
    return hrrr_forecast_root() / f"{month:02d}"

def hrrr_stat_root() -> Path:
    return DATA_ROOT / "hrrr_stat"

def forecast_output_root() -> Path:
    return DATA_ROOT / "forecast_output"

def air_geo_root() -> Path:
    return AIR_ROOT / "Hrrr_rb"

def surface_static_data() -> Path:
    return SURFACE_ROOT / "Hrrr_rb/data"

def ngcm_checkpoint() -> Path:
    return ensure_exists(NGCM_WEIGHT, "NeuralGCM weights")

def air_checkpoint() -> Path:
    return ensure_exists(AIR_CKPT, "AIR checkpoint")

def surface_checkpoint() -> Path:
    return ensure_exists(SURFACE_CKPT, "Surface checkpoint")

if __name__ == "__main__":
    print("ROOT          :", os.fspath(ROOT))
    print("era5_sample   :", os.fspath(era5_sample_nc()), era5_sample_nc().exists())
    print("hrrr_month    :", os.fspath(hrrr_truth_month_dir()), hrrr_truth_month_dir().exists())
    for day in (1, 2, 3, 4):
        h5 = hrrr_truth_month_dir() / f"{day:02d}.h5"
        print(f"hrrr {day:02d}.h5    :", os.fspath(h5), h5.exists())
    print("hrrr_stat     :", os.fspath(hrrr_stat_root()), hrrr_stat_root().exists())
    print("wrf_month     :", os.fspath(hrrr_forecast_month_dir()), hrrr_forecast_month_dir().exists())
    print("wrf 01.h5     :", os.fspath(hrrr_forecast_month_dir() / "01.h5"), (hrrr_forecast_month_dir() / "01.h5").exists())
    print("air_ckpt      :", os.fspath(AIR_CKPT), AIR_CKPT.exists())
    print("surface_ckpt  :", os.fspath(SURFACE_CKPT), SURFACE_CKPT.exists())
    print("ngcm_weight   :", os.fspath(NGCM_WEIGHT), NGCM_WEIGHT.exists())
