from __future__ import annotations

from calendar import monthrange
from datetime import date
from typing import Tuple

import numpy as np

import paths_config as pc
from script.calc_wrf_rmse import evaluate_wrf_rmse
from script.config import MONTH, OUTPUT_ROOT, YEAR, usable_file
from script.project_import import project_on_path

AIR_RMSE_NPY = OUTPUT_ROOT / "rmse_aero_00z.npy"
SURFACE_RMSE_NPY = OUTPUT_ROOT / "rmse_surface_00z.npy"
WRF_RMSE_NPY = OUTPUT_ROOT / "rmse_wrf_00z.npy"


def _as_time_first(arr: np.ndarray, n_channels: int) -> np.ndarray:
    data = np.asarray(arr, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"expected 2D RMSE array, got shape={data.shape}")
    if data.shape[1] == n_channels:
        return data
    if data.shape[0] == n_channels:
        return data.T
    raise ValueError(
        f"RMSE array shape {data.shape} has no axis of size {n_channels}"
    )


def _call_compute_rmse(root, module_name: str):
    with project_on_path(root):
        module = __import__(module_name)
        return module.compute_rmse(write_dir=None, verbose=False)


def _month_bounds() -> Tuple[date, date]:
    last = monthrange(YEAR, MONTH)[1]
    return date(YEAR, MONTH, 1), date(YEAR, MONTH, last)


def _call_wrf_rmse() -> np.ndarray:
    start, end = _month_bounds()
    rmse, _, _ = evaluate_wrf_rmse(start, end, progress_every=0)
    return np.asarray(rmse, dtype=np.float64)


def evaluate_wd_rmse(*, recompute: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    if recompute or not usable_file(AIR_RMSE_NPY):
        air_raw = _call_compute_rmse(pc.AIR_ROOT, "test_film_wb")
        if air_raw is None:
            raise RuntimeError("test_film_wb.compute_rmse() returned no RMSE array")
        np.save(AIR_RMSE_NPY, np.asarray(air_raw, dtype=np.float32))
        air = _as_time_first(air_raw, 25)
    else:
        air = _as_time_first(np.load(AIR_RMSE_NPY), 25)

    if recompute or not usable_file(SURFACE_RMSE_NPY):
        surface_raw = _call_compute_rmse(pc.SURFACE_ROOT, "test_film_00z_RMSE")
        if surface_raw is None:
            raise RuntimeError("test_film_00z_RMSE.compute_rmse() returned no RMSE array")
        np.save(SURFACE_RMSE_NPY, np.asarray(surface_raw, dtype=np.float32))
        surface = _as_time_first(surface_raw, 4)
    else:
        surface = _as_time_first(np.load(SURFACE_RMSE_NPY), 4)

    if recompute or not usable_file(WRF_RMSE_NPY):
        wrf = _call_wrf_rmse()
        np.save(WRF_RMSE_NPY, np.asarray(wrf, dtype=np.float32))
    else:
        wrf = np.load(WRF_RMSE_NPY).astype(np.float64)

    return air, surface, wrf
