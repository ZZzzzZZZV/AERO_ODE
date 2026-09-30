from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import torch
import xarray as xr

import paths_config as pc

def _time_to_str(t: np.datetime64) -> str:
    t64s = np.datetime64(t, "s")
    dt = datetime.utcfromtimestamp(int(t64s.astype("int64")))
    return dt.strftime("%Y/%m/%d/%H")

def _hrrr_array(handle) -> np.ndarray:
    return handle["data"][:] if "data" in handle else handle["fields"][:]

def _surface_slice(n_channels: int) -> slice:
    if n_channels >= 39:
        return slice(35, 39)
    if n_channels >= 24:
        return slice(20, 24)
    raise ValueError(f"HRRR channels too few for surface vars: C={n_channels}")

def load_demo_sample(
    year: int = pc.DEMO_YEAR,
    month: int = pc.DEMO_MONTH,
    day: int = pc.DEMO_DAY,
    time_steps: int = 73,
) -> Tuple[List[xr.Dataset], str, torch.Tensor, torch.Tensor]:
    era5_file = pc.ensure_exists(pc.era5_sample_nc(year, month, day), "ERA5 sample")
    hrrr_dir = pc.ensure_exists(pc.hrrr_truth_month_dir(year, month), "HRRR truth month")

    ds = xr.open_dataset(str(era5_file), decode_timedelta=False)
    ds0 = ds.isel(time=slice(0, 1))
    time_str = _time_to_str(ds0["time"].values[0])

    start_dt = datetime.utcfromtimestamp(
        int(np.datetime64(ds0["time"].values[0], "s").astype("int64"))
    )
    air_chunks, surface_chunks = [], []
    total = 0
    day_cursor = start_dt
    while total < time_steps:
        h5_path = Path(hrrr_dir) / f"{day_cursor.day:02d}.h5"
        pc.ensure_exists(h5_path, "HRRR daily file")
        with h5py.File(str(h5_path), "r") as handle:
            data = _hrrr_array(handle)
            air_chunks.append(data[:, :25, :, :])
            surface_chunks.append(data[:, _surface_slice(data.shape[1]), :, :])
        total += data.shape[0]
        day_cursor += timedelta(days=1)

    air_np = np.concatenate(air_chunks, axis=0)[:time_steps]
    surface_np = np.concatenate(surface_chunks, axis=0)[:time_steps]
    return (
        [ds0],
        time_str,
        torch.from_numpy(air_np[None, ...]),
        torch.from_numpy(surface_np[None, ...]),
    )
