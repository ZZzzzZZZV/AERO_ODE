import os
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


def load_demo_sample(
    year: int = pc.DEMO_YEAR,
    month: int = pc.DEMO_MONTH,
    day: int = 1,
    time_steps: int = 73,
) -> Tuple[List[xr.Dataset], str, torch.Tensor, torch.Tensor]:
    """Load ERA5 IC, HRRR pressure-level truth, and HRRR surface truth for one case."""
    era5_file = pc.ensure_exists(pc.era5_sample_nc(year, month, day), "ERA5 sample")
    hrrr_dir = pc.ensure_exists(pc.hrrr_truth_month_dir(year, month), "HRRR truth month")

    ds = xr.open_dataset(os.fspath(era5_file), decode_timedelta=False)
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
        with h5py.File(os.fspath(h5_path), "r") as f:
            data = f["fields"][...]
            air_chunks.append(data[:, :20, :, :])
            surface_chunks.append(data[:, 20:24, :, :])
        total += data.shape[0]
        day_cursor += timedelta(days=1)

    air_np = np.concatenate(air_chunks, axis=0)[:time_steps]
    surface_np = np.concatenate(surface_chunks, axis=0)[:time_steps]
    air_tensor = torch.from_numpy(air_np[None, ...])
    surface_tensor = torch.from_numpy(surface_np[None, ...])
    return [ds0], time_str, air_tensor, surface_tensor
