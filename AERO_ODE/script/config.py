import os
from pathlib import Path

import paths_config as pc

AERO_ODE_ROOT = pc.AERO_ODE_ROOT
AIR_ROOT = pc.AIR_ROOT
SURFACE_ROOT = pc.SURFACE_ROOT
OUTPUT_ROOT = AERO_ODE_ROOT / "quick-start_output"

FORECAST_HOURS = 72
WRF_FORECAST_HOURS = 48
TIME_STEPS = FORECAST_HOURS + 1


def forecast_skill_slice(arr, time_axis: int = 0):
    import numpy as np

    a = np.asarray(arr)
    n = a.shape[time_axis]
    if n == FORECAST_HOURS or n <= 1:
        return a
    sl = [slice(None)] * a.ndim
    sl[time_axis] = slice(1, None)
    return a[tuple(sl)]


YEAR = pc.DEMO_YEAR
MONTH = pc.DEMO_MONTH
DAY = pc.DEMO_DAY

PLOT_LEADS = [6, 12, 24, 48]
SURFACE_VARS = {"mslp": 0, "t2m": 3}
AIR_LEVELS = (300, 500, 700, 850, 925)
AIR_VARS = ("Z", "T", "S", "U", "V")


def configure_runtime(gpu_id: str = "0") -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".30"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    cache_dir = Path(__file__).resolve().parents[1] / ".jax_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(cache_dir))


def save_tensor(tensor, name: str) -> Path:
    import torch

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_ROOT / name
    torch.save(tensor.cpu(), path)
    return path


def usable_file(path: Path | str) -> bool:
    p = Path(path)
    return p.is_file() and p.stat().st_size > 0
