from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

_COLLIDING = (
    "AERO_ODE_v3",
    "AERO_ODE_Dataset_v3",
    "inference_components",
    "Models_FiLM",
    "l2",
    "test_film_wb",
    "test_film_00z_RMSE",
    "Generate_Forecast_wb",
    "Generate_Forecast_Surface",
)


@contextmanager
def project_on_path(root: Path):
    root_s = str(Path(root).resolve())
    saved_path = list(sys.path)
    sys.path[:] = [root_s] + [p for p in sys.path if str(Path(p).resolve()) != root_s]
    popped = {name: sys.modules.pop(name) for name in _COLLIDING if name in sys.modules}
    try:
        yield
    finally:
        for name in _COLLIDING:
            sys.modules.pop(name, None)
        sys.modules.update(popped)
        sys.path[:] = saved_path
