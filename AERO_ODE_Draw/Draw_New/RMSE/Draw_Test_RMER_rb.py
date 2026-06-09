"""
East (Rb) RMSE curves: upper-air + surface combined in one 6x4 figure.

Run: python Draw_Test_RMER_rb.py
"""
from pathlib import Path
import sys

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_DRAW_NEW = _SCRIPT_DIR.parent
if str(_DRAW_NEW) not in sys.path:
    sys.path.insert(0, str(_DRAW_NEW))

from paths_config import repo  # noqa: E402
from rmse_plot_common import plot_rmse_combined, setup_font  # noqa: E402

# Load data (channels 0:19 upper-air, 20:23 surface)
rmse_Aero_ODE_48 = np.load(str(repo("Rb_RMSE/Aero_ODE_rmse.npy")))[0:48, ...]
rmse_Aero_ODE_72 = np.load(str(repo("Rb_RMSE/Aero_ODE_rmse.npy")))[0:72, ...]

rmse_NeuralGCM_48 = np.load(str(repo("Rb_RMSE/neuralgcm_rmse.npy")))[0:48, :20, ...]
rmse_NeuralGCM_72 = np.load(str(repo("Rb_RMSE/neuralgcm_rmse.npy")))[0:72, :20, ...]

rmse_PanGu_48 = np.load(str(repo("Rb_RMSE/pangu_rmse.npy")))[:48, ...]
rmse_PanGu_72 = np.load(str(repo("Rb_RMSE/pangu_rmse.npy")))[:72, ...]

rmse_yl_bc_nwp_48 = np.load(str(repo("Rb_RMSE/YingLong_nwp_rmse.npy"))).swapaxes(0, 1)[:, :24]
rmse_nwp_48 = np.load(str(repo("Rb_RMSE/nwp_rmse.npy")))

rmse_IFS_72 = np.load(str(repo("Rb_RMSE/ifs_rmse.npy")))[:72, ...]

print("rmse_corrected shape:", rmse_Aero_ODE_48.shape)


def _split_air_surface(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return arr[:, :20, ...], arr[:, 20:24, ...]


if __name__ == "__main__":
    setup_font()

    plot_rmse_combined(
        air_data_list=[
            _split_air_surface(rmse_Aero_ODE_48)[0],
            _split_air_surface(rmse_nwp_48)[0],
            _split_air_surface(rmse_yl_bc_nwp_48)[0],
        ],
        surface_data_list=[
            _split_air_surface(rmse_Aero_ODE_48)[1],
            _split_air_surface(rmse_nwp_48)[1],
            _split_air_surface(rmse_yl_bc_nwp_48)[1],
        ],
        name_list=["AERO-ODE", "WRF-ARW", "YingLong-WRF"],
        save_base="figures_rb/rmse_48_rb",
        show=False,
    )

    lam72_a, lam72_s = _split_air_surface(rmse_Aero_ODE_72)
    pangu72_a, pangu72_s = _split_air_surface(rmse_PanGu_72)
    ngcm72_a = rmse_NeuralGCM_72
    ifs72_a, ifs72_s = _split_air_surface(rmse_IFS_72)

    plot_rmse_combined(
        air_data_list=[lam72_a, pangu72_a, ngcm72_a, ifs72_a],
        surface_data_list=[lam72_s, pangu72_s, None, ifs72_s],
        name_list=["AERO-ODE", "PanGu-Weather", "NeuralGCM 1.4", "IFS"],
        save_base="figures_rb/rmse_72_rb",
        show=False,
    )
