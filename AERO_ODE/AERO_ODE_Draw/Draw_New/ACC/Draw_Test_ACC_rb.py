"""
East (Rb) ACC curves: upper-air + surface combined in one 6x4 figure.

Run: python Draw_Test_ACC_rb.py
"""
from pathlib import Path
import sys

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_DRAW_NEW = _SCRIPT_DIR.parent
for _p in (_DRAW_NEW, _SCRIPT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from paths_config import repo  # noqa: E402
from acc_plot_common import plot_acc_combined, setup_font  # noqa: E402

acc_lam_48 = np.load(str(repo("Rb_ACC/Aero_ODE_acc.npy")))[0:48, ...]
acc_lam_72 = np.load(str(repo("Rb_ACC/Aero_ODE_acc.npy")))

acc_ngcm_72 = np.load(str(repo("Rb_ACC/neuralgcm_acc.npy")))[:, :20, ...]

acc_pangu_72 = np.load(str(repo("Rb_ACC/pangu_acc.npy")))

acc_yl_48 = np.load(str(repo("Rb_ACC/YingLong_nwp_acc.npy"))).swapaxes(0, 1)[:, :24]
acc_nwp_48 = np.load(str(repo("Rb_ACC/nwp_acc.npy")))

acc_ifs_72 = np.load(str(repo("Rb_ACC/ifs_acc.npy")))


def _split_air_surface(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return arr[:, :20, ...], arr[:, 20:24, ...]


if __name__ == "__main__":
    setup_font()

    plot_acc_combined(
        air_data_list=[
            _split_air_surface(acc_lam_48)[0],
            _split_air_surface(acc_nwp_48)[0],
            _split_air_surface(acc_yl_48)[0],
        ],
        surface_data_list=[
            _split_air_surface(acc_lam_48)[1],
            _split_air_surface(acc_nwp_48)[1],
            _split_air_surface(acc_yl_48)[1],
        ],
        name_list=["AERO-ODE", "WRF-ARW", "YingLong-WRF"],
        save_base="figures_rb/acc_48_rb",
        show=False,
    )

    lam72_a, lam72_s = _split_air_surface(acc_lam_72)
    pangu72_a, pangu72_s = _split_air_surface(acc_pangu_72)
    ifs72_a, ifs72_s = _split_air_surface(acc_ifs_72)

    plot_acc_combined(
        air_data_list=[lam72_a, pangu72_a, acc_ngcm_72, ifs72_a],
        surface_data_list=[lam72_s, pangu72_s, None, ifs72_s],
        name_list=["AERO-ODE", "PanGu-Weather", "NeuralGCM 1.4", "IFS"],
        save_base="figures_rb/acc_72_rb",
        show=False,
    )
