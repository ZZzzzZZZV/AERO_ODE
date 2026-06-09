from pathlib import Path
import sys
_DRAW_NEW = Path(__file__).resolve().parent
for _ in range(4):
    if (_DRAW_NEW / "paths_config.py").exists():
        break
    _DRAW_NEW = _DRAW_NEW.parent
else:
    raise RuntimeError("paths_config.py not found")
if str(_DRAW_NEW) not in sys.path:
    sys.path.insert(0, str(_DRAW_NEW))
from paths_config import (
    repo,
    font_times_new_roman,
    SUBPLOT_TITLE_SIZE,
    LEGEND_FONT_SIZE,
    AXES_LABEL_SIZE,
    AIR_FIGSIZE,
    SURFACE_FIGSIZE,
    EXTREME_AIR_FIGSIZE,
    EXTREME_SURFACE_FIGSIZE,
    UPPER_VAR_SHORT,
    UPPER_LEVELS,
    SURFACE_VAR_SHORT,
    XLABEL_FORECAST_TIME,
    upper_air_subplot_title,
    rmse_air_ylabel,
    add_figure_legend_below,
    apply_paper_rcparams,
    FIGURE_SAVE_DPI,
    FIGURE_SAVE_DPI_HIGH,
    save_curve_figure,
)
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Load data
# Run command: python Draw_Test_RMER_surface_rb.py
rmse_Aero_ODE_48 = np.load(str(repo("Rb_Ob_RMSE/Aero_ODE_ob_RMSE_wb.npy")))

# Legacy fine-tune run 14
rmse_yl_nwp_48 = np.load(str(repo("Rb_Ob_RMSE/yl_nwp_marge_ob_RMSE_wb.npy")))
print("rmse_yl_nwp_48.shape: ", rmse_yl_nwp_48.shape)

rmse_nwp_48_surface = np.load(str(repo("Rb_Ob_RMSE/nwp_ob_RMSE_wb.npy")))
print("rmse_nwp_48_surface.shape: ", rmse_nwp_48_surface.shape)

rmse_hrrr = np.load(str(repo("Rb_Ob_RMSE/hrrr_ob_RMSE_wb.npy")))


"""
Surface RMSE comparison plotting (PNG + PDF)
"""
# Times New Roman font path
FONT_PATH = str(font_times_new_roman())

VAR_NAMES = list(SURFACE_VAR_SHORT)
VAR_UNITS = ["Pa", "m/s", "m/s", "K"]

# Okabe-Ito colorblind-friendly colors keyed by method name.
MODEL_COLORS = {
    "AERO-ODE": "#D55E00",
    "YingLong-WRF": "#0072B2",
    "YL NWP": "#0072B2",
    "WRF-ARW": "#009E73",
    "NWP": "#009E73",
    "PanGu-Weather": "#CC79A7",
    "PanGu": "#CC79A7",
    "NeuralGCM 1.4": "#E69F00",
    "NeuralGCM": "#E69F00",
    "NGCM": "#E69F00",
    "IFS": "#56B4E9",
    "HRRR Reference": "#4D4D4D",
}
COLORS = list(dict.fromkeys(MODEL_COLORS.values()))

# Line widths matching series above
LINEWIDTHS = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]


def get_model_color(model_name, fallback_index=0):
    return MODEL_COLORS.get(model_name, COLORS[fallback_index % len(COLORS)])


def setup_font(font_path=FONT_PATH):
    """Setup font; fallback to default on failure."""
    try:
        fm.fontManager.addfont(font_path)
        font_name = fm.FontProperties(fname=font_path).get_name()
        plt.rcParams["font.family"] = font_name
        print(f"Loaded font: {font_name}")
    except Exception as e:
        print(f"Warning: failed to load font {font_path}, using default font. Error: {e}")
    apply_paper_rcparams()


def to_time_var_shape(arr):
    """
    Normalize array to shape (T, 4).
    Accept (T, 4) or (4, T).
    """
    if arr.ndim != 2:
        raise ValueError(f"Input must be 2D; got shape={arr.shape}")
    if arr.shape[1] == 4:
        return arr
    if arr.shape[0] == 4:
        return arr.T
    raise ValueError(f"Cannot infer variable dim (expected axis size 4), shape={arr.shape}")


def plot_rmse_surface(
    data_list,
    name_list,
    save_base=None,
    save_formats=("png", "pdf"),
    dpi=FIGURE_SAVE_DPI_HIGH,
    bbox_inches="tight",
    show=True,
):
    """
    Plot surface RMSE comparison

    Args:
        data_list: arrays (T, 4) or (4, T)
        name_list: series labels
        save_base: path prefix without suffix, e.g. figures/rmse_surface_72
                   writes .png and .pdf
        save_formats: output formats, default ("png", "pdf")
        dpi: raster DPI (PDF vectors ignore dpi)
        bbox_inches: passed to fig.savefig, often "tight"
        show: call plt.show(); set False for batch
    """
    if len(data_list) != len(name_list):
        raise ValueError("data_list and name_list length must match.")

    data = [to_time_var_shape(d) for d in data_list]
    time_len = data[0].shape[0]

    for i, d in enumerate(data):
        if d.shape[0] != time_len:
            raise ValueError(f"Series {i} time length mismatch: {d.shape[0]} vs {time_len}")

    times = np.arange(time_len)
    fig, axes = plt.subplots(1, 4, figsize=SURFACE_FIGSIZE)

    for v_idx, (var_name, unit) in enumerate(zip(VAR_NAMES, VAR_UNITS)):
        ax = axes[v_idx]

        for i, (d, name) in enumerate(zip(data, name_list)):
            ax.plot(
                times,
                d[:, v_idx],
                color=get_model_color(name, i),
                linestyle="-",
                linewidth=LINEWIDTHS[i % len(LINEWIDTHS)],
                alpha=1.0,
                label=name,
            )

        ax.set_title(var_name, fontsize=SUBPLOT_TITLE_SIZE)
        ax.set_xlabel(XLABEL_FORECAST_TIME, fontsize=AXES_LABEL_SIZE)
        ax.set_ylabel(f"RMSE ({unit})", fontsize=AXES_LABEL_SIZE)
        ax.grid(True, alpha=0.3)

    add_figure_legend_below(fig, axes, ncol=len(name_list))

    if save_base:
        directory = os.path.dirname(save_base)
        if directory:
            os.makedirs(directory, exist_ok=True)

        for fmt in save_formats:
            fmt = fmt.lower().lstrip(".")
            out_path = f"{save_base}.{fmt}"
            save_curve_figure(fig, out_path, dpi=dpi, bbox_inches=bbox_inches)
            print(f"Saved: {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


# Example usage
setup_font()

# plot_rmse_surface(
#     [rmse_Aero_ODE_72, rmse_PanGu_72_surface],
#     ["AERO-ODE", "PanGu"],
#     title="Comparison of RMSE for 72 Steps",
#     save_base="./figures/rmse_72_surface_rb",
# )

# plot_rmse_surface(
#     [rmse_Aero_ODE_48, rmse_PanGu_48_surface, rmse_yl_48],
#     ["AERO-ODE", "PanGu", "YingLong(PanGu)"],
#     title="Comparison of RMSE for 48 Steps",
#     save_base="figures/rmse_surface_48_three",
# )

# plot_rmse_surface(
#     [rmse_Aero_ODE_48, rmse_PanGu_48_surface, rmse_nwp_48_surface, rmse_yl_48, rmse_yl_nwp_48],
#     ["AERO-ODE", "PanGu", "NWP", "YingLong(PanGu)", "YingLong(NWP)"],
#     title="Comparison of RMSE for 48 Steps",
#     save_base="figures/rmse_surface_48_four",
# )

plot_rmse_surface(
    [rmse_Aero_ODE_48, rmse_nwp_48_surface, rmse_yl_nwp_48, rmse_hrrr],
    ["AERO-ODE", "WRF-ARW", "YingLong-WRF", "HRRR Reference"],
    save_base="./figures/rmse_48_ob_rb",
)