from __future__ import annotations

import os
from pathlib import Path
import sys

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

_DRAW_NEW = Path(__file__).resolve().parent.parent
_RMSE_DIR = _DRAW_NEW / "RMSE"
for _p in (_DRAW_NEW, _RMSE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from paths_config import (  # noqa: E402
    AIR_GRID_COLS,
    AIR_GRID_ROWS,
    AXES_LABEL_SIZE,
    COMBINED_RMSE_FIGSIZE,
    FIGURE_SAVE_DPI_HIGH,
    SUBPLOT_TITLE_SIZE,
    SURFACE_VAR_SHORT,
    UPPER_LEVELS,
    UPPER_VAR_SHORT,
    XLABEL_FORECAST_TIME,
    add_figure_legend_below,
    apply_paper_rcparams,
    apply_curve_grid_spacing,
    font_times_new_roman,
    save_curve_figure,
    upper_air_subplot_title,
)

from rmse_plot_common import (  # noqa: E402
    LINEWIDTHS,
    N_AIR_CHANNELS,
    N_SURFACE_VARS,
    _prepare_air_surface_lists,
    _plot_series_on_ax,
    get_model_color,
)

FONT_PATH = str(font_times_new_roman())
ACC_YLABEL = "ACC"


def setup_font(font_path: str = FONT_PATH) -> None:
    try:
        fm.fontManager.addfont(font_path)
        font_name = fm.FontProperties(fname=font_path).get_name()
        plt.rcParams["font.family"] = font_name
        print(f"Loaded font: {font_name}")
    except Exception as e:
        print(f"Warning: failed to load font {font_path}, using default font. Error: {e}")
    apply_paper_rcparams()


def _draw_acc_block(
    axes,
    col_start: int,
    air_data,
    surf_data,
    name_list,
    times,
    *,
    show_ylabel: bool,
) -> None:
    """Draw one 6x4 ACC block on axes columns col_start:col_start+4."""
    for v_idx, _var_name in enumerate(UPPER_VAR_SHORT):
        for l_idx, level in enumerate(UPPER_LEVELS):
            ax = axes[v_idx, col_start + l_idx]
            ch_idx = v_idx * 4 + l_idx
            series = [d[ch_idx] for d in air_data]
            _plot_series_on_ax(ax, times, series, name_list)
            ax.set_title(upper_air_subplot_title(v_idx, level), fontsize=SUBPLOT_TITLE_SIZE)
            ax.grid(True, alpha=0.3)
            if show_ylabel and l_idx == 0:
                ax.set_ylabel(ACC_YLABEL, fontsize=AXES_LABEL_SIZE)

    surf_row = AIR_GRID_ROWS
    for v_idx, var_name in enumerate(SURFACE_VAR_SHORT):
        ax = axes[surf_row, col_start + v_idx]
        series = []
        surf_names = []
        for d, name in zip(surf_data, name_list):
            if d is None:
                continue
            series.append(d[:, v_idx])
            surf_names.append(name)
        _plot_series_on_ax(ax, times, series, surf_names)
        ax.set_title(var_name, fontsize=SUBPLOT_TITLE_SIZE)
        if v_idx == AIR_GRID_COLS - 1:
            ax.set_xlabel(XLABEL_FORECAST_TIME, fontsize=AXES_LABEL_SIZE)
        if show_ylabel:
            ax.set_ylabel(ACC_YLABEL, fontsize=AXES_LABEL_SIZE)
        ax.grid(True, alpha=0.3)


def _save_acc_figure(fig, axes, name_list, save_base, save_formats, dpi, bbox_inches, show):
    fig.tight_layout()
    apply_curve_grid_spacing(fig)
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


def plot_acc_combined(
    air_data_list,
    surface_data_list,
    name_list,
    save_base=None,
    save_formats=("png", "pdf"),
    dpi: int = FIGURE_SAVE_DPI_HIGH,
    bbox_inches: str = "tight",
    show: bool = True,
) -> None:
    """
    Plot upper-air + surface ACC comparison (6 rows x 4 columns).

    Args:
        air_data_list: each array shape (20, T) or (T, 20)
        surface_data_list: each array shape (T, 4) or (4, T); None for missing
        name_list: model display names
        save_base: e.g. figures_rb/acc_48_rb
    """
    air_data, surf_data, time_len = _prepare_air_surface_lists(
        air_data_list, surface_data_list, name_list
    )
    times = np.arange(time_len)
    n_rows = AIR_GRID_ROWS + 1
    fig, axes = plt.subplots(n_rows, AIR_GRID_COLS, figsize=COMBINED_RMSE_FIGSIZE)
    _draw_acc_block(axes, 0, air_data, surf_data, name_list, times, show_ylabel=True)
    _save_acc_figure(fig, axes, name_list, save_base, save_formats, dpi, bbox_inches, show)
