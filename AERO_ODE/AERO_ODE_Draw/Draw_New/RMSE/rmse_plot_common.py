"""
Shared RMSE curve plotting: upper-air + surface combined 6x4 subplots.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

_DRAW_NEW = Path(__file__).resolve().parent.parent
if str(_DRAW_NEW) not in sys.path:
    sys.path.insert(0, str(_DRAW_NEW))

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
    rmse_air_ylabel,
    save_curve_figure,
    upper_air_subplot_title,
)

FONT_PATH = str(font_times_new_roman())
N_AIR_CHANNELS = 20
N_SURFACE_VARS = 4
SURFACE_VAR_UNITS = ("Pa", "m/s", "m/s", "K")

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
LINEWIDTHS = [1.5] * 10


def get_model_color(model_name: str, fallback_index: int = 0) -> str:
    return MODEL_COLORS.get(model_name, COLORS[fallback_index % len(COLORS)])


def setup_font(font_path: str = FONT_PATH) -> None:
    try:
        fm.fontManager.addfont(font_path)
        font_name = fm.FontProperties(fname=font_path).get_name()
        plt.rcParams["font.family"] = font_name
        print(f"Loaded font: {font_name}")
    except Exception as e:
        print(f"Warning: failed to load font {font_path}, using default font. Error: {e}")
    apply_paper_rcparams()


def to_channel_time_shape(arr: np.ndarray, n_channels: int = N_AIR_CHANNELS) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"Input must be 2D; got shape={arr.shape}")
    if arr.shape[0] == n_channels:
        return arr
    if arr.shape[1] == n_channels:
        return arr.T
    raise ValueError(
        f"Cannot infer channel dimension (expected one axis to be {n_channels}), got shape={arr.shape}"
    )


def to_time_var_shape(arr: np.ndarray, n_vars: int = N_SURFACE_VARS) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"Input must be 2D; got shape={arr.shape}")
    if arr.shape[1] == n_vars:
        return arr
    if arr.shape[0] == n_vars:
        return arr.T
    raise ValueError(f"Cannot infer variable dimension (expected one axis to be {n_vars}), got shape={arr.shape}")


def _prepare_air_surface_lists(air_data_list, surface_data_list, name_list):
    if not (len(air_data_list) == len(surface_data_list) == len(name_list)):
        raise ValueError("air_data_list, surface_data_list, and name_list must have the same length.")
    air_data = [to_channel_time_shape(d) for d in air_data_list]
    surf_data = []
    for raw in surface_data_list:
        if raw is None:
            surf_data.append(None)
        else:
            surf_data.append(to_time_var_shape(raw))
    time_len = air_data[0].shape[1]
    for i, d in enumerate(air_data):
        if d.shape[1] != time_len:
            raise ValueError(f"Upper-air series {i} time length mismatch: {d.shape[1]} vs {time_len}")
    for i, d in enumerate(surf_data):
        if d is not None and d.shape[0] != time_len:
            raise ValueError(f"Surface series {i} time length mismatch: {d.shape[0]} vs {time_len}")
    return air_data, surf_data, time_len


def _plot_series_on_ax(ax, times, series_list, name_list) -> None:
    for i, (y, name) in enumerate(zip(series_list, name_list)):
        ax.plot(
            times,
            y,
            color=get_model_color(name, i),
            linestyle="-",
            linewidth=LINEWIDTHS[i % len(LINEWIDTHS)],
            alpha=1.0,
            label=name,
        )


def _draw_rmse_block(
    axes,
    col_start: int,
    air_data,
    surf_data,
    name_list,
    times,
    *,
    show_ylabel: bool,
) -> None:
    """Draw one 6x4 RMSE block on axes columns col_start:col_start+4."""
    for v_idx, _var_name in enumerate(UPPER_VAR_SHORT):
        for l_idx, level in enumerate(UPPER_LEVELS):
            ax = axes[v_idx, col_start + l_idx]
            ch_idx = v_idx * 4 + l_idx
            series = [d[ch_idx] for d in air_data]
            _plot_series_on_ax(ax, times, series, name_list)
            ax.set_title(upper_air_subplot_title(v_idx, level), fontsize=SUBPLOT_TITLE_SIZE)
            ax.grid(True, alpha=0.3)
            if show_ylabel and l_idx == 0:
                ax.set_ylabel(rmse_air_ylabel(v_idx), fontsize=AXES_LABEL_SIZE)

    surf_row = AIR_GRID_ROWS
    for v_idx, (var_name, unit) in enumerate(zip(SURFACE_VAR_SHORT, SURFACE_VAR_UNITS)):
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
            ax.set_ylabel(f"RMSE ({unit})", fontsize=AXES_LABEL_SIZE)
        ax.grid(True, alpha=0.3)


def _save_rmse_figure(fig, axes, name_list, save_base, save_formats, dpi, bbox_inches, show):
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


def plot_rmse_combined(
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
    Plot upper-air + surface RMSE comparison (6 rows x 4 columns).

    Args:
        air_data_list: each entry shape (20, T) or (T, 20)
        surface_data_list: each entry shape (T, 4) or (4, T); None for missing
        name_list: model names
        save_base: e.g. figures_rb/rmse_48_rb
    """
    air_data, surf_data, time_len = _prepare_air_surface_lists(
        air_data_list, surface_data_list, name_list
    )
    times = np.arange(time_len)
    n_rows = AIR_GRID_ROWS + 1
    fig, axes = plt.subplots(n_rows, AIR_GRID_COLS, figsize=COMBINED_RMSE_FIGSIZE)
    _draw_rmse_block(axes, 0, air_data, surf_data, name_list, times, show_ylabel=True)
    _save_rmse_figure(fig, axes, name_list, save_base, save_formats, dpi, bbox_inches, show)
