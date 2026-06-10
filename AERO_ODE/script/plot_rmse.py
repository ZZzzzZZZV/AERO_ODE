"""Combined upper-air + surface RMSE curves (6x4 layout, self-contained)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from script.config import OUTPUT_ROOT
from script.plot_style import (
    AIR_GRID_COLS,
    AIR_GRID_ROWS,
    AXES_LABEL_SIZE,
    COMBINED_RMSE_FIGSIZE,
    FIGURE_SAVE_DPI_HIGH,
    SUBPLOT_TITLE_SIZE,
    SURFACE_VAR_SHORT,
    SURFACE_VAR_UNITS,
    UPPER_LEVELS,
    UPPER_VAR_SHORT,
    XLABEL_FORECAST_TIME,
    add_figure_legend_below,
    apply_curve_grid_spacing,
    apply_paper_rcparams,
    get_model_color,
    rmse_air_ylabel,
    save_curve_figure,
    upper_air_subplot_title,
)

N_AIR_CHANNELS = 20
N_SURFACE_VARS = 4
MODEL_NAME = "AERO-ODE"
LINEWIDTH = 1.5


def _align_hw(pred: torch.Tensor, truth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if pred.shape[-2:] != truth.shape[-2:]:
        pred = pred.transpose(-1, -2)
    return pred, truth


def compute_air_rmse_array(
    predicted: torch.Tensor,
    truth: torch.Tensor,
    n_channels: int = N_AIR_CHANNELS,
) -> np.ndarray:
    """Per-channel RMSE, shape (20, T)."""
    pred, gt = _align_hw(predicted.detach().cpu(), truth.detach().cpu())
    n_ch = min(pred.shape[2], gt.shape[2], n_channels)
    t_len = min(pred.shape[1], gt.shape[1])
    out = np.zeros((n_ch, t_len), dtype=np.float64)
    for c in range(n_ch):
        for t in range(t_len):
            diff = pred[0, t, c] - gt[0, t, c]
            out[c, t] = torch.sqrt(torch.mean(diff ** 2)).item()
    return out


def compute_surface_rmse_array(
    predicted: torch.Tensor,
    truth: torch.Tensor,
    n_vars: int = N_SURFACE_VARS,
) -> np.ndarray:
    """Per-variable RMSE, shape (T, 4)."""
    pred, gt = _align_hw(predicted.detach().cpu(), truth.detach().cpu())
    n_vars = min(pred.shape[2], gt.shape[2], n_vars)
    t_len = min(pred.shape[1], gt.shape[1])
    out = np.zeros((t_len, n_vars), dtype=np.float64)
    for v in range(n_vars):
        for t in range(t_len):
            diff = pred[0, t, v] - gt[0, t, v]
            out[t, v] = torch.sqrt(torch.mean(diff ** 2)).item()
    return out


def _plot_series_on_ax(ax, times, series_list, name_list) -> None:
    for i, (y, name) in enumerate(zip(series_list, name_list)):
        ax.plot(
            times, y, color=get_model_color(name, i), linestyle="-",
            linewidth=LINEWIDTH, alpha=1.0, label=name,
        )


def _draw_rmse_block(axes, air_data, surf_data, name_list, times) -> None:
    for v_idx in range(len(UPPER_VAR_SHORT)):
        for l_idx, level in enumerate(UPPER_LEVELS):
            ax = axes[v_idx, l_idx]
            ch_idx = v_idx * 4 + l_idx
            series = [d[ch_idx] for d in air_data]
            _plot_series_on_ax(ax, times, series, name_list)
            ax.set_title(upper_air_subplot_title(v_idx, level), fontsize=SUBPLOT_TITLE_SIZE)
            ax.grid(True, alpha=0.3)
            if l_idx == 0:
                ax.set_ylabel(rmse_air_ylabel(v_idx), fontsize=AXES_LABEL_SIZE)

    surf_row = AIR_GRID_ROWS
    for v_idx, (var_name, unit) in enumerate(zip(SURFACE_VAR_SHORT, SURFACE_VAR_UNITS)):
        ax = axes[surf_row, v_idx]
        series = [d[:, v_idx] for d in surf_data]
        _plot_series_on_ax(ax, times, series, name_list)
        ax.set_title(var_name, fontsize=SUBPLOT_TITLE_SIZE)
        ax.set_xlabel(XLABEL_FORECAST_TIME, fontsize=AXES_LABEL_SIZE)
        ax.set_ylabel(f"RMSE ({unit})", fontsize=AXES_LABEL_SIZE)
        ax.grid(True, alpha=0.3)


def _plot_rmse_combined(air_rmse: np.ndarray, surface_rmse: np.ndarray, name_list, save_base: str) -> None:
    apply_paper_rcparams()
    times = np.arange(air_rmse.shape[1])
    n_rows = AIR_GRID_ROWS + 1
    fig, axes = plt.subplots(n_rows, AIR_GRID_COLS, figsize=COMBINED_RMSE_FIGSIZE)
    _draw_rmse_block(axes, [air_rmse], [surface_rmse], name_list, times)
    fig.tight_layout()
    apply_curve_grid_spacing(fig)
    add_figure_legend_below(fig, axes, ncol=len(name_list))
    directory = os.path.dirname(save_base)
    if directory:
        os.makedirs(directory, exist_ok=True)
    out_path = f"{save_base}.png"
    save_curve_figure(fig, out_path, dpi=FIGURE_SAVE_DPI_HIGH)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_combined_rmse(
    air_pred: torch.Tensor,
    air_truth: torch.Tensor,
    surface_pred: torch.Tensor,
    surface_truth: torch.Tensor,
    output_path: Optional[Path] = None,
    model_name: str = MODEL_NAME,
) -> Path:
    """
    Plot 6x4 combined RMSE figure: upper-air (Z/T/S/U/V x 4 levels) + surface (MSLP/U10/V10/T2M).
    """
    output_path = Path(output_path or OUTPUT_ROOT / "rmse_combined")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_base = str(output_path.with_suffix(""))

    air_rmse = compute_air_rmse_array(air_pred, air_truth)
    surface_rmse = compute_surface_rmse_array(surface_pred, surface_truth)
    _plot_rmse_combined(air_rmse, surface_rmse, [model_name], save_base)
    return Path(f"{save_base}.png")
