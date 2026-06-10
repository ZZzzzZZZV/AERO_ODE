"""Lead-time snapshot maps: forecast vs HRRR and absolute-error panels."""

from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import torch

from script.config import OUTPUT_ROOT, PLOT_LEADS
from script.plot_animation import (
    FIELD_INFO,
    HRRR_PROJ,
    _create_colormap,
    _load_coordinates,
    _maybe_transpose,
    _setup_map_axes,
    _to_numpy,
    air_channel_index,
)
from script.plot_style import (
    CBAR_TICK_LABEL_SIZE,
    MAP_AXES_LABELSIZE,
    MAP_SUBPLOT_HSPACE,
    MAP_SUBPLOT_TITLE_PAD,
    MAP_SUBPLOT_TITLE_SIZE,
    MAP_SUBPLOT_WSPACE,
    SNAPSHOT_COMP_WIDTH,
    SNAPSHOT_ERROR_WIDTH,
    SNAPSHOT_ROW_HEIGHT,
    apply_paper_rcparams,
    save_viz_figure,
)


def _setup_style() -> None:
    apply_paper_rcparams(map_style=True)


def _project_grid(lats: np.ndarray, lons: np.ndarray):
    proj_coords = HRRR_PROJ.transform_points(ccrs.PlateCarree(), lons, lats)
    x = proj_coords[:, :, 0]
    y = proj_coords[:, :, 1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    margin = 0.20
    extent_x = (x_max - x_min) * (1 + margin)
    extent_y = (y_max - y_min) * (1 + margin)
    return x, y, x_min, x_max, y_min, y_max, x_center, y_center, extent_x, extent_y


def _plot_field(ax, x, y, data, grid, cmap, vmin, vmax, title: str):
    x_min, x_max, y_min, y_max, x_center, y_center, extent_x, extent_y = grid
    _setup_map_axes(ax, x, y, x_min, x_max, y_min, y_max, x_center, y_center, extent_x, extent_y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        im = ax.pcolormesh(
            x, y, data, cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest", zorder=1.5,
        )
    ax.set_title(title, fontsize=MAP_SUBPLOT_TITLE_SIZE, pad=MAP_SUBPLOT_TITLE_PAD)
    return im


def _extract_field(
    pred: torch.Tensor,
    truth: torch.Tensor,
    channel: int,
    lats: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pred_np = _to_numpy(pred)
    truth_np = _to_numpy(truth)
    pred_field = _maybe_transpose(pred_np[:, channel, :, :], lats)
    truth_field = _maybe_transpose(truth_np[:, channel, :, :], lats)
    return pred_field, truth_field


def _valid_leads(leads: List[int], t_len: int) -> List[int]:
    valid = [h for h in leads if 0 < h < t_len]
    if not valid:
        raise ValueError(f"No valid lead times in {leads} for T={t_len}")
    return valid


def plot_comparison_snapshots(
    predicted: torch.Tensor,
    truth: torch.Tensor,
    field_key: str,
    init_date: datetime,
    leads: Optional[List[int]] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    """Rows = lead times; columns = [Model | HRRR]."""
    _setup_style()
    info = FIELD_INFO[field_key]
    leads = leads or PLOT_LEADS
    output_dir = Path(output_dir or OUTPUT_ROOT / "snapshots")
    output_dir.mkdir(parents=True, exist_ok=True)

    lats, lons = _load_coordinates()
    channel = air_channel_index("Z", 1000) if field_key == "z1000" else 3
    pred_field, truth_field = _extract_field(predicted, truth, channel, lats)
    valid = _valid_leads(leads, pred_field.shape[0])

    stacks_pred = np.stack([pred_field[h] for h in valid])
    stacks_truth = np.stack([truth_field[h] for h in valid])
    vmin = min(stacks_pred.min(), stacks_truth.min())
    vmax = max(stacks_pred.max(), stacks_truth.max())
    margin = 0.02 * (vmax - vmin)
    vmin -= margin
    vmax += margin
    cmap = _create_colormap(info["cmap"])

    grid = _project_grid(lats, lons)
    x, y = grid[0], grid[1]
    n_rows = len(valid)
    fig, axes = plt.subplots(
        n_rows, 2,
        figsize=(SNAPSHOT_COMP_WIDTH, SNAPSHOT_ROW_HEIGHT * n_rows),
        subplot_kw={"projection": HRRR_PROJ},
    )
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    im = None
    model_name = info["model"]
    for row, lead in enumerate(valid):
        pred_plot = pred_field[lead]
        truth_plot = truth_field[lead]
        rmse = float(np.sqrt(np.mean((pred_plot - truth_plot) ** 2)))

        im = _plot_field(
            axes[row, 0], x, y, pred_plot, grid[2:], cmap, vmin, vmax,
            f"{model_name} (+{lead}h) | RMSE: {rmse:.4f}",
        )
        _plot_field(
            axes[row, 1], x, y, truth_plot, grid[2:], cmap, vmin, vmax,
            f"HRRR Analysis (+{lead}h)",
        )

    cbar_ax = fig.add_axes([0.92, 0.05, 0.018, 0.90])
    cbar = plt.colorbar(im, cax=cbar_ax, orientation="vertical", extend="both")
    cbar.set_label(f"{info['name']} ({info['unit']})", fontsize=MAP_AXES_LABELSIZE)
    cbar.ax.tick_params(labelsize=CBAR_TICK_LABEL_SIZE)
    plt.subplots_adjust(
        left=0.03, right=0.90, top=0.98, bottom=0.04,
        hspace=MAP_SUBPLOT_HSPACE, wspace=MAP_SUBPLOT_WSPACE,
    )

    save_path = output_dir / f"{field_key}_comparison_{init_date.strftime('%Y%m%d')}.png"
    save_viz_figure(fig, save_path)
    plt.close(fig)
    return save_path


def plot_error_snapshots(
    predicted: torch.Tensor,
    truth: torch.Tensor,
    field_key: str,
    init_date: datetime,
    leads: Optional[List[int]] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    """Rows = lead times; one column of absolute error vs HRRR."""
    _setup_style()
    info = FIELD_INFO[field_key]
    leads = leads or PLOT_LEADS
    output_dir = Path(output_dir or OUTPUT_ROOT / "snapshots")
    output_dir.mkdir(parents=True, exist_ok=True)

    lats, lons = _load_coordinates()
    channel = air_channel_index("Z", 1000) if field_key == "z1000" else 3
    pred_field, truth_field = _extract_field(predicted, truth, channel, lats)
    valid = _valid_leads(leads, pred_field.shape[0])

    err_maps = []
    rmse_values = []
    for lead in valid:
        err = np.abs(pred_field[lead] - truth_field[lead])
        err_maps.append(err)
        rmse_values.append(float(np.sqrt(np.mean((pred_field[lead] - truth_field[lead]) ** 2))))

    err_max = np.stack(err_maps).max()
    cmap = _create_colormap("rmse")

    grid = _project_grid(lats, lons)
    x, y = grid[0], grid[1]
    n_rows = len(valid)
    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=(SNAPSHOT_ERROR_WIDTH, SNAPSHOT_ROW_HEIGHT * n_rows),
        subplot_kw={"projection": HRRR_PROJ},
    )
    if n_rows == 1:
        axes = np.array([axes])

    im = None
    model_name = info["model"]
    for row, lead in enumerate(valid):
        im = _plot_field(
            axes[row], x, y, err_maps[row], grid[2:], cmap, 0, err_max,
            f"{model_name} +{lead}h (RMSE: {rmse_values[row]:.4f})",
        )

    cbar_ax = fig.add_axes([0.92, 0.05, 0.018, 0.90])
    cbar = plt.colorbar(im, cax=cbar_ax, orientation="vertical", extend="max")
    cbar.set_label(f"Absolute Error ({info['unit']})", fontsize=MAP_AXES_LABELSIZE)
    cbar.ax.tick_params(labelsize=CBAR_TICK_LABEL_SIZE)
    plt.subplots_adjust(
        left=0.05, right=0.90, top=0.98, bottom=0.04,
        hspace=MAP_SUBPLOT_HSPACE, wspace=MAP_SUBPLOT_WSPACE,
    )

    save_path = output_dir / f"{field_key}_error_{init_date.strftime('%Y%m%d')}.png"
    save_viz_figure(fig, save_path)
    plt.close(fig)
    return save_path


def plot_demo_snapshots(
    air_pred: torch.Tensor,
    air_truth: torch.Tensor,
    surface_pred: torch.Tensor,
    surface_truth: torch.Tensor,
    init_date: Optional[datetime] = None,
    leads: Optional[List[int]] = None,
    output_dir: Optional[Path] = None,
) -> List[Path]:
    """Z1000 (AERO-AIR) and T2M (AERO-Surface): comparison + error maps at selected leads."""
    init_date = init_date or datetime(2024, 8, 1)
    leads = leads or PLOT_LEADS
    saved: List[Path] = []

    for field_key, pred, truth in (
        ("z1000", air_pred, air_truth),
        ("t2m", surface_pred, surface_truth),
    ):
        saved.append(
            plot_comparison_snapshots(
                pred, truth, field_key, init_date, leads=leads, output_dir=output_dir,
            )
        )
        saved.append(
            plot_error_snapshots(
                pred, truth, field_key, init_date, leads=leads, output_dir=output_dir,
            )
        )
    return saved
