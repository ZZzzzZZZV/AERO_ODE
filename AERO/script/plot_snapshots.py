from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from script.config import OUTPUT_ROOT, PLOT_LEADS, usable_file
from script.plot_maps import (
    FIELD_INFO,
    HRRR_PROJ,
    WRF_MODEL,
    air_channel_index,
    create_colormap,
    lead_available,
    load_coordinates,
    load_wrf_forecast_array,
    maybe_transpose,
    project_grid,
    setup_map_axes,
    to_numpy,
    wrf_field_for_leads,
)
from script.plot_style import (
    CBAR_TICK_LABEL_SIZE,
    MAP_AXES_LABELSIZE,
    MAP_COMP_AXES_LABELSIZE,
    MAP_COMP_CBAR_TICK_SIZE,
    MAP_COMP_GRIDLABEL_SIZE,
    MAP_COMP_TITLE_PAD,
    MAP_COMP_TITLE_SIZE,
    MAP_SUBPLOT_HSPACE,
    MAP_SUBPLOT_TITLE_PAD,
    MAP_SUBPLOT_TITLE_SIZE,
    MAP_SUBPLOT_WSPACE,
    SNAPSHOT_COMP_BOTTOM,
    SNAPSHOT_COMP_DISPLAY_WIDTH,
    SNAPSHOT_COMP_HSPACE,
    SNAPSHOT_COMP_LEFT,
    SNAPSHOT_COMP_RIGHT,
    SNAPSHOT_COMP_TOP,
    SNAPSHOT_COMP_WSPACE,
    SNAPSHOT_ERROR_COLS,
    SNAPSHOT_ERROR_DISPLAY_WIDTH,
    SNAPSHOT_ERROR_WIDTH,
    SNAPSHOT_ROW_HEIGHT,
    apply_paper_rcparams,
    comparison_figsize,
    save_viz_figure,
)

def _valid_leads(leads: List[int], t_len: int) -> List[int]:
    valid = [h for h in leads if 0 < h < t_len]
    if not valid:
        raise ValueError(f"No valid lead times in {leads} for T={t_len}")
    return valid

def _extract_field(pred: torch.Tensor, truth: torch.Tensor, channel: int, lats: np.ndarray):
    pred_field = maybe_transpose(to_numpy(pred)[:, channel], lats)
    truth_field = maybe_transpose(to_numpy(truth)[:, channel], lats)
    return pred_field, truth_field

def _plot_field(
    ax, x, y, data, grid, cmap, vmin, vmax, title: str,
    *,
    title_size: int = MAP_SUBPLOT_TITLE_SIZE,
    title_pad: int = MAP_SUBPLOT_TITLE_PAD,
    tick_size: Optional[int] = None,
):
    setup_map_axes(ax, grid, tick_size=tick_size)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        im = ax.pcolormesh(
            x, y, data, cmap=cmap, vmin=vmin, vmax=vmax,
            shading="nearest", zorder=1.5,
        )
    ax.set_title(title, fontsize=title_size, pad=title_pad)
    return im

def snapshot_path(
    field_key: str,
    kind: str,
    init_date: datetime,
    output_dir: Optional[Path] = None,
) -> Path:
    root = Path(output_dir or OUTPUT_ROOT / "snapshots")
    return root / f"{field_key}_{kind}_{init_date.strftime('%Y%m%d')}.png"

def plot_comparison_snapshots(
    predicted: torch.Tensor,
    truth: torch.Tensor,
    field_key: str,
    init_date: datetime,
    leads: Optional[List[int]] = None,
    output_dir: Optional[Path] = None,
    reuse_existing: bool = True,
    wrf_pred: Optional[np.ndarray] = None,
) -> Path:
    save_path = snapshot_path(field_key, "comparison", init_date, output_dir)
    if reuse_existing and usable_file(save_path):
        return save_path
    apply_paper_rcparams(map_style=True)
    info = FIELD_INFO[field_key]
    leads = leads or PLOT_LEADS
    output_dir = Path(output_dir or OUTPUT_ROOT / "snapshots")
    output_dir.mkdir(parents=True, exist_ok=True)
    lats, lons = load_coordinates()
    channel = air_channel_index("Z", 925) if field_key == "z925" else 3
    pred_field, truth_field = _extract_field(predicted, truth, channel, lats)
    wrf_field = wrf_field_for_leads(wrf_pred, field_key, lats)
    valid = _valid_leads(leads, pred_field.shape[0])
    wrf_leads = [h for h in valid if lead_available(wrf_field, h)]
    vmin = min(pred_field[valid].min(), truth_field[valid].min())
    vmax = max(pred_field[valid].max(), truth_field[valid].max())
    if wrf_leads:
        vmin = min(vmin, float(wrf_field[wrf_leads].min()))
        vmax = max(vmax, float(wrf_field[wrf_leads].max()))
    margin = 0.02 * (vmax - vmin)
    vmin, vmax = vmin - margin, vmax + margin
    cmap = create_colormap(info["cmap"])
    grid = project_grid(lats, lons)
    x, y = grid[0], grid[1]

    n_rows = len(valid)
    n_cols = 3 if wrf_field is not None else 2
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=comparison_figsize(n_rows, n_cols),
        subplot_kw={"projection": HRRR_PROJ},
    )
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    im = None
    for row, lead in enumerate(valid):
        pred_plot, truth_plot = pred_field[lead], truth_field[lead]
        rmse = float(np.sqrt(np.mean((pred_plot - truth_plot) ** 2)))
        im = _plot_field(
            axes[row, 0], x, y, pred_plot, grid, cmap, vmin, vmax,
            f"{info['model']} (+{lead}h) | RMSE: {rmse:.4f}",
            title_size=MAP_COMP_TITLE_SIZE, title_pad=MAP_COMP_TITLE_PAD,
            tick_size=MAP_COMP_GRIDLABEL_SIZE,
        )
        if n_cols == 3:
            if lead_available(wrf_field, lead):
                wrf_plot = wrf_field[lead]
                wrf_rmse = float(np.sqrt(np.mean((wrf_plot - truth_plot) ** 2)))
                _plot_field(
                    axes[row, 1], x, y, wrf_plot, grid, cmap, vmin, vmax,
                    f"{WRF_MODEL} (+{lead}h) | RMSE: {wrf_rmse:.4f}",
                    title_size=MAP_COMP_TITLE_SIZE, title_pad=MAP_COMP_TITLE_PAD,
                    tick_size=MAP_COMP_GRIDLABEL_SIZE,
                )
            else:
                axes[row, 1].set_visible(False)
            hrrr_ax = axes[row, 2]
        else:
            hrrr_ax = axes[row, 1]
        _plot_field(
            hrrr_ax, x, y, truth_plot, grid, cmap, vmin, vmax,
            f"HRRR Analysis (+{lead}h)",
            title_size=MAP_COMP_TITLE_SIZE, title_pad=MAP_COMP_TITLE_PAD,
            tick_size=MAP_COMP_GRIDLABEL_SIZE,
        )

    cbar_ax = fig.add_axes([0.92, 0.05, 0.018, 0.90])
    cbar = plt.colorbar(im, cax=cbar_ax, orientation="vertical", extend="both")
    cbar.set_label(
        f"{info['name']} ({info['unit']})", fontsize=MAP_COMP_AXES_LABELSIZE,
    )
    cbar.ax.tick_params(labelsize=MAP_COMP_CBAR_TICK_SIZE)
    plt.subplots_adjust(
        left=SNAPSHOT_COMP_LEFT, right=SNAPSHOT_COMP_RIGHT,
        top=SNAPSHOT_COMP_TOP, bottom=SNAPSHOT_COMP_BOTTOM,
        hspace=SNAPSHOT_COMP_HSPACE, wspace=SNAPSHOT_COMP_WSPACE,
    )
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
    reuse_existing: bool = True,
    wrf_pred: Optional[np.ndarray] = None,
) -> Path:
    save_path = snapshot_path(field_key, "error", init_date, output_dir)
    if reuse_existing and usable_file(save_path):
        return save_path
    apply_paper_rcparams(map_style=True)
    info = FIELD_INFO[field_key]
    leads = leads or PLOT_LEADS
    output_dir = Path(output_dir or OUTPUT_ROOT / "snapshots")
    output_dir.mkdir(parents=True, exist_ok=True)
    lats, lons = load_coordinates()
    channel = air_channel_index("Z", 925) if field_key == "z925" else 3
    pred_field, truth_field = _extract_field(predicted, truth, channel, lats)
    wrf_field = wrf_field_for_leads(wrf_pred, field_key, lats)
    valid = _valid_leads(leads, pred_field.shape[0])
    aero_errs = [np.abs(pred_field[h] - truth_field[h]) for h in valid]
    wrf_errs = [
        np.abs(wrf_field[h] - truth_field[h]) if lead_available(wrf_field, h) else None
        for h in valid
    ]
    err_stack = list(aero_errs)
    err_stack.extend(err for err in wrf_errs if err is not None)
    err_max = float(np.stack(err_stack).max())
    cmap = create_colormap("rmse")
    grid = project_grid(lats, lons)
    x, y = grid[0], grid[1]

    n_rows = len(valid)
    n_cols = 2 if wrf_field is not None else SNAPSHOT_ERROR_COLS
    if wrf_field is None:
        n_rows = int(np.ceil(len(valid) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(SNAPSHOT_ERROR_WIDTH, SNAPSHOT_ROW_HEIGHT * n_rows),
        subplot_kw={"projection": HRRR_PROJ},
    )
    axes = np.atleast_2d(np.asarray(axes))
    im = None
    if wrf_field is not None:
        for row, lead in enumerate(valid):
            rmse = float(np.sqrt(np.mean((pred_field[lead] - truth_field[lead]) ** 2)))
            im = _plot_field(
                axes[row, 0], x, y, aero_errs[row], grid, cmap, 0, err_max,
                f"{info['model']} +{lead}h (RMSE: {rmse:.4f})",
            )
            if wrf_errs[row] is not None:
                wrf_rmse = float(np.sqrt(np.mean((wrf_field[lead] - truth_field[lead]) ** 2)))
                _plot_field(
                    axes[row, 1], x, y, wrf_errs[row], grid, cmap, 0, err_max,
                    f"{WRF_MODEL} +{lead}h (RMSE: {wrf_rmse:.4f})",
                )
            else:
                axes[row, 1].set_visible(False)
    else:
        for idx, lead in enumerate(valid):
            row, col = divmod(idx, n_cols)
            rmse = float(np.sqrt(np.mean((pred_field[lead] - truth_field[lead]) ** 2)))
            im = _plot_field(
                axes[row, col], x, y, aero_errs[idx], grid, cmap, 0, err_max,
                f"{info['model']} +{lead}h (RMSE: {rmse:.4f})",
            )
        for idx in range(len(valid), n_rows * n_cols):
            row, col = divmod(idx, n_cols)
            axes[row, col].set_visible(False)

    cbar_ax = fig.add_axes([0.92, 0.05, 0.018, 0.90])
    cbar = plt.colorbar(im, cax=cbar_ax, orientation="vertical", extend="max")
    cbar.set_label(
        f"Absolute Error ({info['unit']})", fontsize=MAP_AXES_LABELSIZE,
    )
    cbar.ax.tick_params(labelsize=CBAR_TICK_LABEL_SIZE)
    plt.subplots_adjust(
        left=0.03, right=0.90, top=0.98, bottom=0.04,
        hspace=MAP_SUBPLOT_HSPACE, wspace=MAP_SUBPLOT_WSPACE,
    )
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
    reuse_existing: bool = True,
) -> List[Path]:
    init_date = init_date or datetime(2024, 1, 1)
    cached = reuse_existing and all(
        usable_file(snapshot_path(key, kind, init_date, output_dir))
        for key in ("z925", "t2m")
        for kind in ("comparison", "error")
    )
    wrf_pred = None if cached else load_wrf_forecast_array(init_date)
    saved = []
    for field_key, pred, truth in (
        ("z925", air_pred, air_truth),
        ("t2m", surface_pred, surface_truth),
    ):
        saved.append(
            plot_comparison_snapshots(
                pred, truth, field_key, init_date, leads, output_dir,
                reuse_existing, wrf_pred=wrf_pred,
            )
        )
        saved.append(
            plot_error_snapshots(
                pred, truth, field_key, init_date, leads, output_dir,
                reuse_existing, wrf_pred=wrf_pred,
            )
        )
    return saved

def display_snapshots(paths: List[Path], width: Optional[int] = None) -> None:
    from IPython.display import Image, display

    for path in paths:
        if width is not None:
            disp_w = width
        elif "error" in Path(path).stem:
            disp_w = SNAPSHOT_ERROR_DISPLAY_WIDTH
        else:
            disp_w = SNAPSHOT_COMP_DISPLAY_WIDTH
        display(Image(filename=str(path), width=disp_w))
