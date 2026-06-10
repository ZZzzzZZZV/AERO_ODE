"""72 h AERO-ODE field animations (AERO-AIR upper-air + AERO-Surface)."""

from __future__ import annotations

import os
import subprocess
import tempfile
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from matplotlib.patches import Rectangle

import paths_config as pc
from script.config import FORECAST_HOURS, OUTPUT_ROOT
from script.plot_style import (
    MAP_DATA_BORDER_WIDTH,
    MAP_LON_LABEL_PAD,
    MAP_SPINE_WIDTH,
    MAP_SUBPLOT_TITLE_PAD,
    MAP_SUBPLOT_TITLE_SIZE,
    MAP_TICK_LABEL_SIZE,
)

HRRR_PROJ = ccrs.LambertConformal(
    central_longitude=-97.5,
    central_latitude=38.5,
    standard_parallels=(38.5, 38.5),
    globe=ccrs.Globe(semimajor_axis=6371229.0, semiminor_axis=6371229.0),
)

UPPER_LEVELS = (50, 500, 850, 1000)
UPPER_VAR_INDEX = {"Z": 0, "T": 1, "S": 2, "U": 3, "V": 4}
UPPER_VAR_UNITS = {"Z": "gpm", "T": "K", "S": "kg/kg", "U": "m/s", "V": "m/s"}

FIELD_INFO = {
    "z1000": {
        "name": "Geopotential Height (1000 hPa)",
        "short": "Z1000",
        "unit": "gpm",
        "cmap": "height",
        "model": "AERO-AIR",
    },
    "t2m": {
        "name": "2m Temperature",
        "short": "T2M",
        "unit": "K",
        "cmap": "coolwarm",
        "model": "AERO-Surface",
    },
}


def air_channel_index(var: str, level: int) -> int:
    if var not in UPPER_VAR_INDEX:
        raise ValueError(f"Unknown upper-air variable: {var}")
    if level not in UPPER_LEVELS:
        raise ValueError(f"Unknown level: {level}, expected one of {UPPER_LEVELS}")
    return UPPER_VAR_INDEX[var] * len(UPPER_LEVELS) + UPPER_LEVELS.index(level)


def _to_numpy(data: torch.Tensor) -> np.ndarray:
    arr = data.detach().cpu().numpy()
    if arr.ndim == 5:
        arr = arr[0]
    if arr.ndim != 4:
        raise ValueError(f"Expected tensor (1,T,C,H,W), got shape {data.shape}")
    return arr


def _load_coordinates() -> tuple[np.ndarray, np.ndarray]:
    stat_root = pc.ensure_exists(pc.hrrr_stat_root(), "HRRR stat root")
    lats = np.load(os.fspath(stat_root / "lats.npy"))
    lons = np.load(os.fspath(stat_root / "lons.npy"))
    return lats, lons


def _create_colormap(cmap_type: str):
    if cmap_type == "coolwarm":
        colors = [
            "#2c1e96", "#3552c2", "#4575b4", "#5c91c2", "#74add1",
            "#abd9e9", "#d4eef7", "#ffffbf", "#fee090", "#fdae61",
            "#f46d43", "#d73027", "#b2182b", "#8c0d25",
        ]
        return mcolors.LinearSegmentedColormap.from_list("scientific", colors, N=256)
    if cmap_type == "height":
        colors = [
            "#313695", "#4575b4", "#74add1", "#abd9e9", "#e0f3f8",
            "#ffffbf", "#fee090", "#fdae61", "#f46d43", "#d73027", "#a50026",
        ]
        return mcolors.LinearSegmentedColormap.from_list("height", colors, N=256)
    if cmap_type == "rmse":
        colors = [
            "#ffffff", "#fff5f0", "#fee0d2", "#fcbba1", "#fc9272",
            "#fb6a4a", "#ef3b2c", "#cb181d", "#a50f15", "#67000d",
        ]
        return mcolors.LinearSegmentedColormap.from_list("rmse", colors, N=256)
    return plt.get_cmap(cmap_type)


def _setup_map_axes(ax, x, y, x_min, x_max, y_min, y_max, x_center, y_center, extent_x, extent_y):
    ax.set_xlim(x_center - extent_x / 2, x_center + extent_x / 2)
    ax.set_ylim(y_center - extent_y / 2, y_center + extent_y / 2)
    for spine in ax.spines.values():
        spine.set_linewidth(MAP_SPINE_WIDTH)

    ax.add_feature(cfeature.LAND, facecolor="#f8f8f8", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="#e8f4fc", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.LAKES, facecolor="#cce5ff", edgecolor="#888888", linewidth=0.5, zorder=1)
    ax.add_feature(cfeature.COASTLINE, edgecolor="#444444", linewidth=0.8, zorder=3)
    ax.add_feature(cfeature.BORDERS, edgecolor="#777777", linewidth=0.5, linestyle="--", zorder=2)
    ax.add_feature(cfeature.STATES, edgecolor="#aaaaaa", linewidth=0.3, zorder=2)

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5, color="gray",
        alpha=0.6, linestyle=":", x_inline=False, y_inline=False, zorder=2,
    )
    gl.xlocator = mticker.FixedLocator(np.arange(-105, -75, 5))
    gl.ylocator = mticker.FixedLocator(np.arange(30, 50, 5))
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"size": MAP_TICK_LABEL_SIZE, "color": "#333333", "rotation": 0, "va": "top"}
    gl.ylabel_style = {"size": MAP_TICK_LABEL_SIZE, "color": "#333333"}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    rect = Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        fill=False, edgecolor="#333333", linewidth=MAP_DATA_BORDER_WIDTH, zorder=4,
    )
    ax.add_patch(rect)


def _maybe_transpose(field: np.ndarray, lats: np.ndarray) -> np.ndarray:
    """Align spatial dims (H, W) with lats; supports (T, H, W) or (H, W)."""
    hw = field.shape[-2:]
    if hw == lats.shape:
        return field
    if hw == lats.T.shape:
        if field.ndim == 2:
            return field.T
        return field.transpose(0, 2, 1)
    raise ValueError(f"Field spatial shape {hw} does not match grid {lats.shape}")


def _frame_title(model_title: str, hour: int, rmse: Optional[float], show_rmse: bool) -> str:
    if show_rmse and rmse is not None:
        return f"{model_title} (+{hour:03d}h)\nRMSE: {rmse:.4f}"
    return f"{model_title} (+{hour:03d}h)"


def _save_animation(
    fig, anim, output_path: Path, fps: int, total_frames: int,
    field: np.ndarray, vmin: float, vmax: float,
    rmse_values: List[Optional[float]], start_hour: int, show_rmse: bool,
    model_title: str,
):
    mp4_path = output_path.with_suffix(".mp4")
    try:
        writer = animation.FFMpegWriter(
            fps=fps, codec="libx264", bitrate=8000, extra_args=["-pix_fmt", "yuv420p"],
        )
        anim.interval = 1000 / fps
        anim.save(str(mp4_path), writer=writer, dpi=120)
    except Exception:
        with tempfile.TemporaryDirectory() as tmpdir:
            ax = fig.axes[0]
            im = ax.collections[0]
            for t_idx in range(total_frames):
                hour = start_hour + t_idx
                im.set_array(field[t_idx].ravel())
                im.set_clim(vmin, vmax)
                ax.set_title(
                    _frame_title(model_title, hour, rmse_values[t_idx], show_rmse),
                    fontsize=MAP_SUBPLOT_TITLE_SIZE, pad=MAP_SUBPLOT_TITLE_PAD,
                )
                fig.savefig(
                    f"{tmpdir}/frame_{t_idx:04d}.png",
                    dpi=120, bbox_inches="tight", facecolor="white", edgecolor="none",
                )
            subprocess.run([
                "ffmpeg", "-y", "-framerate", str(fps),
                "-i", f"{tmpdir}/frame_%04d.png",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(mp4_path),
            ], check=True, capture_output=True)
    plt.close(fig)
    return mp4_path


def _create_animation(
    field: np.ndarray,
    truth_field: Optional[np.ndarray],
    lats: np.ndarray,
    lons: np.ndarray,
    field_key: str,
    init_date: datetime,
    output_dir: Path,
    fps: int,
    start_hour: int,
    end_hour: int,
    show_rmse: bool,
) -> Path:
    info = FIELD_INFO[field_key]
    model_title = info["model"]
    cmap = _create_colormap(info["cmap"])

    start_idx = start_hour
    end_idx = min(end_hour + 1, field.shape[0])
    frames = field[start_idx:end_idx]
    truth_frames = None if truth_field is None else truth_field[start_idx:end_idx]
    total_frames = frames.shape[0]
    if total_frames < 1:
        raise ValueError(f"{field_key}: no frames in +{start_hour}h ~ +{end_hour}h")

    vmin = float(np.nanmin(frames))
    vmax = float(np.nanmax(frames))
    margin = 0.02 * (vmax - vmin)
    vmin -= margin
    vmax += margin

    rmse_values: List[Optional[float]] = []
    if show_rmse and truth_frames is not None:
        for t_idx in range(total_frames):
            rmse_values.append(float(np.sqrt(np.mean((frames[t_idx] - truth_frames[t_idx]) ** 2))))
    else:
        rmse_values = [None] * total_frames

    proj_coords = HRRR_PROJ.transform_points(ccrs.PlateCarree(), lons, lats)
    x, y = proj_coords[:, :, 0], proj_coords[:, :, 1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2
    extent_x = (x_max - x_min) * 1.25
    extent_y = (y_max - y_min) * 1.25

    fig = plt.figure(figsize=(10, 8), dpi=120)
    ax = fig.add_subplot(1, 1, 1, projection=HRRR_PROJ)
    _setup_map_axes(ax, x, y, x_min, x_max, y_min, y_max, x_center, y_center, extent_x, extent_y)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        im = ax.pcolormesh(x, y, frames[0], cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest", zorder=1.5)

    ax.set_title(
        _frame_title(model_title, start_hour, rmse_values[0], show_rmse),
        fontsize=MAP_SUBPLOT_TITLE_SIZE, pad=MAP_SUBPLOT_TITLE_PAD,
    )
    fig.suptitle(
        f"{info['name']} | Init: {init_date.strftime('%Y-%m-%d')} 00:00 UTC",
        fontsize=16, fontweight="bold", y=0.98,
    )

    cbar_ax = fig.add_axes([0.15, 0.06, 0.7, 0.025])
    cbar = plt.colorbar(im, cax=cbar_ax, orientation="horizontal", extend="both")
    cbar.set_label(f"{info['name']} ({info['unit']})", fontsize=12)
    plt.subplots_adjust(left=0.03, right=0.97, top=0.88, bottom=0.15)

    def update(frame):
        hour = start_hour + frame
        im.set_array(frames[frame].ravel())
        im.set_clim(vmin, vmax)
        ax.set_title(
            _frame_title(model_title, hour, rmse_values[frame], show_rmse),
            fontsize=MAP_SUBPLOT_TITLE_SIZE, pad=MAP_SUBPLOT_TITLE_PAD,
        )
        return [im]

    anim = animation.FuncAnimation(fig, update, frames=total_frames, interval=1000 / fps, blit=False)
    out_stem = output_dir / f"{info['model'].lower().replace('-', '_')}_{info['short'].lower()}_{init_date.strftime('%Y%m%d')}"
    return _save_animation(
        fig, anim, out_stem, fps, total_frames, frames, vmin, vmax,
        rmse_values, start_hour, show_rmse, model_title,
    )


def animate_air_z1000(
    predicted: torch.Tensor,
    truth: Optional[torch.Tensor] = None,
    init_date: Optional[datetime] = None,
    fps: int = 8,
    start_hour: int = 1,
    end_hour: int = FORECAST_HOURS,
    show_rmse: bool = True,
    output_dir: Optional[Path] = None,
) -> Path:
    """72 h animation for AERO-AIR Z at 1000 hPa."""
    init_date = init_date or datetime(2024, 8, 1)
    output_dir = Path(output_dir or OUTPUT_ROOT / "animations")
    output_dir.mkdir(parents=True, exist_ok=True)

    ch = air_channel_index("Z", 1000)
    pred_np = _to_numpy(predicted)
    truth_np = None if truth is None else _to_numpy(truth)
    lats, lons = _load_coordinates()

    field = _maybe_transpose(pred_np[:, ch, :, :], lats)
    truth_field = None if truth_np is None else _maybe_transpose(truth_np[:, ch, :, :], lats)
    return _create_animation(
        field, truth_field, lats, lons, "z1000", init_date, output_dir,
        fps=fps, start_hour=start_hour, end_hour=end_hour, show_rmse=show_rmse,
    )


def animate_surface_t2m(
    predicted: torch.Tensor,
    truth: Optional[torch.Tensor] = None,
    init_date: Optional[datetime] = None,
    fps: int = 8,
    start_hour: int = 1,
    end_hour: int = FORECAST_HOURS,
    show_rmse: bool = True,
    output_dir: Optional[Path] = None,
) -> Path:
    """72 h animation for AERO-Surface 2m temperature."""
    init_date = init_date or datetime(2024, 8, 1)
    output_dir = Path(output_dir or OUTPUT_ROOT / "animations")
    output_dir.mkdir(parents=True, exist_ok=True)

    ch = 3  # T2M
    pred_np = _to_numpy(predicted)
    truth_np = None if truth is None else _to_numpy(truth)
    lats, lons = _load_coordinates()

    field = _maybe_transpose(pred_np[:, ch, :, :], lats)
    truth_field = None if truth_np is None else _maybe_transpose(truth_np[:, ch, :, :], lats)
    return _create_animation(
        field, truth_field, lats, lons, "t2m", init_date, output_dir,
        fps=fps, start_hour=start_hour, end_hour=end_hour, show_rmse=show_rmse,
    )


def animate_demo_fields(
    air_pred: torch.Tensor,
    air_truth: torch.Tensor,
    surface_pred: torch.Tensor,
    surface_truth: torch.Tensor,
    init_date: Optional[datetime] = None,
    fps: int = 8,
    start_hour: int = 1,
    end_hour: int = FORECAST_HOURS,
    show_rmse: bool = True,
    output_dir: Optional[Path] = None,
) -> List[Path]:
    """Build AERO-AIR Z1000 and AERO-Surface T2M animations."""
    init_date = init_date or datetime(2024, 8, 1)
    kwargs = dict(
        init_date=init_date, fps=fps, start_hour=start_hour,
        end_hour=end_hour, show_rmse=show_rmse, output_dir=output_dir,
    )
    return [
        animate_air_z1000(air_pred, air_truth, **kwargs),
        animate_surface_t2m(surface_pred, surface_truth, **kwargs),
    ]


def display_animation(video_path: Path | str, width: int = 720) -> None:
    """Show MP4 inline in Jupyter / VS Code / Cursor notebooks."""
    import base64

    from IPython.display import HTML, Video, display

    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(path)

    try:
        display(Video(str(path), embed=True, width=width, html_attributes="controls loop"))
        return
    except Exception:
        pass

    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    display(HTML(
        f'<video width="{width}" controls loop>'
        f'<source src="data:video/mp4;base64,{b64}" type="video/mp4">'
        f"</video>"
    ))
