from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from matplotlib.patches import Rectangle

import paths_config as pc
from script.config import AIR_LEVELS
from script.plot_style import (
    MAP_DATA_BORDER_WIDTH,
    MAP_SPINE_WIDTH,
    MAP_TICK_LABEL_SIZE,
)

HRRR_PROJ = ccrs.LambertConformal(
    central_longitude=-97.5,
    central_latitude=38.5,
    standard_parallels=(38.5, 38.5),
    globe=ccrs.Globe(semimajor_axis=6371229.0, semiminor_axis=6371229.0),
)

UPPER_VAR_INDEX = {"Z": 0, "T": 1, "S": 2, "U": 3, "V": 4}
FIELD_INFO = {
    "z925": {
        "name": "Geopotential Height (925 hPa)",
        "short": "Z925",
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

WRF_MODEL = "WRF-ARW"
WRF_CHANNELS = {"z925": 4, "t2m": 38}

def air_channel_index(var: str, level: int) -> int:
    if var not in UPPER_VAR_INDEX:
        raise ValueError(f"Unknown upper-air variable: {var}")
    if level not in AIR_LEVELS:
        raise ValueError(f"Unknown level: {level}, expected one of {AIR_LEVELS}")
    return UPPER_VAR_INDEX[var] * len(AIR_LEVELS) + AIR_LEVELS.index(level)

def to_numpy(data: torch.Tensor) -> np.ndarray:
    arr = data.detach().cpu().numpy()
    if arr.ndim == 5:
        arr = arr[0]
    if arr.ndim != 4:
        raise ValueError(f"Expected tensor (1,T,C,H,W), got shape {data.shape}")
    return arr

def load_coordinates() -> tuple[np.ndarray, np.ndarray]:
    geo = pc.ensure_exists(pc.air_geo_root() / "data", "west grid")
    lats = np.load(os.fspath(geo / "hrrr_west_lat.npy"))
    lons = np.load(os.fspath(geo / "hrrr_west_lon.npy"))
    lons = np.where(lons > 180.0, lons - 360.0, lons)
    return lats, lons

def create_colormap(cmap_type: str):
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

def project_grid(lats: np.ndarray, lons: np.ndarray, extent_scale: float = 1.20):
    points = HRRR_PROJ.transform_points(ccrs.PlateCarree(), lons, lats)
    x, y = points[:, :, 0], points[:, :, 1]
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2
    extent_x = (x_max - x_min) * extent_scale
    extent_y = (y_max - y_min) * extent_scale
    return x, y, x_min, x_max, y_min, y_max, x_center, y_center, extent_x, extent_y

def setup_map_axes(ax, grid, *, tick_size: Optional[int] = None) -> None:
    x, y, x_min, x_max, y_min, y_max, x_center, y_center, extent_x, extent_y = grid
    label_size = MAP_TICK_LABEL_SIZE if tick_size is None else tick_size
    ax.set_xlim(x_center - extent_x / 2, x_center + extent_x / 2)
    ax.set_ylim(y_center - extent_y / 2, y_center + extent_y / 2)
    for spine in ax.spines.values():
        spine.set_linewidth(MAP_SPINE_WIDTH)

    ax.add_feature(cfeature.LAND, facecolor="#f8f8f8", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="#e8f4fc", edgecolor="none", zorder=0)
    ax.add_feature(
        cfeature.LAKES, facecolor="#cce5ff", edgecolor="#888888",
        linewidth=0.5, zorder=1,
    )
    ax.add_feature(cfeature.COASTLINE, edgecolor="#444444", linewidth=0.8, zorder=3)
    ax.add_feature(
        cfeature.BORDERS, edgecolor="#777777", linewidth=0.5,
        linestyle="--", zorder=2,
    )
    ax.add_feature(cfeature.STATES, edgecolor="#aaaaaa", linewidth=0.3, zorder=2)

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.5,
        color="gray", alpha=0.6, linestyle=":", x_inline=False,
        y_inline=False, zorder=2,
    )
    gl.xlocator = mticker.FixedLocator(np.arange(-120, -85, 5))
    gl.ylocator = mticker.FixedLocator(np.arange(25, 50, 5))
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {
        "size": label_size, "color": "#333333",
        "rotation": 0, "va": "top",
    }
    gl.ylabel_style = {"size": label_size, "color": "#333333"}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    ax.add_patch(
        Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            fill=False, edgecolor="#333333",
            linewidth=MAP_DATA_BORDER_WIDTH, zorder=4,
        )
    )

def maybe_transpose(field: np.ndarray, lats: np.ndarray) -> np.ndarray:
    hw = field.shape[-2:]
    if hw == lats.shape:
        return field
    if hw == lats.T.shape:
        return field.T if field.ndim == 2 else field.transpose(0, 2, 1)
    raise ValueError(f"Field spatial shape {hw} does not match grid {lats.shape}")

def load_wrf_forecast_array(init_date: datetime) -> Optional[np.ndarray]:
    from script.calc_scores import load_wrf_forecast

    wrf = load_wrf_forecast(pc.hrrr_forecast_root(), init_date.date())
    if wrf is None:
        print("[WARN] WRF forecast missing; plotting AERO vs HRRR only", flush=True)
        return None
    return wrf

def wrf_field_for_leads(
    wrf: Optional[np.ndarray],
    field_key: str,
    lats: np.ndarray,
) -> Optional[np.ndarray]:
    if wrf is None:
        return None
    if field_key not in WRF_CHANNELS:
        raise ValueError(f"unknown WRF field: {field_key}")
    raw = np.asarray(wrf[:, WRF_CHANNELS[field_key]])
    padded = np.full((raw.shape[0] + 1, *raw.shape[1:]), np.nan, dtype=np.float64)
    padded[1:] = raw
    return maybe_transpose(padded, lats)

def lead_available(field: Optional[np.ndarray], lead: int) -> bool:
    return (
        field is not None
        and 0 <= lead < field.shape[0]
        and np.isfinite(field[lead]).any()
    )
