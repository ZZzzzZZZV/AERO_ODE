"""
Plot AERO-ODE RMSE by UTC init hour (upper-air + surface in one 6x4 figure).

- Colored thin lines per init hour; red bold mean
- Light shaded band between inits and mean

Usage:
    python Draw_inithour_rmse_air_surface.py              # default combined 6x4
    python Draw_inithour_rmse_air_surface.py --kind air   # upper-air 5x4 only
    python Draw_inithour_rmse_air_surface.py --kind surface
    # Override paths: --data-dir / --save-dir (single kind only)
"""
from __future__ import annotations

from pathlib import Path

# =============================================================================
# Path configuration (edit here; upper-air and surface are separate)
# =============================================================================
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent  # AERO_ODE_Draw

# Upper-air / surface RMSE npy (local repo first, then NFS)
_DATA_AIR_LOCAL = _REPO_ROOT / "Rb_AIR_Test_Data_Rmse_by_inithour"
_DATA_SURFACE_LOCAL = _REPO_ROOT / "Rb_Surface_Test_Data_Rmse_by_inithour"
_DATA_AIR_NFS = Path(
    "/nfs/gpu_homes/gpu09/home/zhangjing09/Code/AERO_AIR/Test_Data_Rmse_by_inithour"
)
_DATA_SURFACE_NFS = Path(
    "/nfs/gpu_homes/gpu09/home/zhangjing09/Code/AERO_Surface/Test_Surface_Rmse_by_inithour"
)


def _pick_data_dir(local: Path, nfs: Path) -> Path:
    if local.is_dir() and any(local.glob("rmse_*_inithour_*.npy")):
        return local
    if nfs.is_dir() and any(nfs.glob("rmse_*_inithour_*.npy")):
        return nfs
    return local


DATA_DIR_AIR = _pick_data_dir(_DATA_AIR_LOCAL, _DATA_AIR_NFS)
DATA_DIR_SURFACE = _pick_data_dir(_DATA_SURFACE_LOCAL, _DATA_SURFACE_NFS)


# Output figure directory
FIG_SAVE_DIR = _SCRIPT_DIR / "figures_inithour"
# =============================================================================

import argparse
import os
import sys

import matplotlib.colors as mcolors
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

_DRAW_NEW = Path(__file__).resolve().parent
for _ in range(4):
    if (_DRAW_NEW / "paths_config.py").exists():
        break
    _DRAW_NEW = _DRAW_NEW.parent
else:
    raise RuntimeError("paths_config.py not found")
if str(_DRAW_NEW) not in sys.path:
    sys.path.insert(0, str(_DRAW_NEW))

from paths_config import (  # noqa: E402
    AIR_FIGSIZE,
    AIR_GRID_COLS,
    AIR_GRID_ROWS,
    AXES_LABEL_SIZE,
    COMBINED_RMSE_FIGSIZE,
    SUBPLOT_TITLE_SIZE,
    FIGURE_SAVE_DPI_HIGH,
    SURFACE_FIGSIZE,
    SURFACE_VAR_SHORT,
    UPPER_LEVELS,
    UPPER_VAR_SHORT,
    XLABEL_FORECAST_TIME,
    add_figure_legend_below,
    apply_curve_grid_spacing,
    apply_paper_rcparams,
    font_times_new_roman,
    rmse_air_ylabel,
    save_curve_figure,
    upper_air_subplot_title,
)

FONT_PATH = str(font_times_new_roman())
MODEL_TAG = "aero_ode"
N_AIR_CHANNELS = 20
N_SURFACE_VARS = 4

# Init-hour curve colors
INIT_HOUR_COLORS = {
    0: "#0072B2",
    6: "#009E73",
    12: "#E69F00",
    18: "#CC79A7",
}
FALLBACK_INIT_COLORS = ("#0072B2", "#009E73", "#E69F00", "#CC79A7", "#56B4E9", "#D55E00")

MEAN_COLOR_BASE = "#C62828"
MEAN_SATURATION_SCALE = 0.75
MEAN_LIGHTNESS_ADD = 0.0
MEAN_LINEWIDTH = 2.6
INIT_LINEWIDTH = 1.4
# Shaded band: desaturated, semi-transparent
SHADE_SATURATION_SCALE = 0.32
SHADE_LIGHTNESS_ADD = 0.28
SHADE_ALPHA = 0.18


def _adjust_color(hex_color: str, saturation_scale: float, lightness_add: float) -> str:
    r, g, b = mcolors.to_rgb(hex_color)
    h, s, v = mcolors.rgb_to_hsv((r, g, b))
    s *= saturation_scale
    v = min(1.0, v + lightness_add)
    return mcolors.to_hex(mcolors.hsv_to_rgb((h, s, v)))


MEAN_COLOR = _adjust_color(MEAN_COLOR_BASE, MEAN_SATURATION_SCALE, MEAN_LIGHTNESS_ADD)


def _shade_fill_color(hex_color: str) -> str:
    """Convert line color to lighter desaturated fill."""
    return _adjust_color(hex_color, SHADE_SATURATION_SCALE, SHADE_LIGHTNESS_ADD)


def setup_font(font_path: str = FONT_PATH) -> None:
    try:
        fm.fontManager.addfont(font_path)
        font_name = fm.FontProperties(fname=font_path).get_name()
        plt.rcParams["font.family"] = font_name
        print(f"Loaded font: {font_name}")
    except Exception as e:
        print(f"Warning: failed to load font {font_path}, using default font. Error: {e}")
    apply_paper_rcparams()


def resolve_default_data_dir(kind: str) -> Path:
    """Default to DATA_DIR_AIR / DATA_DIR_SURFACE at top of file."""
    if kind == "air":
        return DATA_DIR_AIR
    if kind == "surface":
        return DATA_DIR_SURFACE
    raise ValueError(f"Unknown kind: {kind}")


def load_inithour_rmse(data_dir: Path, model_tag: str = MODEL_TAG):
    data_dir = Path(data_dir)
    labels_path = data_dir / "inithour_labels.npy"
    if labels_path.exists():
        init_hours = [int(h) for h in np.load(labels_path).tolist()]
    else:
        init_hours = []
        for h in range(24):
            p = data_dir / f"rmse_{model_tag}_inithour_{h:02d}Z.npy"
            if p.exists():
                init_hours.append(h)
        if not init_hours:
            raise FileNotFoundError(f"No rmse_{model_tag}_inithour_XXZ.npy under {data_dir}")

    per_hour = {}
    for h in init_hours:
        p = data_dir / f"rmse_{model_tag}_inithour_{h:02d}Z.npy"
        if not p.exists():
            raise FileNotFoundError(f"Missing init-hour file: {p}")
        per_hour[h] = np.load(p)

    mean_path = data_dir / f"rmse_{model_tag}_inithour_mean.npy"
    if not mean_path.exists():
        raise FileNotFoundError(f"Missing mean RMSE file: {mean_path}")
    return init_hours, per_hour, np.load(mean_path)


def to_channel_time_shape(arr: np.ndarray, n_channels: int) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"Input must be 2D; got shape={arr.shape}")
    if arr.shape[0] == n_channels:
        return arr
    if arr.shape[1] == n_channels:
        return arr.T
    raise ValueError(f"Cannot infer channel dim (expected {n_channels}), shape={arr.shape}")


def to_time_var_shape(arr: np.ndarray, n_vars: int) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"Input must be 2D; got shape={arr.shape}")
    if arr.shape[1] == n_vars:
        return arr
    if arr.shape[0] == n_vars:
        return arr.T
    raise ValueError(f"Cannot infer variable dim (expected {n_vars}), shape={arr.shape}")


def _init_hour_color(hour: int, idx: int) -> str:
    return INIT_HOUR_COLORS.get(hour, FALLBACK_INIT_COLORS[idx % len(FALLBACK_INIT_COLORS)])


def _mean_legend_label(init_hours: list[int]) -> str:
    tags = "/".join(f"{h:02d}" for h in init_hours)
    return f"Mean ({tags}Z)"


def _plot_inithour_curves(ax, times, y_mean, hour_series, init_hours):
    """Plot init-hour curves with shaded band and red mean."""
    for i, h in enumerate(init_hours):
        color = _init_hour_color(h, i)
        label = f"{h:02d}Z"
        y_h = hour_series[h]
        ax.plot(times, y_h, color=color, linestyle="-", linewidth=INIT_LINEWIDTH, label=label, zorder=2)
        ax.fill_between(
            times,
            y_h,
            y_mean,
            facecolor=_shade_fill_color(color),
            alpha=SHADE_ALPHA,
            linewidth=0,
            edgecolor="none",
            zorder=1,
        )
    ax.plot(
        times,
        y_mean,
        color=MEAN_COLOR,
        linestyle="-",
        linewidth=MEAN_LINEWIDTH,
        label=_mean_legend_label(init_hours),
        zorder=5,
    )


def plot_rmse_upper_air_by_inithour(
    init_hours: list[int],
    per_hour: dict[int, np.ndarray],
    mean_rmse: np.ndarray,
    *,
    save_base: str | None = None,
    save_formats=("png", "pdf"),
    dpi: int = FIGURE_SAVE_DPI_HIGH,
    max_lead: int | None = None,
) -> None:
    mean_ct = to_channel_time_shape(mean_rmse, N_AIR_CHANNELS)
    hour_ct = {h: to_channel_time_shape(per_hour[h], N_AIR_CHANNELS) for h in init_hours}

    time_len = mean_ct.shape[1]
    if max_lead is not None:
        time_len = min(time_len, max_lead)
    times = np.arange(time_len)

    fig, axes = plt.subplots(5, 4, figsize=AIR_FIGSIZE)

    for v_idx, _var_name in enumerate(UPPER_VAR_SHORT):
        for l_idx, level in enumerate(UPPER_LEVELS):
            ax = axes[v_idx, l_idx]
            ch_idx = v_idx * 4 + l_idx
            y_mean = mean_ct[ch_idx, :time_len]
            hour_series = {h: hour_ct[h][ch_idx, :time_len] for h in init_hours}
            _plot_inithour_curves(ax, times, y_mean, hour_series, init_hours)

            ax.set_title(upper_air_subplot_title(v_idx, level), fontsize=SUBPLOT_TITLE_SIZE)
            ax.grid(True, alpha=0.3)
            if l_idx == 0:
                ax.set_ylabel(rmse_air_ylabel(v_idx), fontsize=AXES_LABEL_SIZE)
            if v_idx == 4:
                ax.set_xlabel(XLABEL_FORECAST_TIME, fontsize=AXES_LABEL_SIZE)

    add_figure_legend_below(fig, axes, ncol=min(len(init_hours) + 1, 5))
    _save_figure(fig, save_base, save_formats, dpi)


def plot_rmse_surface_by_inithour(
    init_hours: list[int],
    per_hour: dict[int, np.ndarray],
    mean_rmse: np.ndarray,
    *,
    save_base: str | None = None,
    save_formats=("png", "pdf"),
    dpi: int = FIGURE_SAVE_DPI_HIGH,
    max_lead: int | None = None,
) -> None:
    var_units = ["Pa", "m/s", "m/s", "K"]
    mean_tv = to_time_var_shape(mean_rmse, N_SURFACE_VARS)
    hour_tv = {h: to_time_var_shape(per_hour[h], N_SURFACE_VARS) for h in init_hours}

    time_len = mean_tv.shape[0]
    if max_lead is not None:
        time_len = min(time_len, max_lead)
    times = np.arange(time_len)

    fig, axes = plt.subplots(1, 4, figsize=SURFACE_FIGSIZE)

    for v_idx, var_name in enumerate(SURFACE_VAR_SHORT):
        ax = axes[v_idx]
        y_mean = mean_tv[:time_len, v_idx]
        hour_series = {h: hour_tv[h][:time_len, v_idx] for h in init_hours}
        _plot_inithour_curves(ax, times, y_mean, hour_series, init_hours)
        ax.set_title(var_name, fontsize=SUBPLOT_TITLE_SIZE)
        ax.set_xlabel(XLABEL_FORECAST_TIME, fontsize=AXES_LABEL_SIZE)
        ax.set_ylabel(f"RMSE ({var_units[v_idx]})", fontsize=AXES_LABEL_SIZE)
        ax.grid(True, alpha=0.3)

    add_figure_legend_below(fig, axes, ncol=min(len(init_hours) + 1, 5))
    _save_figure(fig, save_base, save_formats, dpi)


def plot_rmse_combined_by_inithour(
    init_hours: list[int],
    per_hour_air: dict[int, np.ndarray],
    mean_air: np.ndarray,
    per_hour_surface: dict[int, np.ndarray],
    mean_surface: np.ndarray,
    *,
    save_base: str | None = None,
    save_formats=("png", "pdf"),
    dpi: int = FIGURE_SAVE_DPI_HIGH,
    max_lead: int | None = None,
) -> None:
    """Upper-air 5x4 + surface 1x4 init-hour curves in one figure."""
    mean_ct = to_channel_time_shape(mean_air, N_AIR_CHANNELS)
    hour_ct = {h: to_channel_time_shape(per_hour_air[h], N_AIR_CHANNELS) for h in init_hours}
    mean_tv = to_time_var_shape(mean_surface, N_SURFACE_VARS)
    hour_tv = {h: to_time_var_shape(per_hour_surface[h], N_SURFACE_VARS) for h in init_hours}

    time_len = mean_ct.shape[1]
    if max_lead is not None:
        time_len = min(time_len, max_lead)
    times = np.arange(time_len)

    n_rows = AIR_GRID_ROWS + 1
    fig, axes = plt.subplots(n_rows, AIR_GRID_COLS, figsize=COMBINED_RMSE_FIGSIZE)
    var_units = ["Pa", "m/s", "m/s", "K"]

    for v_idx, _var_name in enumerate(UPPER_VAR_SHORT):
        for l_idx, level in enumerate(UPPER_LEVELS):
            ax = axes[v_idx, l_idx]
            ch_idx = v_idx * 4 + l_idx
            y_mean = mean_ct[ch_idx, :time_len]
            hour_series = {h: hour_ct[h][ch_idx, :time_len] for h in init_hours}
            _plot_inithour_curves(ax, times, y_mean, hour_series, init_hours)
            ax.set_title(upper_air_subplot_title(v_idx, level), fontsize=SUBPLOT_TITLE_SIZE)
            ax.grid(True, alpha=0.3)
            if l_idx == 0:
                ax.set_ylabel(rmse_air_ylabel(v_idx), fontsize=AXES_LABEL_SIZE)

    surf_row = AIR_GRID_ROWS
    for v_idx, var_name in enumerate(SURFACE_VAR_SHORT):
        ax = axes[surf_row, v_idx]
        y_mean = mean_tv[:time_len, v_idx]
        hour_series = {h: hour_tv[h][:time_len, v_idx] for h in init_hours}
        _plot_inithour_curves(ax, times, y_mean, hour_series, init_hours)
        ax.set_title(var_name, fontsize=SUBPLOT_TITLE_SIZE)
        ax.set_xlabel(XLABEL_FORECAST_TIME, fontsize=AXES_LABEL_SIZE)
        ax.set_ylabel(f"RMSE ({var_units[v_idx]})", fontsize=AXES_LABEL_SIZE)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    apply_curve_grid_spacing(fig)
    add_figure_legend_below(fig, axes, ncol=min(len(init_hours) + 1, 5))
    _save_figure(fig, save_base, save_formats, dpi)


def _save_figure(fig, save_base, save_formats, dpi) -> None:
    if save_base:
        directory = os.path.dirname(save_base)
        if directory:
            os.makedirs(directory, exist_ok=True)
        for fmt in save_formats:
            fmt = fmt.lower().lstrip(".")
            out_path = f"{save_base}.{fmt}"
            save_curve_figure(fig, out_path, dpi=dpi, bbox_inches="tight")
            print(f"Saved: {out_path}")
    plt.close(fig)


def _run_one_kind(
    kind: str,
    *,
    data_dir: Path | None,
    save_dir: Path,
    max_lead: int | None,
) -> None:
    """Plot RMSE curves for one type (air / surface)."""
    resolved_dir = Path(data_dir or resolve_default_data_dir(kind))
    if not resolved_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {resolved_dir}")

    init_hours, per_hour, mean_rmse = load_inithour_rmse(resolved_dir)
    print(f"\n[{kind}] data directory: {resolved_dir}")
    print(f"[{kind}] init hours: {init_hours}")
    print(f"[{kind}] mean shape: {mean_rmse.shape}")

    lead_tag = f"{max_lead}step" if max_lead else f"{mean_rmse.shape[0]}step"
    if kind == "air":
        save_base = str(save_dir / f"rmse_air_aero_ode_by_inithour_{lead_tag}")
        plot_rmse_upper_air_by_inithour(
            init_hours, per_hour, mean_rmse,
            save_base=save_base, max_lead=max_lead,
        )
    else:
        save_base = str(save_dir / f"rmse_surface_aero_ode_by_inithour_{lead_tag}")
        plot_rmse_surface_by_inithour(
            init_hours, per_hour, mean_rmse,
            save_base=save_base, max_lead=max_lead,
        )


def _run_combined(
    *,
    save_dir: Path,
    max_lead: int | None,
) -> None:
    """Upper-air + surface combined 6x4 figure."""
    air_dir = resolve_default_data_dir("air")
    surf_dir = resolve_default_data_dir("surface")
    if not air_dir.is_dir():
        raise FileNotFoundError(f"Upper-air data directory not found: {air_dir}")
    if not surf_dir.is_dir():
        raise FileNotFoundError(f"Surface data directory not found: {surf_dir}")

    init_hours_a, per_hour_a, mean_a = load_inithour_rmse(air_dir)
    init_hours_s, per_hour_s, mean_s = load_inithour_rmse(surf_dir)
    if init_hours_a != init_hours_s:
        print(f"[WARN] init hours differ: air={init_hours_a}, surface={init_hours_s}")
    init_hours = init_hours_a

    print(f"\n[combined] upper-air: {air_dir}")
    print(f"[combined] surface: {surf_dir}")
    print(f"[combined] init hours: {init_hours}")

    lead_tag = f"{max_lead}step" if max_lead else f"{mean_a.shape[0]}step"
    save_base = str(save_dir / f"rmse_aero_ode_by_inithour_{lead_tag}")
    plot_rmse_combined_by_inithour(
        init_hours,
        per_hour_a,
        mean_a,
        per_hour_s,
        mean_s,
        save_base=save_base,
        max_lead=max_lead,
    )


def main():
    parser = argparse.ArgumentParser(description="Plot AERO-ODE RMSE by init hour")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override data directory (--kind air or surface only; both uses top-level paths)",
    )
    parser.add_argument(
        "--kind",
        choices=("both", "air", "surface"),
        default="both",
        help="Plot kind: both=combined 6x4 (default), air/surface=one only",
    )
    parser.add_argument("--max-lead", type=int, default=None)
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Override FIG_SAVE_DIR",
    )
    args = parser.parse_args()

    if args.kind == "both" and args.data_dir is not None:
        raise ValueError("--data-dir cannot be used with --kind both; use air or surface")

    setup_font()

    save_dir = Path(args.save_dir or FIG_SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    if args.kind == "both":
        _run_combined(save_dir=save_dir, max_lead=args.max_lead)
    else:
        _run_one_kind(
            args.kind,
            data_dir=args.data_dir,
            save_dir=save_dir,
            max_lead=args.max_lead,
        )


if __name__ == "__main__":
    main()
