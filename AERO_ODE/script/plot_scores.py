from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from script.config import OUTPUT_ROOT, WRF_FORECAST_HOURS, forecast_skill_slice, usable_file
from script.plot_style import (
    AIR_AXES_LABEL_SIZE,
    AIR_FIGSIZE,
    AIR_GRID_COLS,
    AIR_GRID_ROWS,
    AIR_SUBPLOT_TITLE_SIZE,
    AIR_TICK_LABEL_SIZE,
    AXES_LABEL_SIZE,
    FIGURE_SAVE_DPI_HIGH,
    LEGEND_FONT_SIZE,
    LINEWIDTHS,
    N_AIR_CHANNELS,
    N_SURFACE_VARS,
    SUBPLOT_TITLE_SIZE,
    SURFACE_FIGSIZE,
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
    setup_font,
    upper_air_subplot_title,
)

CANON_TOTAL_VARS = 39
AIR_SLICE = slice(0, 25)
SURFACE_SLICE = slice(35, 39)
ACC_YLABEL = "ACC"
DEFAULT_LEAD_HOURS = 48
DEFAULT_MODELS = ("AERO-ODE", "WRF-ARW")
MODEL_FILES = {
    "AERO-ODE": "aero",
    "WRF-ARW": "wrf",
}

def canonical_air(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    return None if arr is None else np.asarray(arr)[:, AIR_SLICE]

def canonical_surface(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if arr is None:
        return None
    surf = np.asarray(arr)[:, SURFACE_SLICE]
    return surf if np.isfinite(surf).any() else None

def to_channel_time_shape(arr: np.ndarray, n_channels: int = N_AIR_CHANNELS) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"expected 2D array, got shape={arr.shape}")
    if arr.shape[0] == n_channels:
        return arr
    if arr.shape[1] == n_channels:
        return arr.T
    raise ValueError(f"cannot find channel dim (need size {n_channels}), shape={arr.shape}")

def to_time_var_shape(arr: np.ndarray, n_vars: int = N_SURFACE_VARS) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"expected 2D array, got shape={arr.shape}")
    if arr.shape[1] == n_vars:
        return arr
    if arr.shape[0] == n_vars:
        return arr.T
    raise ValueError(f"cannot find var dim (need size {n_vars}), shape={arr.shape}")

def _pad_time_first(arr: np.ndarray, t_len: int) -> np.ndarray:
    if arr.shape[0] == t_len:
        return arr
    if arr.shape[0] > t_len:
        return arr[:t_len]
    out = np.full((t_len,) + arr.shape[1:], np.nan, dtype=np.float64)
    out[: arr.shape[0]] = arr
    return out

def _prepare_air_surface_lists(air_data_list, surface_data_list, name_list):
    if not (len(air_data_list) == len(surface_data_list) == len(name_list)):
        raise ValueError("air_data_list, surface_data_list, and name_list length mismatch")
    air_tv = [to_channel_time_shape(d).T for d in air_data_list]
    surf_tv: List[Optional[np.ndarray]] = []
    for raw in surface_data_list:
        if raw is None:
            surf_tv.append(None)
        else:
            surf_tv.append(to_time_var_shape(raw))
    time_len = max(d.shape[0] for d in air_tv)
    air_data = [to_channel_time_shape(_pad_time_first(d, time_len).T) for d in air_tv]
    surf_data = [None if d is None else _pad_time_first(d, time_len) for d in surf_tv]
    return air_data, surf_data, time_len

def _plot_series_on_ax(
    ax,
    times,
    series_list,
    name_list,
    *,
    linestyle_list=None,
    linewidths=LINEWIDTHS,
    alpha: float = 1.0,
) -> int:
    plotted = 0
    for i, (y, name) in enumerate(zip(series_list, name_list)):
        if y is None or np.all(np.isnan(y)):
            continue
        ls = "-" if linestyle_list is None else linestyle_list[i]
        ax.plot(
            times,
            y,
            color=get_model_color(name, i),
            linestyle=ls,
            linewidth=linewidths[i % len(linewidths)],
            alpha=alpha,
            label=name,
            zorder=2.5,
            solid_capstyle="round",
            dash_capstyle="round",
        )
        plotted += 1
    return plotted

def _legend_handles(names: Sequence[str], linewidths=LINEWIDTHS, linestyle_list=None):
    return [
        Line2D(
            [],
            [],
            color=get_model_color(name, i),
            linewidth=linewidths[i % len(linewidths)],
            linestyle="-" if linestyle_list is None else linestyle_list[i],
        )
        for i, name in enumerate(names)
    ]

def _save_ab_figure(
    fig,
    axes,
    name_list,
    save_base,
    save_formats,
    dpi,
    bbox_inches,
    show,
    *,
    show_legend: bool,
    linewidths=LINEWIDTHS,
    linestyle_list=None,
) -> List[Path]:
    fig.tight_layout()
    apply_curve_grid_spacing(fig)
    if show_legend:
        add_figure_legend_below(
            fig,
            axes,
            ncol=len(name_list),
            fontsize=LEGEND_FONT_SIZE,
            legend_handles=_legend_handles(name_list, linewidths, linestyle_list),
            legend_labels=list(name_list),
        )
    saved: List[Path] = []
    if save_base:
        directory = os.path.dirname(save_base)
        if directory:
            os.makedirs(directory, exist_ok=True)
        for fmt in save_formats:
            fmt = fmt.lower().lstrip(".")
            out_path = Path(f"{save_base}.{fmt}")
            save_curve_figure(fig, out_path, dpi=dpi, bbox_inches=bbox_inches)
            saved.append(out_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return saved

def _draw_rmse_air(axes, air_data, name_list, times, linestyle_list=None) -> None:
    n_levels = len(UPPER_LEVELS)
    for v_idx, _var_name in enumerate(UPPER_VAR_SHORT):
        for l_idx, level in enumerate(UPPER_LEVELS):
            ax = axes[v_idx, l_idx]
            ch_idx = v_idx * n_levels + l_idx
            _plot_series_on_ax(
                ax,
                times,
                [d[ch_idx] for d in air_data],
                name_list,
                linestyle_list=linestyle_list,
            )
            ax.set_title(upper_air_subplot_title(v_idx, level), fontsize=AIR_SUBPLOT_TITLE_SIZE)
            ax.tick_params(axis="both", labelsize=AIR_TICK_LABEL_SIZE)
            ax.grid(True, alpha=0.3)
            if l_idx == 0:
                ax.set_ylabel(rmse_air_ylabel(v_idx), fontsize=AIR_AXES_LABEL_SIZE)
            if v_idx == AIR_GRID_ROWS - 1:
                ax.set_xlabel(XLABEL_FORECAST_TIME, fontsize=AIR_AXES_LABEL_SIZE)

def _draw_rmse_surface(axes, surf_data, name_list, times) -> None:
    for v_idx, (var_name, unit) in enumerate(zip(SURFACE_VAR_SHORT, SURFACE_VAR_UNITS)):
        ax = axes[v_idx]
        series, surf_names = [], []
        for data, name in zip(surf_data, name_list):
            if data is None:
                continue
            series.append(data[:, v_idx])
            surf_names.append(name)
        _plot_series_on_ax(ax, times, series, surf_names)
        ax.set_title(var_name, fontsize=SUBPLOT_TITLE_SIZE)
        ax.set_xlabel(XLABEL_FORECAST_TIME, fontsize=AXES_LABEL_SIZE)
        ax.set_ylabel(f"RMSE ({unit})", fontsize=AXES_LABEL_SIZE)
        ax.grid(True, alpha=0.3)

def _draw_acc_air(axes, air_data, name_list, times) -> None:
    n_levels = len(UPPER_LEVELS)
    for v_idx, _var_name in enumerate(UPPER_VAR_SHORT):
        for l_idx, level in enumerate(UPPER_LEVELS):
            ax = axes[v_idx, l_idx]
            ch_idx = v_idx * n_levels + l_idx
            _plot_series_on_ax(ax, times, [d[ch_idx] for d in air_data], name_list)
            ax.set_title(upper_air_subplot_title(v_idx, level), fontsize=AIR_SUBPLOT_TITLE_SIZE)
            ax.tick_params(axis="both", labelsize=AIR_TICK_LABEL_SIZE)
            ax.grid(True, alpha=0.3)
            if l_idx == 0:
                ax.set_ylabel(ACC_YLABEL, fontsize=AIR_AXES_LABEL_SIZE)
            if v_idx == AIR_GRID_ROWS - 1:
                ax.set_xlabel(XLABEL_FORECAST_TIME, fontsize=AIR_AXES_LABEL_SIZE)

def _draw_acc_surface(axes, surf_data, name_list, times) -> None:
    for v_idx, var_name in enumerate(SURFACE_VAR_SHORT):
        ax = axes[v_idx]
        series, surf_names = [], []
        for data, name in zip(surf_data, name_list):
            if data is None:
                continue
            series.append(data[:, v_idx])
            surf_names.append(name)
        _plot_series_on_ax(ax, times, series, surf_names)
        ax.set_title(var_name, fontsize=SUBPLOT_TITLE_SIZE)
        ax.set_xlabel(XLABEL_FORECAST_TIME, fontsize=AXES_LABEL_SIZE)
        ax.set_ylabel(ACC_YLABEL, fontsize=AXES_LABEL_SIZE)
        ax.grid(True, alpha=0.3)

def plot_rmse_combined(
    air_data_list,
    surface_data_list,
    name_list,
    save_base=None,
    save_formats=("png", "pdf"),
    dpi: int = FIGURE_SAVE_DPI_HIGH,
    bbox_inches: str = "tight",
    show: bool = False,
    linestyle_list=None,
    times=None,
) -> List[Path]:
    apply_paper_rcparams()
    air_data, surf_data, time_len = _prepare_air_surface_lists(
        air_data_list, surface_data_list, name_list
    )
    if times is None:
        times = np.arange(time_len)
    else:
        times = np.asarray(times)
        if times.shape[0] != time_len:
            raise ValueError(f"times length {times.shape[0]} != {time_len}")
    surface_names = [
        name
        for data, name in zip(surf_data, name_list)
        if data is not None and np.isfinite(data).any()
    ]
    legends_differ = surface_names != list(name_list)
    saved: List[Path] = []

    fig_air, axes_air = plt.subplots(AIR_GRID_ROWS, AIR_GRID_COLS, figsize=AIR_FIGSIZE)
    _draw_rmse_air(axes_air, air_data, name_list, times, linestyle_list=linestyle_list)
    saved.extend(
        _save_ab_figure(
            fig_air,
            axes_air,
            name_list,
            f"{save_base}_air" if save_base else None,
            save_formats,
            dpi,
            bbox_inches,
            show,
            show_legend=legends_differ,
            linestyle_list=linestyle_list,
        )
    )

    if not any(data is not None and np.isfinite(data).any() for data in surf_data):
        return saved

    fig_surface, axes_surface = plt.subplots(1, N_SURFACE_VARS, figsize=SURFACE_FIGSIZE)
    _draw_rmse_surface(axes_surface, surf_data, name_list, times)
    saved.extend(
        _save_ab_figure(
            fig_surface,
            axes_surface,
            surface_names,
            f"{save_base}_surface" if save_base else None,
            save_formats,
            dpi,
            bbox_inches,
            show,
            show_legend=True,
            linestyle_list=linestyle_list,
        )
    )
    return saved

def plot_acc_combined(
    air_data_list,
    surface_data_list,
    name_list,
    save_base=None,
    save_formats=("png", "pdf"),
    dpi: int = FIGURE_SAVE_DPI_HIGH,
    bbox_inches: str = "tight",
    show: bool = False,
) -> List[Path]:
    apply_paper_rcparams()
    air_data, surf_data, time_len = _prepare_air_surface_lists(
        air_data_list, surface_data_list, name_list
    )
    times = np.arange(time_len)
    surface_names = [
        name
        for data, name in zip(surf_data, name_list)
        if data is not None and np.isfinite(data).any()
    ]
    legends_differ = surface_names != list(name_list)
    saved: List[Path] = []

    fig_air, axes_air = plt.subplots(AIR_GRID_ROWS, AIR_GRID_COLS, figsize=AIR_FIGSIZE)
    _draw_acc_air(axes_air, air_data, name_list, times)
    saved.extend(
        _save_ab_figure(
            fig_air,
            axes_air,
            name_list,
            f"{save_base}_air" if save_base else None,
            save_formats,
            dpi,
            bbox_inches,
            show,
            show_legend=legends_differ,
        )
    )

    if not any(data is not None and np.isfinite(data).any() for data in surf_data):
        return saved

    fig_surface, axes_surface = plt.subplots(1, N_SURFACE_VARS, figsize=SURFACE_FIGSIZE)
    _draw_acc_surface(axes_surface, surf_data, name_list, times)
    saved.extend(
        _save_ab_figure(
            fig_surface,
            axes_surface,
            surface_names,
            f"{save_base}_surface" if save_base else None,
            save_formats,
            dpi,
            bbox_inches,
            show,
            show_legend=True,
        )
    )
    return saved

def _load_curve(path: Path, steps: Optional[int] = None) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    arr = np.load(str(path))
    if arr.ndim != 2 or arr.shape[1] < CANON_TOTAL_VARS:
        return None
    if steps is not None:
        arr = arr[:steps]
    if not np.isfinite(arr[:, AIR_SLICE]).any():
        return None
    return arr

def collect_curve_lists(
    input_dir: Path,
    steps: int,
    model_names: Sequence[str] = DEFAULT_MODELS,
    kind: str = "rmse",
):
    air_list, surf_list, names = [], [], []
    for name in model_names:
        stub = MODEL_FILES[name]
        arr = _load_curve(input_dir / f"{stub}_{kind}.npy", steps)
        if arr is None:
            print(f"[skip] {name}: no {kind} data")
            continue
        air_list.append(canonical_air(arr))
        surf_list.append(canonical_surface(arr))
        names.append(name)
        print(f"[ok] {name} {kind} <- {stub}_{kind}.npy[:{arr.shape[0]}]")
    return air_list, surf_list, names

def _score_figure_paths(
    save_dir: Path,
    kind: str,
    lead_hours: int,
    formats: Sequence[str],
) -> List[Path]:
    return [
        save_dir / f"{kind}_{lead_hours}_wb_{panel}.{fmt}"
        for panel in ("air", "surface")
        for fmt in formats
    ]

def _reuse_score_figures(
    save_dir: Path,
    input_dir: Path,
    kind: str,
    lead_hours: int,
    formats: Sequence[str],
) -> Optional[List[Path]]:
    required = _score_figure_paths(save_dir, kind, lead_hours, formats)
    if not all(usable_file(path) for path in required):
        return None
    npy_mtime = max(
        (input_dir / f"{stub}_{kind}.npy").stat().st_mtime
        for stub in ("aero", "wrf")
        if (input_dir / f"{stub}_{kind}.npy").exists()
    ) if any((input_dir / f"{stub}_{kind}.npy").exists() for stub in ("aero", "wrf")) else 0
    if npy_mtime and any(path.stat().st_mtime < npy_mtime for path in required):
        return None
    return required

def _wrf_air_surface(wrf_rmse: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    wrf = np.asarray(wrf_rmse, dtype=np.float64)
    if wrf.ndim != 2:
        raise ValueError(f"expected 2D WRF RMSE array, got shape={wrf.shape}")
    if wrf.shape[1] >= 39:
        return wrf[:, :N_AIR_CHANNELS], wrf[:, 35:39]
    if wrf.shape[0] >= 39:
        return wrf[:N_AIR_CHANNELS].T, wrf[35:39].T
    raise ValueError(f"WRF RMSE expected (T, 39), got {wrf.shape}")


def plot_wd_rmse(
    air_rmse: np.ndarray,
    surface_rmse: np.ndarray,
    wrf_rmse: Optional[np.ndarray] = None,
    save_dir: Optional[Path] = None,
    lead_hours: Optional[int] = None,
    save_formats: Iterable[str] = ("png", "pdf"),
    show: bool = False,
) -> List[Path]:
    setup_font()
    apply_paper_rcparams()
    save_dir = Path(save_dir or OUTPUT_ROOT / "figures_wb")
    save_dir.mkdir(parents=True, exist_ok=True)

    air = np.asarray(air_rmse, dtype=np.float64)
    surf = np.asarray(surface_rmse, dtype=np.float64)
    if air.ndim != 2 or surf.ndim != 2:
        raise ValueError(f"expected 2D RMSE arrays, got air={air.shape} surface={surf.shape}")
    if air.shape[0] == N_AIR_CHANNELS:
        air = air.T
    if surf.shape[0] == N_SURFACE_VARS:
        surf = surf.T
    if air.shape[1] != N_AIR_CHANNELS or surf.shape[1] != N_SURFACE_VARS:
        raise ValueError(
            f"unexpected RMSE channels: air={air.shape} surface={surf.shape}"
        )
    air = forecast_skill_slice(air)
    surf = forecast_skill_slice(surf)

    air_list = [air.T]
    surf_list = [surf]
    names = ["AERO-ODE"]
    if wrf_rmse is not None and np.isfinite(wrf_rmse).any():
        wrf_air, wrf_surf = _wrf_air_surface(wrf_rmse)
        if lead_hours is None:
            lead_hours = min(WRF_FORECAST_HOURS, wrf_air.shape[0])
        wrf_air = wrf_air[:lead_hours]
        wrf_surf = wrf_surf[:lead_hours]
        air_list.append(wrf_air.T)
        surf_list.append(wrf_surf)
        names.append("WRF-ARW")
    if lead_hours is not None:
        air = air[:lead_hours]
        surf = surf[:lead_hours]
        air_list[0] = air.T
        surf_list[0] = surf
    t_len = air.shape[0]
    lead_times = np.arange(1, t_len + 1)

    return plot_rmse_combined(
        air_list,
        surf_list,
        names,
        save_base=str(save_dir / "rmse_wd"),
        save_formats=tuple(save_formats),
        show=show,
        times=lead_times,
    )


def plot_aero_wrf_scores(
    input_dir: Optional[Path] = None,
    save_dir: Optional[Path] = None,
    lead_hours: int = DEFAULT_LEAD_HOURS,
    model_names: Sequence[str] = DEFAULT_MODELS,
    save_formats: Iterable[str] = ("png", "pdf"),
    show: bool = False,
    reuse_existing: bool = True,
) -> List[Path]:
    setup_font()
    apply_paper_rcparams()
    input_dir = Path(input_dir or OUTPUT_ROOT)
    save_dir = Path(save_dir or input_dir / "figures_wb")
    save_dir.mkdir(parents=True, exist_ok=True)
    formats = tuple(save_formats)
    saved: List[Path] = []

    cached = (
        _reuse_score_figures(save_dir, input_dir, "rmse", lead_hours, formats)
        if reuse_existing
        else None
    )
    if cached is not None:
        saved.extend(cached)
    else:
        air, surf, names = collect_curve_lists(input_dir, lead_hours, model_names, "rmse")
        if names:
            saved.extend(
                plot_rmse_combined(
                    air,
                    surf,
                    names,
                    save_base=str(save_dir / f"rmse_{lead_hours}_wb"),
                    save_formats=formats,
                    show=show,
                )
            )

    cached = (
        _reuse_score_figures(save_dir, input_dir, "acc", lead_hours, formats)
        if reuse_existing
        else None
    )
    if cached is not None:
        saved.extend(cached)
    else:
        air, surf, names = collect_curve_lists(input_dir, lead_hours, model_names, "acc")
        if names:
            saved.extend(
                plot_acc_combined(
                    air,
                    surf,
                    names,
                    save_base=str(save_dir / f"acc_{lead_hours}_wb"),
                    save_formats=formats,
                    show=show,
                )
            )
    return saved

def png_paths(saved: Sequence[Path]) -> List[Path]:
    return [p for p in saved if p.suffix.lower() == ".png"]

def main(argv: Optional[Sequence[str]] = None) -> List[Path]:
    parser = argparse.ArgumentParser(description="Plot AERO-ODE / WRF-ARW RMSE and ACC in the paper style")
    parser.add_argument("--input_dir", type=Path, default=None)
    parser.add_argument("--save_dir", type=Path, default=None)
    parser.add_argument("--lead", type=int, default=DEFAULT_LEAD_HOURS)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--recompute", action="store_true", help="ignore existing score plots and redraw all")
    args = parser.parse_args(argv)
    setup_font()
    return plot_aero_wrf_scores(
        input_dir=args.input_dir,
        save_dir=args.save_dir,
        lead_hours=args.lead,
        show=args.show,
        reuse_existing=not args.recompute,
    )

if __name__ == "__main__":
    main()
