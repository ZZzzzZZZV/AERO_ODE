"""
West (Wb) extreme-event metric curves.

One 6x4 figure: 5 upper-air rows + 1 surface row.
Run: python Draw_Test_Extreme_Metrics_wb.py
"""
from __future__ import annotations

from pathlib import Path
import os
import sys

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_DRAW_NEW = _SCRIPT_DIR.parent
if str(_DRAW_NEW) not in sys.path:
    sys.path.insert(0, str(_DRAW_NEW))

from paths_config import (  # noqa: E402
    AIR_GRID_COLS,
    AIR_GRID_ROWS,
    AXES_LABEL_SIZE,
    COMBINED_RMSE_FIGSIZE,
    LEGEND_FONT_SIZE,
    SUBPLOT_TITLE_SIZE,
    SURFACE_VAR_SHORT,
    TICK_LABEL_SIZE,
    UPPER_LEVELS,
    XLABEL_FORECAST_TIME,
    add_figure_legend_below,
    apply_curve_grid_spacing,
    apply_paper_rcparams,
    font_times_new_roman,
    repo,
    save_curve_figure,
    upper_air_subplot_title,
)

FONT_PATH = str(font_times_new_roman())
BASE_DIR = str(repo())
SAVE_DIR = "./figures_extreme_wb"
REGION_TAG = "wb"
DATA_PREFIX = "Wb"

MODEL_FILES = {
    "AERO-ODE": "Aero_ODE",
    "YingLong-WRF": "yinglong_nwp_bc",
    "WRF-ARW": "nwp",
    "PanGu-Weather": "pangu",
    "NeuralGCM 1.4": "neuralgcm",
    "IFS": "ifs",
}
LEAD_MODEL_GROUPS = {
    48: ["AERO-ODE", "YingLong-WRF", "WRF-ARW"],
    72: ["AERO-ODE", "PanGu-Weather", "NeuralGCM 1.4", "IFS"],
}
METRICS = ["POD", "FAR", "CSI", "ETS"]
EVENT_INDEX = {"low": 0, "high": 1}
N_AIR_VARS = 20
N_SURFACE_VARS = 4
UPPER_LEVELS_LIST = list(UPPER_LEVELS)
SURFACE_VAR_NAMES = list(SURFACE_VAR_SHORT)

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
LINEWIDTHS = [1.8] * len(COLORS)


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


def metric_file_path(metric_name: str, model_stub: str) -> str:
    return os.path.join(BASE_DIR, f"{DATA_PREFIX}_{metric_name.upper()}", f"{model_stub}.npy")


def load_metric_array(metric_name: str, model_stub: str):
    path = metric_file_path(metric_name, model_stub)
    if not os.path.exists(path):
        print(f"[WARN] file not found: {path}")
        return None
    arr = np.load(path)
    if arr.ndim != 3 or arr.shape[0] < 2:
        print(f"[WARN] bad shape: {path}, shape={arr.shape}, expected (2, steps, vars)")
        return None
    return arr


def _plot_series(ax, times, series_list, name_list) -> int:
    plotted = 0
    for i, (y, name) in enumerate(zip(series_list, name_list)):
        if y is None or np.all(np.isnan(y)):
            continue
        ax.plot(
            times,
            y,
            label=name,
            color=get_model_color(name, i),
            linewidth=LINEWIDTHS[i % len(LINEWIDTHS)],
            alpha=0.95,
        )
        plotted += 1
    return plotted


def plot_metric_figure(metric_name: str, event_name: str, lead_hours: int, loaded: dict) -> None:
    """Plot upper-air 20 vars + surface 4 vars in one 6x4 figure."""
    model_names = LEAD_MODEL_GROUPS[lead_hours]
    event_idx = EVENT_INDEX[event_name]
    metric_data = loaded[metric_name]
    times = np.arange(1, lead_hours + 1)

    n_rows = AIR_GRID_ROWS + 1
    fig, axes = plt.subplots(n_rows, AIR_GRID_COLS, figsize=COMBINED_RMSE_FIGSIZE)

    for local_idx in range(N_AIR_VARS):
        row = local_idx // AIR_GRID_COLS
        col = local_idx % AIR_GRID_COLS
        ax = axes[row, col]
        series_list, names = [], []
        for i, model_name in enumerate(model_names):
            if model_name not in metric_data:
                continue
            arr = metric_data[model_name]
            if arr.shape[2] <= local_idx or arr.shape[1] < lead_hours:
                continue
            s = arr[event_idx, :lead_hours, local_idx]
            if np.all(np.isnan(s)):
                continue
            series_list.append(s)
            names.append(model_name)
        plotted = _plot_series(ax, times, series_list, names)
        ax.set_title(upper_air_subplot_title(row, UPPER_LEVELS_LIST[col]), fontsize=SUBPLOT_TITLE_SIZE)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
        ax.set_xlim(1, lead_hours)
        ax.grid(True, alpha=0.35, linestyle="--")
        if col == 0:
            ax.set_ylabel(metric_name, fontsize=AXES_LABEL_SIZE)
        if plotted == 0:
            ax.text(0.5, 0.5, "No valid data", transform=ax.transAxes, ha="center", va="center", fontsize=8)

    surf_row = AIR_GRID_ROWS
    for local_idx, var_name in enumerate(SURFACE_VAR_NAMES):
        ax = axes[surf_row, local_idx]
        global_idx = N_AIR_VARS + local_idx
        series_list, names = [], []
        for i, model_name in enumerate(model_names):
            if model_name not in metric_data:
                continue
            arr = metric_data[model_name]
            if arr.shape[2] <= global_idx or arr.shape[1] < lead_hours:
                continue
            s = arr[event_idx, :lead_hours, global_idx]
            if np.all(np.isnan(s)):
                continue
            series_list.append(s)
            names.append(model_name)
        plotted = _plot_series(ax, times, series_list, names)
        ax.set_title(var_name, fontsize=SUBPLOT_TITLE_SIZE)
        ax.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
        ax.set_xlim(1, lead_hours)
        ax.grid(True, alpha=0.35, linestyle="--")
        ax.set_ylabel(metric_name, fontsize=AXES_LABEL_SIZE)
        if local_idx == N_SURFACE_VARS - 1:
            ax.set_xlabel(XLABEL_FORECAST_TIME, fontsize=AXES_LABEL_SIZE)
        if plotted == 0:
            ax.text(0.5, 0.5, "No valid data", transform=ax.transAxes, ha="center", va="center", fontsize=8)

    fig.tight_layout()
    apply_curve_grid_spacing(fig)
    add_figure_legend_below(fig, axes, ncol=len(model_names), fontsize=LEGEND_FONT_SIZE)

    os.makedirs(SAVE_DIR, exist_ok=True)
    metric_lower = metric_name.lower()
    stem = f"extreme_{metric_lower}_{event_name}_{lead_hours}h_{REGION_TAG}"
    for ext in ("png", "pdf"):
        out_path = os.path.join(SAVE_DIR, f"{stem}.{ext}")
        save_curve_figure(fig, out_path)
        print(f"[SAVED] {out_path}")
    plt.close(fig)


def main():
    setup_font()
    loaded = {m: {} for m in METRICS}

    print("=== Load data ===")
    for metric_name in METRICS:
        for model_name, model_stub in MODEL_FILES.items():
            arr = load_metric_array(metric_name, model_stub)
            if arr is None:
                continue
            loaded[metric_name][model_name] = arr
            print(
                f"[OK] {metric_name:4s} | {model_name:15s} | "
                f"shape={arr.shape}, NaN={np.isnan(arr).sum()}"
            )

    print("\n=== Plotting (upper-air+surfacesame figure)===")
    for lead_hours in (48, 72):
        for event_name in ("low", "high"):
            for metric_name in METRICS:
                plot_metric_figure(metric_name, event_name, lead_hours, loaded)
    print("Done。")


if __name__ == "__main__":
    main()
