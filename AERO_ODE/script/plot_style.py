from __future__ import annotations

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

from script.config import AIR_LEVELS, AIR_VARS

MAP_SUBPLOT_TITLE_SIZE = 9
MAP_SUBPLOT_TITLE_PAD = 3
MAP_DATA_BORDER_WIDTH = 0.4
MAP_SPINE_WIDTH = 0.4
MAP_SUBPLOT_HSPACE = 0.32
MAP_SUBPLOT_WSPACE = 0.05
MAP_AXES_LABELSIZE = 14
MAP_TICK_LABEL_SIZE = 7
CBAR_TICK_LABEL_SIZE = 11
FIGURE_SAVE_DPI = 150

MAP_COMP_TITLE_SIZE = 24
MAP_COMP_TITLE_PAD = 8
MAP_COMP_GRIDLABEL_SIZE = 17
MAP_COMP_AXES_LABELSIZE = 18
MAP_COMP_CBAR_TICK_SIZE = 14
SNAPSHOT_COMP_COL_WIDTH = 6.0
SNAPSHOT_COMP_ROW_HEIGHT = 4.0
SNAPSHOT_COMP_HSPACE = 0.22
SNAPSHOT_COMP_WSPACE = 0.05
SNAPSHOT_COMP_LEFT = 0.03
SNAPSHOT_COMP_RIGHT = 0.90
SNAPSHOT_COMP_TOP = 0.98
SNAPSHOT_COMP_BOTTOM = 0.03
SNAPSHOT_COMP_DISPLAY_WIDTH = 1500

SNAPSHOT_ERROR_WIDTH = 10.0
SNAPSHOT_ERROR_COLS = 2
SNAPSHOT_ROW_HEIGHT = 2.4
SNAPSHOT_DISPLAY_WIDTH = 900
SNAPSHOT_ERROR_DISPLAY_WIDTH = 600

SUBPLOT_TITLE_SIZE = 18
LEGEND_FONT_SIZE = SUBPLOT_TITLE_SIZE
TICK_LABEL_SIZE = 12
AXES_LABEL_SIZE = 18
AIR_SUBPLOT_TITLE_SIZE = 23
AIR_AXES_LABEL_SIZE = 23
AIR_TICK_LABEL_SIZE = 15
FIGURE_SAVE_DPI_HIGH = 350

AIR_GRID_ROWS, AIR_GRID_COLS = 5, 5
N_AIR_CHANNELS = AIR_GRID_ROWS * AIR_GRID_COLS
N_SURFACE_VARS = 4
COMBINED_RMSE_ROWS = AIR_GRID_ROWS + 1
COMBINED_FIGSIZE = (20, 18)
COMBINED_RMSE_FIGSIZE = COMBINED_FIGSIZE
AIR_FIGSIZE = (COMBINED_FIGSIZE[0], COMBINED_FIGSIZE[1] * AIR_GRID_ROWS / COMBINED_RMSE_ROWS)
SURFACE_FIGSIZE = (COMBINED_FIGSIZE[0] * N_SURFACE_VARS / AIR_GRID_COLS, 3.6)

AIR_FIGSIZE_REF = (16, 15)
EXTREME_AIR_FIGSIZE = (18, 16)
CURVE_SUBPLOT_WSPACE = 0.30
CURVE_SUBPLOT_HSPACE = 0.38
CURVE_SUBPLOT_LEFT = 0.057
CURVE_SUBPLOT_RIGHT = 0.99
AIR_BOTTOM_ROW_Y0 = 0.1065
AIR_BOTTOM_ROW_HEIGHT = 0.138
SURFACE_FIG_TOP_MARGIN_IN = 0.2
AIR_LEGEND_GAP_SCALE = 0.45

UPPER_VAR_SHORT = AIR_VARS
UPPER_VAR_UNITS = ("gpm", "K", "kg/kg", "m/s", "m/s")
UPPER_LEVELS = AIR_LEVELS
SURFACE_VAR_SHORT = ("MSLP", "U10", "V10", "T2M")
SURFACE_VAR_UNITS = ("Pa", "m/s", "m/s", "K")
XLABEL_FORECAST_TIME = "Forecast Time (hours)"

MODEL_COLORS = {
    "AERO-ODE": "#D55E00",
    "NeuralGCM-LAM": "#D55E00",
    "WRF-ARW": "#009E73",
    "WRF": "#009E73",
    "NWP": "#009E73",
    "YingLong-WRF": "#0072B2",
    "PanGu-Weather": "#CC79A7",
    "NeuralGCM 1.4": "#E69F00",
    "IFS": "#56B4E9",
    "HRRR Reference": "#4D4D4D",
}
COLORS = list(dict.fromkeys(MODEL_COLORS.values()))
LINEWIDTHS = [1.5] * 10

def setup_font() -> None:
    times_name = None
    for font in fm.fontManager.ttflist:
        if font.name.lower() == "times new roman":
            times_name = font.name
            break
    if times_name is not None:
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = [times_name]
    else:
        plt.rcParams["font.family"] = "DejaVu Serif"
    plt.rcParams["axes.unicode_minus"] = False

def upper_air_subplot_title(var_idx: int, level: int) -> str:
    return f"{UPPER_VAR_SHORT[var_idx]}{level}"

def rmse_air_ylabel(var_idx: int) -> str:
    return f"RMSE({UPPER_VAR_UNITS[var_idx]})"

def get_model_color(model_name: str, fallback_index: int = 0) -> str:
    return MODEL_COLORS.get(model_name, COLORS[fallback_index % len(COLORS)])

def comparison_figsize(n_rows: int, n_cols: int) -> tuple[float, float]:
    return (SNAPSHOT_COMP_COL_WIDTH * n_cols, SNAPSHOT_COMP_ROW_HEIGHT * n_rows)

def apply_paper_rcparams(*, map_style: bool = False) -> None:
    setup_font()
    if map_style:
        plt.rcParams["axes.labelsize"] = MAP_AXES_LABELSIZE
        plt.rcParams["xtick.labelsize"] = MAP_TICK_LABEL_SIZE
        plt.rcParams["ytick.labelsize"] = MAP_TICK_LABEL_SIZE
    else:
        plt.rcParams["xtick.labelsize"] = TICK_LABEL_SIZE
        plt.rcParams["ytick.labelsize"] = TICK_LABEL_SIZE
        plt.rcParams["axes.labelsize"] = AXES_LABEL_SIZE
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["axes.linewidth"] = 1.5

def apply_curve_grid_spacing(fig) -> None:
    fig.subplots_adjust(wspace=CURVE_SUBPLOT_WSPACE, hspace=CURVE_SUBPLOT_HSPACE)

def save_viz_figure(fig, path, *, dpi: int = FIGURE_SAVE_DPI) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")

def save_curve_figure(
    fig,
    path,
    *,
    dpi: int = FIGURE_SAVE_DPI_HIGH,
    bbox_inches: str = "tight",
) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches, facecolor="white")

def curve_subplot_column_layout(
    ncols: int = N_SURFACE_VARS,
    wspace: float = CURVE_SUBPLOT_WSPACE,
    left: float = CURVE_SUBPLOT_LEFT,
    right: float = CURVE_SUBPLOT_RIGHT,
) -> tuple[tuple[float, ...], float]:
    span = right - left
    width = span / (ncols + (ncols - 1) * wspace)
    x0s: list[float] = []
    x = left
    for _ in range(ncols):
        x0s.append(x)
        x += width * (1.0 + wspace)
    return tuple(x0s), width

def _air_figsize_for_figure(fig) -> tuple[float, float]:
    if abs(fig.get_figwidth() - EXTREME_AIR_FIGSIZE[0]) < 0.5:
        return EXTREME_AIR_FIGSIZE
    return AIR_FIGSIZE_REF

def layout_single_row_axes_like_air(
    fig,
    axes_list: list,
    ncols: int,
    air_figsize: tuple[float, float] = AIR_FIGSIZE_REF,
) -> None:
    h_fig = fig.get_figheight()
    w_air, h_air = air_figsize
    y0 = AIR_BOTTOM_ROW_Y0 * h_air / h_fig
    height = AIR_BOTTOM_ROW_HEIGHT * h_air / h_fig
    x0s, width = curve_subplot_column_layout(ncols)
    for i, ax in enumerate(axes_list):
        ax.set_position([x0s[i], y0, width, height])

def _iter_axes(axes) -> list:
    if hasattr(axes, "ravel"):
        return list(axes.ravel())
    if hasattr(axes, "flat"):
        return list(axes.flat)
    return list(axes)

def _subplot_row_gap(fig, axes_list: list, nrows: int, ncols: int) -> float:
    if nrows >= 2:
        ax_upper = axes_list[-ncols * 2]
        ax_lower = axes_list[-ncols]
        return ax_upper.get_position().y0 - ax_lower.get_position().y1
    ax = axes_list[0]
    pos = ax.get_position()
    return max(0.016, pos.height * 0.14)

def _bottom_label_bottom_y(fig, axes_list: list, ncols: int) -> float:
    fig.canvas.draw()
    y_min = float("inf")
    for ax in axes_list[-ncols:]:
        if ax.xaxis.label.get_text().strip():
            bb = ax.xaxis.label.get_window_extent().transformed(fig.transFigure.inverted())
            y_min = min(y_min, bb.y0)
        for lbl in ax.get_xticklabels():
            if lbl.get_visible() and lbl.get_text():
                bb = lbl.get_window_extent().transformed(fig.transFigure.inverted())
                y_min = min(y_min, bb.y0)
    if y_min == float("inf"):
        y_min = min(ax.get_position().y0 for ax in axes_list[-ncols:])
    return y_min

def add_figure_legend_below(
    fig,
    axes,
    ncol: int | None = None,
    fontsize: float = LEGEND_FONT_SIZE,
    *,
    legend_handles=None,
    legend_labels=None,
) -> None:
    if (legend_handles is None) != (legend_labels is None):
        raise ValueError("legend_handles and legend_labels must both be provided")
    if legend_handles is not None:
        handles = list(legend_handles)
        labels = list(legend_labels)
    else:
        handles, labels = [], []
        for ax in _iter_axes(axes):
            h, lab = ax.get_legend_handles_labels()
            if h:
                handles, labels = h, lab
                break
    if not handles:
        return
    n = len(labels)
    ncol = ncol or n
    axes_list = _iter_axes(axes)
    gs = axes_list[0].get_subplotspec().get_gridspec()
    nrows, ncols_gs = gs.nrows, gs.ncols
    fig_pad = 0.008
    single_row = nrows == 1
    air_ref = _air_figsize_for_figure(fig)
    if single_row:
        layout_single_row_axes_like_air(fig, axes_list, ncols_gs, air_ref)
    else:
        fig.tight_layout()
        apply_curve_grid_spacing(fig)
    fig.canvas.draw()

    row_gap = _subplot_row_gap(fig, axes_list, nrows, ncols_gs)
    legend_gap = row_gap * AIR_LEGEND_GAP_SCALE if nrows >= 2 else row_gap

    if not single_row:
        tmp = fig.legend(
            handles, labels, loc="upper center", ncol=ncol, fontsize=fontsize, frameon=True
        )
        fig.canvas.draw()
        leg_h = tmp.get_window_extent().transformed(fig.transFigure.inverted()).height
        tmp.remove()
        extra_bottom = legend_gap + leg_h + fig_pad
        fig.subplots_adjust(bottom=fig.subplotpars.bottom + extra_bottom)
        fig.canvas.draw()
        row_gap = _subplot_row_gap(fig, axes_list, nrows, ncols_gs)
        legend_gap = row_gap * AIR_LEGEND_GAP_SCALE

    xlabel_bottom = _bottom_label_bottom_y(fig, axes_list, ncols_gs)
    legend_y_top = xlabel_bottom - legend_gap
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=ncol,
        fontsize=fontsize,
        frameon=True,
        bbox_to_anchor=(0.5, legend_y_top),
        bbox_transform=fig.transFigure,
    )
    fig.canvas.draw()
    leg_bb = legend.get_window_extent().transformed(fig.transFigure.inverted())
    dy = legend_y_top - leg_bb.y1
    if abs(dy) > 0.002:
        legend.set_bbox_to_anchor((0.5, legend_y_top + dy), transform=fig.transFigure)
        fig.canvas.draw()
        leg_bb = legend.get_window_extent().transformed(fig.transFigure.inverted())
    if leg_bb.y0 < fig_pad and not single_row:
        fig.subplots_adjust(bottom=fig.subplotpars.bottom + fig_pad - leg_bb.y0)
