"""Self-contained plot styling for quick-start (no AERO_ODE_Draw imports)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# Map figure style
MAP_SUBPLOT_TITLE_SIZE = 9
MAP_SUBPLOT_TITLE_PAD = 3
MAP_DATA_BORDER_WIDTH = 0.4
MAP_SPINE_WIDTH = 0.4
MAP_SUBPLOT_HSPACE = 0.32
MAP_SUBPLOT_WSPACE = 0.05
MAP_LON_LABEL_PAD = 0.10  # extra bottom margin inside axes for lon tick labels
MAP_AXES_LABELSIZE = 14
MAP_TICK_LABEL_SIZE = 7
CBAR_TICK_LABEL_SIZE = 11
FIGURE_SAVE_DPI = 150

# Snapshot layout (inches per row)
SNAPSHOT_COMP_WIDTH = 10.0
SNAPSHOT_ERROR_WIDTH = 5.5
SNAPSHOT_ROW_HEIGHT = 2.4

# RMSE curve layout (matches Draw_Test_RMER_rb)
SUBPLOT_TITLE_SIZE = 18
LEGEND_FONT_SIZE = 18
TICK_LABEL_SIZE = 12
AXES_LABEL_SIZE = 18
FIGURE_SAVE_DPI_HIGH = 350

AIR_FIGSIZE = (16, 15)
AIR_GRID_ROWS, AIR_GRID_COLS = 5, 4
COMBINED_RMSE_ROWS = AIR_GRID_ROWS + 1
COMBINED_RMSE_FIGSIZE = (
    AIR_FIGSIZE[0],
    AIR_FIGSIZE[1] * COMBINED_RMSE_ROWS / AIR_GRID_ROWS,
)
CURVE_SUBPLOT_WSPACE = 0.30
CURVE_SUBPLOT_HSPACE = 0.38
AIR_LEGEND_GAP_SCALE = 0.45

UPPER_VAR_SHORT = ("Z", "T", "S", "U", "V")
UPPER_VAR_UNITS = ("gpm", "K", "kg/kg", "m/s", "m/s")
UPPER_LEVELS = (50, 500, 850, 1000)
SURFACE_VAR_SHORT = ("MSLP", "U10", "V10", "T2M")
SURFACE_VAR_UNITS = ("Pa", "m/s", "m/s", "K")
XLABEL_FORECAST_TIME = "Forecast Time (hours)"

MODEL_COLORS = {
    "AERO-ODE": "#6BAED6",
}


def font_times_new_roman() -> Path:
    return Path(__file__).resolve().parents[1] / "AERO_ODE_Draw" / "Times New Roman.ttf"


def setup_font() -> None:
    """Load Times New Roman from AERO_ODE_Draw/Times New Roman.ttf."""
    font_path = font_times_new_roman()
    try:
        if font_path.exists():
            fm.fontManager.addfont(str(font_path))
            font_name = FontProperties(fname=str(font_path)).get_name()
            plt.rcParams["font.family"] = font_name
        else:
            plt.rcParams["font.family"] = "serif"
            plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "serif"]
    except Exception:
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "serif"]
    plt.rcParams["axes.unicode_minus"] = False


def upper_air_subplot_title(var_idx: int, level: int) -> str:
    return f"{UPPER_VAR_SHORT[var_idx]}{level}"


def rmse_air_ylabel(var_idx: int) -> str:
    return f"RMSE({UPPER_VAR_UNITS[var_idx]})"


def get_model_color(model_name: str, fallback_index: int = 0) -> str:
    colors = list(MODEL_COLORS.values())
    return MODEL_COLORS.get(model_name, colors[fallback_index % len(colors)])


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
    st = getattr(fig, "_suptitle", None)
    if st is not None:
        st.set_visible(False)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")


def save_curve_figure(fig, path, *, dpi: int = FIGURE_SAVE_DPI_HIGH, bbox_inches: str = "tight") -> None:
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches, facecolor="white")


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


def _subplot_row_gap(fig, axes_list: list, nrows: int, ncols: int) -> float:
    if nrows >= 2:
        ax_upper = axes_list[-ncols * 2]
        ax_lower = axes_list[-ncols]
        return ax_upper.get_position().y0 - ax_lower.get_position().y1
    return axes_list[0].get_position().height * 0.14


def add_figure_legend_below(fig, axes, ncol: int | None = None, fontsize: float = LEGEND_FONT_SIZE) -> None:
    """Place shared legend below the bottom x-axis labels."""
    handles, labels = [], []
    for ax in axes.ravel():
        h, lab = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, lab
            break
    if not handles:
        return

    ncol = ncol or len(labels)
    axes_list = list(axes.ravel())
    nrows, ncols = axes.shape
    fig_pad = 0.008

    fig.canvas.draw()
    row_gap = _subplot_row_gap(fig, axes_list, nrows, ncols)
    legend_gap = row_gap * AIR_LEGEND_GAP_SCALE

    xlabel_bottom = _bottom_label_bottom_y(fig, axes_list, ncols)
    legend_y_top = xlabel_bottom - legend_gap

    tmp = fig.legend(
        handles, labels, loc="upper center", ncol=ncol, fontsize=fontsize, frameon=True,
    )
    fig.canvas.draw()
    leg_h = tmp.get_window_extent().transformed(fig.transFigure.inverted()).height
    tmp.remove()

    extra_bottom = legend_gap + leg_h + fig_pad
    fig.subplots_adjust(bottom=fig.subplotpars.bottom + extra_bottom)
    fig.canvas.draw()

    row_gap = _subplot_row_gap(fig, axes_list, nrows, ncols)
    legend_gap = row_gap * AIR_LEGEND_GAP_SCALE
    xlabel_bottom = _bottom_label_bottom_y(fig, axes_list, ncols)
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
    if leg_bb.y0 < fig_pad:
        fig.subplots_adjust(bottom=fig.subplotpars.bottom + fig_pad - leg_bb.y0)
