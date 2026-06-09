"""
Cross-platform path configuration for Draw_New (Windows / Ubuntu).

- repo(): data inside this AERO_ODE_Draw repo (Rb_RMSE, Hrrr_rb, etc.)
- draw_new(): resources under Draw_New (fonts, output figures, etc.)
- sibling(): sibling project directories (optional external code)
- nfs(): large forecast data; override mount via NEURALGCM_NFS_ROOT

Lat/lon (relative paths in repo):
  East (rb): Hrrr_rb/lats.npy, Hrrr_rb/lons.npy
  West (wb): Hrrr_wb/hrrr_west_lat.npy, Hrrr_wb/hrrr_west_lon.npy
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DRAW_NEW = REPO_ROOT / "Draw_New"
CODE_ROOT = REPO_ROOT.parent
# Default NFS mount on the data-fusion cluster (override with NEURALGCM_NFS_ROOT)
_DEFAULT_NFS = "/nfs/samba/数据聚变/气象数据"
NFS_ROOT = Path(os.environ.get("NEURALGCM_NFS_ROOT", _DEFAULT_NFS))


def repo(*parts: str) -> Path:
    """Relative path under repo root."""
    return REPO_ROOT.joinpath(*parts)


def draw_new(*parts: str) -> Path:
    """Relative path under Draw_New."""
    return DRAW_NEW.joinpath(*parts)


def sibling(*parts: str) -> Path:
    """Sibling project directory."""
    return CODE_ROOT.joinpath(*parts)


def nfs(*parts: str) -> Path:
    """Large forecast data on NFS/shared storage (set NEURALGCM_NFS_ROOT)."""
    return NFS_ROOT.joinpath(*parts)


def coords_rb() -> tuple[Path, Path]:
    """East region (rb) lat/lon."""
    return repo("Hrrr_rb/lats.npy"), repo("Hrrr_rb/lons.npy")


def coords_wb() -> tuple[Path, Path]:
    """West region (wb) lat/lon."""
    return repo("Hrrr_wb/hrrr_west_lat.npy"), repo("Hrrr_wb/hrrr_west_lon.npy")


def font_times_new_roman() -> Path:
    """Times New Roman.ttf at repo root (fallback to system font if missing)."""
    return repo("Times New Roman.ttf")


def _env_int(name: str, default: int) -> int:
    """Read integer font size from env; overridable by run_all_drawings.py."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    """Read float layout parameter from env; overridable by run_all_drawings.py."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


# Figure font sizes (pt)
SUBPLOT_TITLE_SIZE = 18
LEGEND_FONT_SIZE = SUBPLOT_TITLE_SIZE
TICK_LABEL_SIZE = 12
AXES_LABEL_SIZE = 18
# Map subplot title size; override via DRAW_MAP_SUBPLOT_TITLE_SIZE
MAP_SUBPLOT_TITLE_SIZE = _env_int("DRAW_MAP_SUBPLOT_TITLE_SIZE", 24)
# Vertical spacing between map subplot rows (matplotlib hspace)
MAP_SUBPLOT_HSPACE = _env_float("DRAW_MAP_SUBPLOT_HSPACE", 0.25)
# Horizontal spacing between map subplot columns (matplotlib wspace)
MAP_SUBPLOT_WSPACE = _env_float("DRAW_MAP_SUBPLOT_WSPACE", 0.05)
MAP_TICK_LABEL_SIZE = 16
MAP_GRIDLABEL_SIZE = 17
MAP_AXES_LABELSIZE = 18
CBAR_TICK_LABEL_SIZE = 14
FIGURE_SAVE_DPI = 300
FIGURE_SAVE_DPI_HIGH = 350
# Extreme curve plots share font sizes with RMSE/ACC (aliases kept for legacy imports)
EXTREME_SUBPLOT_TITLE_SIZE = SUBPLOT_TITLE_SIZE
EXTREME_LEGEND_FONT_SIZE = LEGEND_FONT_SIZE
EXTREME_AXES_LABEL_SIZE = AXES_LABEL_SIZE

# Curve layout: air 5x4 cell is baseline; surface/ob 1x4 cells match width/height
AIR_FIGSIZE = (16, 15)
AIR_GRID_ROWS, AIR_GRID_COLS = 5, 4
# Combined upper-air 5x4 + surface 1x4 curve figure
COMBINED_RMSE_ROWS = AIR_GRID_ROWS + 1
COMBINED_RMSE_FIGSIZE = (
    AIR_FIGSIZE[0],
    AIR_FIGSIZE[1] * COMBINED_RMSE_ROWS / AIR_GRID_ROWS,
)
# Multi-row air 5x4: legend/x-label gap = row spacing * this scale
AIR_LEGEND_GAP_SCALE = 0.45
SURFACE_NCOLS = 4
# Curve subplot spacing (matplotlib wspace/hspace)
CURVE_SUBPLOT_WSPACE = 0.30
CURVE_SUBPLOT_HSPACE = 0.38
CURVE_SUBPLOT_LEFT = 0.057
CURVE_SUBPLOT_RIGHT = 0.99
# Bottom row y-position for air 5x4 (paired with AIR_FIGSIZE and legend)
AIR_BOTTOM_ROW_Y0 = 0.1065
AIR_BOTTOM_ROW_HEIGHT = 0.138
SURFACE_FIG_TOP_MARGIN_IN = 0.2
EXTREME_AIR_FIGSIZE = (18, 16)


def curve_subplot_column_layout(
    ncols: int = SURFACE_NCOLS,
    wspace: float = CURVE_SUBPLOT_WSPACE,
    left: float = CURVE_SUBPLOT_LEFT,
    right: float = CURVE_SUBPLOT_RIGHT,
) -> tuple[tuple[float, ...], float]:
    """Compute subplot x0/width for n columns from wspace (figure coords 0-1)."""
    span = right - left
    width = span / (ncols + (ncols - 1) * wspace)
    x0s: list[float] = []
    x = left
    for _ in range(ncols):
        x0s.append(x)
        x += width * (1.0 + wspace)
    return tuple(x0s), width


def apply_curve_grid_spacing(fig) -> None:
    """Adjust row/column spacing for multi-row grids (air/ACC/Extreme)."""
    fig.subplots_adjust(
        wspace=CURVE_SUBPLOT_WSPACE,
        hspace=CURVE_SUBPLOT_HSPACE,
    )


def apply_curve_grid_wspace(fig) -> None:
    """Legacy alias."""
    apply_curve_grid_spacing(fig)


def _air_figsize_for_figure(fig) -> tuple[float, float]:
    """Infer air reference figsize from figure width (incl. Extreme 18x16)."""
    if abs(fig.get_figwidth() - EXTREME_AIR_FIGSIZE[0]) < 0.5:
        return EXTREME_AIR_FIGSIZE
    return AIR_FIGSIZE


def air_subplot_cell_inches(
    air_figsize: tuple[float, float] = AIR_FIGSIZE,
    air_rows: int = AIR_GRID_ROWS,
    air_cols: int = AIR_GRID_COLS,
) -> tuple[float, float]:
    """Width/height (inches) of one cell in air 5x4 grid."""
    w, h = air_figsize
    return w / air_cols, h / air_rows


def figsize_one_row_match_air(
    air_figsize: tuple[float, float] = AIR_FIGSIZE,
    air_rows: int = AIR_GRID_ROWS,
    air_cols: int = AIR_GRID_COLS,
    ncols: int = SURFACE_NCOLS,
) -> tuple[float, float]:
    """Single-row ncols: panel size matches air bottom row; total height includes legend band."""
    w, h_air = air_figsize
    bottom_in = AIR_BOTTOM_ROW_Y0 * h_air
    axes_h_in = AIR_BOTTOM_ROW_HEIGHT * h_air
    total_h = bottom_in + axes_h_in + SURFACE_FIG_TOP_MARGIN_IN
    return (w, total_h)


def single_row_legend_strip_frac(
    air_figsize: tuple[float, float] = AIR_FIGSIZE,
    fig_height_in: float | None = None,
) -> float:
    """Legend band height fraction for single-row figures."""
    _, h_air = air_figsize
    bottom_in = AIR_BOTTOM_ROW_Y0 * h_air
    if fig_height_in is None:
        axes_h_in = AIR_BOTTOM_ROW_HEIGHT * h_air
        fig_height_in = bottom_in + axes_h_in + SURFACE_FIG_TOP_MARGIN_IN
    return bottom_in / fig_height_in


def layout_single_row_axes_like_air(
    fig,
    axes_list: list,
    ncols: int,
    air_figsize: tuple[float, float] = AIR_FIGSIZE,
) -> None:
    """Align 1xncols subplots with air bottom row size and y-position."""
    h_fig = fig.get_figheight()
    w_fig = fig.get_figwidth()
    w_air, h_air = air_figsize
    y0 = AIR_BOTTOM_ROW_Y0 * h_air / h_fig
    height = AIR_BOTTOM_ROW_HEIGHT * h_air / h_fig
    x0s, width = curve_subplot_column_layout(ncols)
    for i, ax in enumerate(axes_list):
        ax.set_position([x0s[i], y0, width, height])


SURFACE_FIGSIZE = figsize_one_row_match_air()
EXTREME_SURFACE_FIGSIZE = figsize_one_row_match_air(EXTREME_AIR_FIGSIZE)

# Subplot titles use short variable names (no long labels)
UPPER_VAR_SHORT = ("Z", "T", "S", "U", "V")
UPPER_VAR_UNITS = ("gpm", "K", "kg/kg", "m/s", "m/s")
UPPER_LEVELS = (50, 500, 850, 1000)
SURFACE_VAR_SHORT = ("MSLP", "U10", "V10", "T2M")
XLABEL_FORECAST_TIME = "Forecast Time (hours)"


def upper_air_subplot_title(var_idx: int, level: int) -> str:
    """e.g. Z850"""
    return f"{UPPER_VAR_SHORT[var_idx]}{level}"


def rmse_air_ylabel(var_idx: int) -> str:
    """Upper-air RMSE y-axis: RMSE (units), not bold"""
    return f"RMSE({UPPER_VAR_UNITS[var_idx]})"


def _iter_axes(axes) -> list:
    if hasattr(axes, "ravel"):
        return list(axes.ravel())
    if hasattr(axes, "flat"):
        return list(axes.flat)
    return list(axes)


def _subplot_row_gap(fig, axes_list: list, nrows: int, ncols: int) -> float:
    """Vertical gap between subplot title and row above (figure coords)."""
    if nrows >= 2:
        ax_upper = axes_list[-ncols * 2]
        ax_lower = axes_list[-ncols]
        return ax_upper.get_position().y0 - ax_lower.get_position().y1
    ax = axes_list[0]
    pos = ax.get_position()
    return max(0.016, pos.height * 0.14)


def _bottom_label_bottom_y(fig, axes_list: list, ncols: int) -> float:
    """Bottom edge of lowest-row x labels/ticks (figure coords)."""
    fig.canvas.draw()
    y_min = float("inf")
    for ax in axes_list[-ncols:]:
        if ax.xaxis.label.get_text().strip():
            bb = ax.xaxis.label.get_window_extent().transformed(
                fig.transFigure.inverted()
            )
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
) -> None:
    """Shared bottom legend; gap to x-labels ~ inter-row title spacing."""
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
            handles,
            labels,
            loc="upper center",
            ncol=ncol,
            fontsize=fontsize,
            frameon=True,
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
        legend.set_bbox_to_anchor(
            (0.5, legend_y_top + dy), transform=fig.transFigure
        )
        fig.canvas.draw()
        leg_bb = legend.get_window_extent().transformed(fig.transFigure.inverted())
    if leg_bb.y0 < fig_pad and not single_row:
        fig.subplots_adjust(bottom=fig.subplotpars.bottom + fig_pad - leg_bb.y0)


def apply_paper_rcparams(*, map_style: bool = False) -> None:
    """Set global matplotlib font sizes (curves / maps)."""
    import matplotlib.pyplot as plt

    if map_style:
        plt.rcParams["axes.labelsize"] = MAP_AXES_LABELSIZE
        plt.rcParams["xtick.labelsize"] = MAP_TICK_LABEL_SIZE
        plt.rcParams["ytick.labelsize"] = MAP_TICK_LABEL_SIZE
    else:
        plt.rcParams["xtick.labelsize"] = TICK_LABEL_SIZE
        plt.rcParams["ytick.labelsize"] = TICK_LABEL_SIZE
        plt.rcParams["axes.labelsize"] = AXES_LABEL_SIZE


def setup_paper_font(*, map_style: bool = False) -> None:
    """Load Times New Roman and apply RMSE-curve font sizes."""
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    font_path = font_times_new_roman()
    try:
        fm.fontManager.addfont(str(font_path))
        font_name = fm.FontProperties(fname=str(font_path)).get_name()
        plt.rcParams["font.family"] = font_name
        plt.rcParams["axes.unicode_minus"] = False
    except Exception as exc:
        print(f"[WARN] Font load failed {font_path}: {exc}")
    apply_paper_rcparams(map_style=map_style)


def significance_heatmap_fontsizes() -> dict[str, int]:
    """Significance heatmap font sizes (match RMSE curve paths_config)。"""
    return {
        "xlabel": AXES_LABEL_SIZE,
        "ylabel": AXES_LABEL_SIZE,
        "xtick": TICK_LABEL_SIZE,
        "ytick": TICK_LABEL_SIZE,
        "cbar_label": AXES_LABEL_SIZE,
        "cbar_tick": CBAR_TICK_LABEL_SIZE,
        "legend": LEGEND_FONT_SIZE,
        "section": TICK_LABEL_SIZE,
        "shade_note": TICK_LABEL_SIZE,
    }


def suppress_figure_suptitle(fig) -> None:
    """Hide fig.suptitle; keep per-subplot ax.set_title only."""
    st = getattr(fig, "_suptitle", None)
    if st is not None:
        st.set_visible(False)


def save_viz_figure(fig, path, *, dpi: int = FIGURE_SAVE_DPI) -> None:
    """Save forecast comparison maps without overall suptitle."""
    suppress_figure_suptitle(fig)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")


def save_curve_figure(
    fig,
    path,
    *,
    dpi: int = FIGURE_SAVE_DPI_HIGH,
    bbox_inches: str = "tight",
    facecolor: str = "white",
) -> None:
    """Save RMSE/ACC/Extreme/Ob_RMSE curve figures."""
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches, facecolor=facecolor)


def ensure_draw_new_on_path(script_file: str) -> None:
    """Imported by Draw_New sub-scripts."""
    draw_new_dir = Path(script_file).resolve().parent
    for _ in range(4):
        if (draw_new_dir / "paths_config.py").exists():
            p = str(draw_new_dir)
            if p not in sys.path:
                sys.path.insert(0, p)
            return
        draw_new_dir = draw_new_dir.parent
    raise RuntimeError("paths_config.py not found above script path")
