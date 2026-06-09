#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
T-statistic significance heatmap — ED(Rb) and WD(Wb) side by side.

Heatmap color: paired t-statistic (fixed range ±30).
Black dots: p_value < alpha.

Default inputs:
  Rb: .../T_test_rb/rmse_ttest_results_2024.npz
  Wb: .../T_test_wb/rmse_ttest_results_2024_wb.npz

Run:
  python draw_significance_heatmap_dual_region.py
  python draw_significance_heatmap_dual_region.py --comparison nwp
  python draw_significance_heatmap_dual_region.py --comparison neuralgcm --steps 48
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

SCRIPT_DIR = Path(__file__).resolve().parent
FONT_PATH = SCRIPT_DIR / "Times New Roman.ttf"

FONT_SIZES = {
    "title": 20,
    "axis_label": 18,
    "tick": 15,
    "cbar_label": 18,
    "cbar_tick": 15,
    "legend": 16,
    "shade_note": 14,
}

EXCLUDED_50HPA_INDICES = frozenset({0, 4, 8, 12, 16})
T_STAT_COLOR_LIMIT = 30.0

VARIABLE_LABELS = [
    "Z50", "Z500", "Z850", "Z1000",
    "T50", "T500", "T850", "T1000",
    "S50", "S500", "S850", "S1000",
    "U50", "U500", "U850", "U1000",
    "V50", "V500", "V850", "V1000",
    "MSLP", "U10", "V10", "T2M",
]

COMPARISONS = {
    "nwp": {
        "t_key": "aero_vs_nwp_t_stat",
        "p_key": "aero_vs_nwp_p_value",
        "include_surface": True,
    },
    "neuralgcm": {
        "t_key": "aero_vs_neuralgcm_t_stat",
        "p_key": "aero_vs_neuralgcm_p_value",
        "include_surface": False,
    },
}

DEFAULT_RB_NPZ = Path(
    "./T_test_rb/rmse_ttest_results_2024.npz"
)
DEFAULT_WB_NPZ = Path(
    "./T_test_wb/rmse_ttest_results_2024_wb.npz"
)

N_AIR_ROWS = 15
N_SURFACE_ROWS = 4
AIR_FIG_HEIGHT = 6.6
SURFACE_EXTRA_HEIGHT = 2.4

REGION_PANELS = (
    ("rb", "ED", "Eastern Domain (ED)"),
    ("wb", "WD", "Western Domain (WD)"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw dual-region t-statistic heatmaps (ED + WD).")
    parser.add_argument("--rb-input", type=Path, default=DEFAULT_RB_NPZ)
    parser.add_argument("--wb-input", type=Path, default=DEFAULT_WB_NPZ)
    parser.add_argument("--output-dir", type=Path, default=Path("figures_significance_dual"))
    parser.add_argument("--comparison", choices=("all", "nwp", "neuralgcm"), default="all")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--shade-hours", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-pdf", action="store_true")
    return parser.parse_args()


def setup_font() -> None:
    if FONT_PATH.exists():
        font_manager.fontManager.addfont(str(FONT_PATH))
        font_name = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
        plt.rcParams["font.family"] = font_name
    else:
        print(f"[WARN] Font file not found, using default font: {FONT_PATH}")
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.linewidth"] = 1.2
    plt.rcParams["xtick.major.width"] = 1.1
    plt.rcParams["ytick.major.width"] = 1.1
    plt.rcParams["xtick.major.size"] = 5.0
    plt.rcParams["ytick.major.size"] = 5.0


def scatter_s_to_legend_markersize(scatter_s: float) -> float:
    return float(np.sqrt(scatter_s))


def comparisons_to_run(name: str) -> list[str]:
    if name == "all":
        return ["nwp", "neuralgcm"]
    return [name]


def select_plot_variables(include_surface: bool) -> tuple[list[str], list[int]]:
    n_vars = 24 if include_surface else 20
    indices = [i for i in range(n_vars) if i not in EXCLUDED_50HPA_INDICES]
    labels = [VARIABLE_LABELS[i] for i in indices]
    return labels, indices


def dual_fig_height(n_rows: int, include_surface: bool) -> float:
    if include_surface:
        fig_h = AIR_FIG_HEIGHT + SURFACE_EXTRA_HEIGHT * (n_rows - N_AIR_ROWS) / max(N_SURFACE_ROWS, 1)
        if n_rows <= N_AIR_ROWS:
            fig_h = AIR_FIG_HEIGHT
    else:
        fig_h = AIR_FIG_HEIGHT
    return fig_h


def load_panel_arrays(
    npz_path: Path,
    comparison: str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    cfg = COMPARISONS[comparison]
    with np.load(npz_path, allow_pickle=True) as data:
        t_stat = data[cfg["t_key"]]
        p_values = data[cfg["p_key"]]

    labels, var_indices = select_plot_variables(cfg["include_surface"])
    return t_stat[:, var_indices], p_values[:, var_indices], labels


def plot_dual_heatmap(
    *,
    rb_path: Path,
    wb_path: Path,
    comparison: str,
    output_base: Path,
    alpha: float,
    steps: int,
    shade_hours: int,
    dpi: int,
    show: bool,
    save_pdf: bool,
) -> bool:
    cfg = COMPARISONS[comparison]
    panels: list[tuple[np.ndarray, np.ndarray, list[str], str]] = []
    paths = {"rb": rb_path, "wb": wb_path}

    for region_key, _tag, title in REGION_PANELS:
        npz_path = paths[region_key]
        if not npz_path.exists():
            print(f"[WARN] Skip {region_key}: file not found -> {npz_path}")
            return False
        try:
            t_plot, p_plot, labels = load_panel_arrays(npz_path, comparison)
            panels.append((t_plot, p_plot, labels, title))
        except Exception as exc:
            print(f"[WARN] Skip {region_key} ({comparison}): {exc}")
            return False

    if len(panels) != 2:
        return False

    labels = panels[0][2]
    if panels[1][2] != labels:
        print(f"[WARN] Y labels differ Rb/Wb for {comparison}; using Rb labels")

    n_rows = len(labels)
    fig_h = dual_fig_height(n_rows, cfg["include_surface"])
    fig_w = 13.2
    fig = plt.figure(figsize=(fig_w, fig_h), layout="constrained")
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.0, 0.05], wspace=0.10)

    axes: list[plt.Axes] = []
    im_last = None
    dot_size = 12 if n_rows > 4 else 20
    tick_step = 6 if steps <= 48 else 12
    x_ticks = np.arange(0, steps, tick_step)

    for col, (t_stat, p_values, _lbl, title) in enumerate(panels):
        heatmap_plot = t_stat[:steps, :].T
        p_plot = p_values[:steps, :].T
        significant = np.isfinite(p_plot) & (p_plot < alpha)

        ax = fig.add_subplot(gs[0, col])
        axes.append(ax)
        im_last = ax.imshow(
            heatmap_plot,
            aspect="auto",
            origin="upper",
            cmap="RdBu_r",
            vmin=-T_STAT_COLOR_LIMIT,
            vmax=T_STAT_COLOR_LIMIT,
            interpolation="nearest",
        )

        if shade_hours > 0:
            ax.axvspan(
                -0.5,
                min(shade_hours, steps) - 0.5,
                color="#D9D9D9",
                alpha=0.58,
                zorder=3,
            )
            if col == 0:
                ax.text(
                    max(0.2, shade_hours / 2.0 - 0.5),
                    -0.72,
                    f"first {shade_hours} h shaded",
                    ha="center",
                    va="bottom",
                    fontsize=FONT_SIZES["shade_note"],
                    color="#555555",
                    zorder=5,
                )

        yy, xx = np.where(significant & np.isfinite(heatmap_plot))
        if xx.size > 0:
            ax.scatter(
                xx,
                yy,
                s=dot_size,
                c="black",
                marker=".",
                linewidths=0,
                alpha=0.85,
                zorder=6,
            )

        ax.set_xticks(x_ticks)
        ax.set_xticklabels([str(i + 1) for i in x_ticks])
        ax.tick_params(axis="x", labelsize=FONT_SIZES["tick"])
        if col == 0:
            ax.set_ylabel("Variable", fontsize=FONT_SIZES["axis_label"])
            ax.set_yticks(np.arange(len(labels)))
            ax.set_yticklabels(labels, fontsize=FONT_SIZES["tick"])
        else:
            ax.set_yticks(np.arange(len(labels)))
            ax.set_yticklabels([""] * len(labels))
            ax.tick_params(axis="y", length=0)

        ax.set_title(title, fontsize=FONT_SIZES["axis_label"], pad=10)
        ax.grid(False)

    axes[0].set_xlabel("Lead time (h)", fontsize=FONT_SIZES["axis_label"])
    axes[1].set_xlabel("Lead time (h)", fontsize=FONT_SIZES["axis_label"])

    cax = fig.add_subplot(gs[0, 2])
    cbar = fig.colorbar(im_last, cax=cax)
    cbar.set_label("Paired t-statistic", fontsize=FONT_SIZES["cbar_label"])
    cbar.ax.tick_params(labelsize=FONT_SIZES["cbar_tick"])

    dot_legend = Line2D(
        [0],
        [0],
        linestyle="none",
        marker=".",
        markersize=scatter_s_to_legend_markersize(dot_size),
        color="black",
        label=f"p < {alpha:g}",
    )
    air_only = not cfg["include_surface"] or n_rows <= N_AIR_ROWS
    if air_only:
        fig.canvas.draw()
        panel_bottom = min(ax.get_position().y0 for ax in axes)
        fig.legend(
            handles=[dot_legend],
            loc="upper center",
            bbox_to_anchor=(0.46, panel_bottom - 0.082),
            bbox_transform=fig.transFigure,
            fontsize=FONT_SIZES["legend"],
            frameon=True,
            ncol=1,
        )
    else:
        fig.legend(
            handles=[dot_legend],
            loc="lower center",
            bbox_to_anchor=(0.46, -0.06),
            fontsize=FONT_SIZES["legend"],
            frameon=True,
            ncol=1,
        )

    output_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_base.with_suffix(".png")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {png_path}")
    if save_pdf:
        pdf_path = output_base.with_suffix(".pdf")
        fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {pdf_path}")

    if show:
        plt.show()
    plt.close(fig)
    return True


def main() -> None:
    args = parse_args()
    setup_font()

    total_ok = 0
    for comparison in comparisons_to_run(args.comparison):
        stem = f"heatmap_tstat_{args.steps}h_{comparison}_ed_wd"
        if args.shade_hours > 0:
            stem += f"_shade{args.shade_hours}h"
        output_base = args.output_dir / stem
        if plot_dual_heatmap(
            rb_path=args.rb_input,
            wb_path=args.wb_input,
            comparison=comparison,
            output_base=output_base,
            alpha=args.alpha,
            steps=args.steps,
            shade_hours=args.shade_hours,
            dpi=args.dpi,
            show=args.show,
            save_pdf=not args.no_pdf,
        ):
            total_ok += 1

    if total_ok == 0:
        raise SystemExit("Plot failed: check Rb/Wb npz paths and contents")
    print(f"[INFO] Saved {total_ok} dual-panel figures -> {args.output_dir}")


if __name__ == "__main__":
    main()
