#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AERO-ODE RMSE significance heatmap.

Default heatmap color is paired t-statistic from:
    stats.ttest_rel(aero_rmse, baseline_rmse)

Therefore:
    t < 0 means AERO-ODE has lower RMSE than the baseline.
    t > 0 means AERO-ODE has higher RMSE than the baseline.

Black dots mark paired t-test significance: p_value < alpha.

Run:
  python draw_significance_heatmap.py --input rmse_ttest_results_2024.npz
  python draw_significance_heatmap.py --comparison nwp
  python draw_significance_heatmap.py --comparison neuralgcm --no-pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
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

# 50 hPa variables in the 24-channel order (z/t/s/u/v at index 0,4,8,12,16).
EXCLUDED_50HPA_INDICES = frozenset({0, 4, 8, 12, 16})
T_STAT_COLOR_LIMIT = 30.0

VARIABLE_LABELS = [
    "Z50",
    "Z500",
    "Z850",
    "Z1000",
    "T50",
    "T500",
    "T850",
    "T1000",
    "S50",
    "S500",
    "S850",
    "S1000",
    "U50",
    "U500",
    "U850",
    "U1000",
    "V50",
    "V500",
    "V850",
    "V1000",
    "MSLP",
    "U10",
    "V10",
    "T2M",
]

COMPARISONS = {
    "nwp": {
        "t_key": "aero_vs_nwp_t_stat",
        "p_key": "aero_vs_nwp_p_value",
        "label": "AERO-ODE vs NWP",
        "stem": "significance_heatmap_aero_vs_nwp",
        "include_surface": True,
    },
    "neuralgcm": {
        "t_key": "aero_vs_neuralgcm_t_stat",
        "p_key": "aero_vs_neuralgcm_p_value",
        "label": "AERO-ODE vs NeuralGCM",
        "stem": "significance_heatmap_aero_vs_neuralgcm",
        "include_surface": False,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw t-statistic heatmap with p-value significance dots.")
    parser.add_argument("--input", type=Path, default=Path("rmse_ttest_results_2024_wb.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures_rb/significance_heatmap"))
    parser.add_argument("--comparison", choices=("all", "nwp", "neuralgcm"), default="all")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--shade-hours", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-pdf", action="store_true")
    parser.add_argument("--title", action="store_true", help="Show a figure title.")
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


def select_plot_variables(include_surface: bool) -> tuple[list[str], list[int]]:
    """Return y-axis labels and source column indices, excluding 50 hPa variables."""
    n_vars = 24 if include_surface else 20
    indices = [i for i in range(n_vars) if i not in EXCLUDED_50HPA_INDICES]
    labels = [VARIABLE_LABELS[i] for i in indices]
    return labels, indices


def plot_heatmap(
    heatmap_values: np.ndarray,
    p_values: np.ndarray,
    labels: list[str],
    comparison_label: str,
    output_base: Path,
    *,
    alpha: float,
    steps: int,
    shade_hours: int,
    dpi: int,
    show: bool,
    save_pdf: bool,
    show_title: bool,
) -> None:
    if heatmap_values.shape != p_values.shape:
        raise ValueError(f"shape mismatch: heatmap_values={heatmap_values.shape}, p_values={p_values.shape}")
    if heatmap_values.shape[0] < steps:
        raise ValueError(f"requested {steps} steps, but data has only {heatmap_values.shape[0]}")

    heatmap_plot = heatmap_values[:steps, : len(labels)].T
    p_plot = p_values[:steps, : len(labels)].T
    significant = np.isfinite(p_plot) & (p_plot < alpha)

    fig_h = 7.8 if len(labels) > 4 else 3.2
    fig, ax = plt.subplots(figsize=(12.5, fig_h))

    im = ax.imshow(
        heatmap_plot,
        aspect="auto",
        origin="upper",
        cmap="RdBu_r",
        vmin=-T_STAT_COLOR_LIMIT,
        vmax=T_STAT_COLOR_LIMIT,
        interpolation="nearest",
    )

    if shade_hours > 0:
        ax.axvspan(-0.5, min(shade_hours, steps) - 0.5, color="#D9D9D9", alpha=0.58, zorder=3)
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

    dot_size = 12 if len(labels) > 4 else 20
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

    tick_step = 6 if steps <= 48 else 12
    x_ticks = np.arange(0, steps, tick_step)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(i + 1) for i in x_ticks])
    ax.set_xlabel("Lead time (h)", fontsize=FONT_SIZES["axis_label"])
    ax.tick_params(axis="x", labelsize=FONT_SIZES["tick"])

    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=FONT_SIZES["tick"])
    ax.set_ylabel("Variable", fontsize=FONT_SIZES["axis_label"])

    if show_title:
        ax.set_title(comparison_label, fontsize=FONT_SIZES["title"])

    cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.94)
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
    ax.legend(
        handles=[dot_legend],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        fontsize=FONT_SIZES["legend"],
        frameon=True,
    )
    ax.grid(False)
    plt.tight_layout()

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


def comparisons_to_run(name: str) -> list[str]:
    if name == "all":
        return ["nwp", "neuralgcm"]
    return [name]


def main() -> None:
    args = parse_args()
    setup_font()

    with np.load(args.input, allow_pickle=True) as data:
        loaded = {key: data[key] for key in data.files}

    for comparison in comparisons_to_run(args.comparison):
        cfg = COMPARISONS[comparison]
        t_stat = loaded[cfg["t_key"]]
        p_values = loaded[cfg["p_key"]]

        labels, var_indices = select_plot_variables(cfg["include_surface"])
        t_stat_plot = t_stat[:, var_indices]
        p_values_plot = p_values[:, var_indices]
        output_base = args.output_dir / cfg["stem"]
        plot_heatmap(
            t_stat_plot,
            p_values_plot,
            labels,
            cfg["label"],
            output_base,
            alpha=args.alpha,
            steps=args.steps,
            shade_hours=args.shade_hours,
            dpi=args.dpi,
            show=args.show,
            save_pdf=not args.no_pdf,
            show_title=args.title,
        )


if __name__ == "__main__":
    main()
