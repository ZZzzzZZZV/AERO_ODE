"""
Draw p-value curves from rmse_ttest.py results.

Run:
  python draw_pvalue_curves.py --input rmse_ttest_results_2024.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
FONT_PATH = SCRIPT_DIR / "Times New Roman.ttf"

FIGURE_SIZE = (22, 24)
FONT_SIZES = {
    "suptitle": 35,
    "panel_title": 20,
    "axis_label": 18,
    "tick": 15,
    "no_data": 18,
}

DEFAULT_VARIABLE_NAMES = [
    "z50",
    "z500",
    "z850",
    "z1000",
    "t50",
    "t500",
    "t850",
    "t1000",
    "s50",
    "s500",
    "s850",
    "s1000",
    "u50",
    "u500",
    "u850",
    "u1000",
    "v50",
    "v500",
    "v850",
    "v1000",
    "mslp",
    "u10",
    "v10",
    "t2m",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw 6x4 p-value curve panels.")
    parser.add_argument("--input", type=Path, default=Path("rmse_ttest_results_2024.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures_rb"))
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--log-y", action="store_true", help="Use log scale for the p-value axis.")
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


def load_variable_names(npz_file: np.lib.npyio.NpzFile) -> list[str]:
    if "variable_names" not in npz_file:
        return DEFAULT_VARIABLE_NAMES
    names = npz_file["variable_names"].tolist()
    return [str(name) for name in names]


def plot_pvalue_panel(
    p_values: np.ndarray,
    variable_names: list[str],
    title: str,
    save_path: Path,
    alpha: float = 0.05,
    show: bool = False,
    log_y: bool = False,
) -> None:
    if p_values.shape != (48, 24):
        raise ValueError(f"{title}: expected p-value shape (48, 24), got {p_values.shape}")
    if len(variable_names) != 24:
        raise ValueError(f"expected 24 variable names, got {len(variable_names)}")

    leads = np.arange(1, 49)
    fig, axes = plt.subplots(6, 4, figsize=FIGURE_SIZE, sharex=True, sharey=True)
    axes_flat = axes.ravel()

    for var_idx, ax in enumerate(axes_flat):
        series = p_values[:, var_idx]
        valid = np.isfinite(series)
        ax.axhline(alpha, color="tab:red", linestyle="--", linewidth=1.8, label=f"p={alpha:g}")

        if valid.any():
            ax.plot(leads[valid], series[valid], color="tab:blue", linewidth=2.2)
        else:
            ax.text(
                0.5,
                0.5,
                "No data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=FONT_SIZES["no_data"],
            )

        ax.set_title(variable_names[var_idx], fontsize=FONT_SIZES["panel_title"], pad=8)
        ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
        ax.set_xlim(1, 48)
        ax.tick_params(axis="both", labelsize=FONT_SIZES["tick"])
        if log_y:
            ax.set_yscale("log")
            ax.set_ylim(1e-6, 1.0)
        else:
            ax.set_ylim(0.0, 1.0)

    for ax in axes[-1, :]:
        ax.set_xlabel("Forecast lead time (hour)", fontsize=FONT_SIZES["axis_label"])
    for ax in axes[:, 0]:
        ax.set_ylabel("p-value", fontsize=FONT_SIZES["axis_label"])

    fig.suptitle(title, fontsize=FONT_SIZES["suptitle"], y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975), h_pad=2.0, w_pad=1.0)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300)
    print(f"[saved] {save_path}")

    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = parse_args()
    setup_font()

    with np.load(args.input, allow_pickle=True) as data:
        variable_names = load_variable_names(data)
        aero_vs_nwp = data["aero_vs_nwp_p_value"]
        aero_vs_neuralgcm = data["aero_vs_neuralgcm_p_value"]

    print(f"[input] aero_vs_nwp_p_value shape: {aero_vs_nwp.shape}")
    print(f"[input] aero_vs_neuralgcm_p_value shape: {aero_vs_neuralgcm.shape}")

    plot_pvalue_panel(
        aero_vs_nwp,
        variable_names,
        "P-value Curves: AERO-ODE vs NWP",
        args.output_dir / "pvalue_aero_vs_nwp.png",
        alpha=args.alpha,
        show=args.show,
        log_y=args.log_y,
    )
    plot_pvalue_panel(
        aero_vs_neuralgcm,
        variable_names,
        "P-value Curves: AERO-ODE vs NeuralGCM",
        args.output_dir / "pvalue_aero_vs_neuralgcm.png",
        alpha=args.alpha,
        show=args.show,
        log_y=args.log_y,
    )


if __name__ == "__main__":
    main()
