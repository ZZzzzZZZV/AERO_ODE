from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import torch

from script.config import OUTPUT_ROOT, PLOT_LEADS, SURFACE_VARS


def _align_hw(pred: torch.Tensor, truth: torch.Tensor):
    if pred.shape[-2:] != truth.shape[-2:]:
        pred = pred.transpose(-1, -2)
    return pred, truth


def plot_mslp_t2m(
    predicted: torch.Tensor,
    truth: torch.Tensor,
    model_name: str,
    lead_hours=None,
    output_dir: Optional[Path] = None,
):
    """Plot MSLP and T2m maps (prediction vs HRRR) for selected lead times."""
    output_dir = Path(output_dir or OUTPUT_ROOT / "maps")
    output_dir.mkdir(parents=True, exist_ok=True)
    lead_hours = lead_hours or PLOT_LEADS

    pred, gt = _align_hw(predicted.detach().cpu(), truth.detach().cpu())
    var_cfg = {
        "mslp": {"title": "MSLP", "unit": "Pa", "cmap": "RdYlBu_r"},
        "t2m": {"title": "T2m", "unit": "K", "cmap": "RdYlBu_r"},
    }

    saved = []
    for key, ch in SURFACE_VARS.items():
        cfg = var_cfg[key]
        valid = [t for t in lead_hours if t < pred.shape[1]]
        if not valid:
            continue

        n_rows = len(valid)
        fig, axes = plt.subplots(n_rows, 2, figsize=(12, 3.5 * n_rows), constrained_layout=True)
        if n_rows == 1:
            axes = axes[None, :]

        all_pred = pred[0, valid, ch]
        all_gt = gt[0, valid, ch]
        vmin = min(all_pred.min(), all_gt.min()).item()
        vmax = max(all_pred.max(), all_gt.max()).item()

        for row, t_step in enumerate(valid):
            p_data = pred[0, t_step, ch]
            g_data = gt[0, t_step, ch]
            rmse = torch.sqrt(torch.mean((p_data - g_data) ** 2)).item()

            for col, data, title in (
                (0, p_data, f"{model_name} (+{t_step}h) | RMSE={rmse:.3f}"),
                (1, g_data, f"HRRR (+{t_step}h)"),
            ):
                ax = axes[row, col]
                im = ax.imshow(data, cmap=cfg["cmap"], vmin=vmin, vmax=vmax, origin="lower")
                ax.set_title(title)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cfg["unit"])

        fig.suptitle(f"{model_name} — {cfg['title']}", fontsize=14)
        out_path = output_dir / f"{model_name.lower().replace('-', '_')}_{key}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        saved.append(out_path)
    return saved
