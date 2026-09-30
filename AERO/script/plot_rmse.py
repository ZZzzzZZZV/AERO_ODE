from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from script.config import OUTPUT_ROOT
from script.plot_scores import plot_rmse_combined as plot_rmse_ab

N_AIR_CHANNELS = 25
N_SURFACE_VARS = 4
MODEL_NAME = "AERO-ODE"

def _align_hw(pred: torch.Tensor, truth: torch.Tensor):
    if pred.shape[-2:] != truth.shape[-2:]:
        pred = pred.transpose(-1, -2)
    return pred, truth

def compute_air_rmse_array(predicted: torch.Tensor, truth: torch.Tensor) -> np.ndarray:
    pred, gt = _align_hw(predicted.detach().cpu(), truth.detach().cpu())
    n_ch = min(pred.shape[2], gt.shape[2], N_AIR_CHANNELS)
    t_len = min(pred.shape[1], gt.shape[1])
    out = np.zeros((n_ch, t_len), dtype=np.float64)
    for c in range(n_ch):
        for t in range(t_len):
            out[c, t] = torch.sqrt(torch.mean((pred[0, t, c] - gt[0, t, c]) ** 2)).item()
    return out

def compute_surface_rmse_array(predicted: torch.Tensor, truth: torch.Tensor) -> np.ndarray:
    pred, gt = _align_hw(predicted.detach().cpu(), truth.detach().cpu())
    n_vars = min(pred.shape[2], gt.shape[2], N_SURFACE_VARS)
    t_len = min(pred.shape[1], gt.shape[1])
    out = np.zeros((t_len, n_vars), dtype=np.float64)
    for v in range(n_vars):
        for t in range(t_len):
            out[t, v] = torch.sqrt(torch.mean((pred[0, t, v] - gt[0, t, v]) ** 2)).item()
    return out

def _split_wrf_rmse(wrf_rmse: np.ndarray):
    wrf = np.asarray(wrf_rmse)
    if wrf.ndim != 2 or wrf.shape[1] < 29:
        raise ValueError(f"wrf_rmse expected (T, 39) or >=29 cols, got {wrf.shape}")
    air = wrf[:, :N_AIR_CHANNELS]
    surface = wrf[:, -N_SURFACE_VARS:] if wrf.shape[1] == 24 else wrf[:, 35:39]
    return air, surface

def plot_combined_rmse(
    air_pred: torch.Tensor,
    air_truth: torch.Tensor,
    surface_pred: torch.Tensor,
    surface_truth: torch.Tensor,
    output_path: Optional[Path] = None,
    model_name: str = MODEL_NAME,
    wrf_rmse: Optional[np.ndarray] = None,
    wrf_name: str = "WRF-ARW",
) -> Path:
    output_path = Path(output_path or OUTPUT_ROOT / "rmse_combined.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    air_rmse = compute_air_rmse_array(air_pred, air_truth)
    surface_rmse = compute_surface_rmse_array(surface_pred, surface_truth)
    air_list = [air_rmse]
    surf_list = [surface_rmse]
    names = [model_name]
    if wrf_rmse is not None:
        wrf_air, wrf_surface = _split_wrf_rmse(wrf_rmse)
        air_list.append(wrf_air)
        surf_list.append(wrf_surface)
        names.append(wrf_name)
    save_base = str(output_path.with_suffix(""))
    saved = plot_rmse_ab(
        air_list,
        surf_list,
        names,
        save_base=save_base,
        show=False,
    )
    air_png = next((p for p in saved if p.name.endswith("_air.png")), output_path)
    return air_png
