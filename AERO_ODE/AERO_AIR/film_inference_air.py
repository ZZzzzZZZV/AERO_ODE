import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F

from Models_FiLM import FiLM_UNet, FiLM_UNet3Plus


DT_DYNAMICS_S: float = 3600.0


def _resolved_dynamics_tendency(state_pre, state_post, dt):
    return (state_post - state_pre) / dt


def _subgrid_tendency(increment, dt):
    return increment / dt


class StrangSplittingIntegrator:
    """Symmetric 2nd-order operator-splitting integrator."""

    def __init__(self, dt: float = DT_DYNAMICS_S,
                 internal_dtype: torch.dtype = torch.float64):
        self.dt = float(dt)
        self.internal_dtype = internal_dtype

    @property
    def order(self) -> int:
        return 2

    def _half_dynamics_step(self, state, F_dyn):
        return state + 0.5 * self.dt * F_dyn

    def _full_subgrid_step(self, state, F_sgs):
        return state + self.dt * F_sgs

    def step(self, state_pre, state_post, increment):
        orig_dtype = state_pre.dtype
        x_pre = state_pre.to(self.internal_dtype)
        x_post = state_post.to(self.internal_dtype)
        delta = increment.to(self.internal_dtype)

        F_dyn = _resolved_dynamics_tendency(x_pre, x_post, self.dt)
        F_sgs = _subgrid_tendency(delta, self.dt)

        x = self._half_dynamics_step(x_pre, F_dyn)
        x = self._full_subgrid_step(x, F_sgs)
        x = self._half_dynamics_step(x, F_dyn)
        return x.to(orig_dtype)


class DualTimeEmbedding(nn.Module):
    """Dual time embedding: step index + cyclic time encoding"""

    def __init__(self, max_steps=48, total_dim=128):
        super().__init__()
        self.step_dim = total_dim // 2
        self.step_emb = nn.Embedding(max_steps, self.step_dim)

        cyclic_dim = total_dim - self.step_dim
        self.cyclic_proj = nn.Sequential(
            nn.Linear(4, cyclic_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cyclic_dim, cyclic_dim)
        )

    def forward(self, steps, hours, months):
        steps = torch.clamp(steps, max=self.step_emb.num_embeddings - 1)
        e_step = self.step_emb(steps)

        h, m = hours.float(), months.float()
        cyclic = torch.stack([
            torch.sin(2 * np.pi * h / 24.0),
            torch.cos(2 * np.pi * h / 24.0),
            torch.sin(2 * np.pi * m / 12.0),
            torch.cos(2 * np.pi * m / 12.0)
        ], dim=1)
        e_cyclic = self.cyclic_proj(cyclic)

        return torch.cat([e_step, e_cyclic], dim=1)


class CorrectionModel(nn.Module):
    """Meteorological field correction model"""

    def __init__(self, in_channels_dynamic=56, in_channels_static=5, out_channels=20,
                 time_emb_dim=128, backbone='unet3plus', use_checkpoint=False):
        super().__init__()
        self.time_embedder = DualTimeEmbedding(max_steps=48, total_dim=time_emb_dim)

        total_in = in_channels_dynamic + in_channels_static
        if backbone == 'unet3plus':
            self.backbone = FiLM_UNet3Plus(total_in, out_channels, time_emb_dim,
                                           use_checkpoint=use_checkpoint)
        else:
            self.backbone = FiLM_UNet(total_in, out_channels, time_emb_dim)

    def forward(self, input_dynamic, input_static, steps, hours, months):
        t_vec = self.time_embedder(steps, hours, months)
        full_input = torch.cat([input_dynamic, input_static], dim=1)

        _, _, H, W = input_dynamic.shape
        pad_h = (16 - H % 16) % 16
        pad_w = (16 - W % 16) % 16
        if pad_h > 0 or pad_w > 0:
            padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
            x_padded = F.pad(full_input, padding, mode='reflect')
        else:
            x_padded = full_input
            padding = (0, 0, 0, 0)

        output_padded = self.backbone(x_padded, t_vec)

        if pad_h > 0 or pad_w > 0:
            output = output_padded[:, :, padding[2]:output_padded.shape[2]-padding[3] or None,
                                   padding[0]:output_padded.shape[3]-padding[1] or None]
        else:
            output = output_padded
        return output


class StaticDataManager:
    """Static data manager: terrain, position encoding, statistics"""

    def __init__(self, base_path, device='cuda'):
        self.device = device

        self.mean = torch.from_numpy(
            np.load(os.path.join(base_path, 'stat/mean_crop.npy'))
        ).float().to(device)
        self.std = torch.from_numpy(
            np.load(os.path.join(base_path, 'stat/std_crop.npy'))
        ).float().to(device)
        self.target_std = self.std[:20].view(1, 20, 1, 1)

        with h5py.File(os.path.join(base_path, 'geo.h5'), "r") as f:
            geo = torch.from_numpy(f["fields"][:]).float().permute(0, 2, 1).unsqueeze(0)
            self.geo = ((geo - geo.mean()) / (geo.std() + 1e-6)).to(device)

        lats = torch.from_numpy(np.load(os.path.join(base_path, 'lats.npy'))).float().T
        lons = torch.from_numpy(np.load(os.path.join(base_path, 'lons.npy'))).float().T
        self.pos_emb = torch.stack([
            torch.sin(torch.deg2rad(lats)), torch.cos(torch.deg2rad(lats)),
            torch.sin(torch.deg2rad(lons)), torch.cos(torch.deg2rad(lons))
        ], dim=0).unsqueeze(0).to(device)

        self._static = torch.cat([self.geo, self.pos_emb], dim=1)

    def get_static_input(self, batch_size):
        return self._static.expand(batch_size, -1, -1, -1)


def compute_time_indices(time_list, total_steps, device):
    steps, hours, months = [], [], []
    for t_str in time_list:
        dt = pd.to_datetime(t_str)
        rng = pd.date_range(start=dt, periods=total_steps, freq='h')
        steps.append(np.arange(total_steps))
        hours.append(rng.hour.values)
        months.append(rng.month.values)

    return (
        torch.from_numpy(np.concatenate(steps)).long().to(device),
        torch.from_numpy(np.concatenate(hours)).long().to(device),
        torch.from_numpy(np.concatenate(months)).long().to(device)
    )
