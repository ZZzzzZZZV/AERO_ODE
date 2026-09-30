from __future__ import annotations

import os

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from Models_FiLM import FiLM_UNet, FiLM_UNet3Plus

class DualTimeEmbedding(nn.Module):

    def __init__(self, max_steps=48, total_dim=128):
        super().__init__()
        self.step_dim = total_dim // 2
        self.step_emb = nn.Embedding(max_steps, self.step_dim)
        cyclic_dim = total_dim - self.step_dim
        self.cyclic_proj = nn.Sequential(
            nn.Linear(4, cyclic_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cyclic_dim, cyclic_dim),
        )

    def forward(self, steps, hours, months):
        steps = torch.clamp(steps, max=self.step_emb.num_embeddings - 1)
        e_step = self.step_emb(steps)
        hours = hours.float()
        months = months.float()
        cyclic = torch.stack(
            [
                torch.sin(2 * np.pi * hours / 24.0),
                torch.cos(2 * np.pi * hours / 24.0),
                torch.sin(2 * np.pi * months / 12.0),
                torch.cos(2 * np.pi * months / 12.0),
            ],
            dim=1,
        )
        return torch.cat([e_step, self.cyclic_proj(cyclic)], dim=1)

class CorrectionModel(nn.Module):

    def __init__(
        self,
        in_channels_dynamic=70,
        in_channels_static=5,
        out_channels=25,
        time_emb_dim=128,
        backbone="unet3plus",
        use_checkpoint=False,
    ):
        super().__init__()
        self.time_embedder = DualTimeEmbedding(max_steps=48, total_dim=time_emb_dim)
        total_in = in_channels_dynamic + in_channels_static
        if backbone == "unet3plus":
            self.backbone = FiLM_UNet3Plus(
                total_in,
                out_channels,
                time_emb_dim,
                use_checkpoint=use_checkpoint,
            )
        else:
            self.backbone = FiLM_UNet(total_in, out_channels, time_emb_dim)

    def forward(self, input_dynamic, input_static, steps, hours, months):
        _, _, height, width = input_dynamic.shape
        time_embedding = self.time_embedder(steps, hours, months)
        full_input = torch.cat([input_dynamic, input_static], dim=1)

        pad_h = (16 - height % 16) % 16
        pad_w = (16 - width % 16) % 16
        if pad_h > 0 or pad_w > 0:
            padding = (
                pad_w // 2,
                pad_w - pad_w // 2,
                pad_h // 2,
                pad_h - pad_h // 2,
            )
            padded = F.pad(full_input, padding, mode="reflect")
        else:
            padded = full_input
            padding = (0, 0, 0, 0)

        output = self.backbone(padded, time_embedding)
        if pad_h > 0 or pad_w > 0:
            output = output[
                :,
                :,
                padding[2] : output.shape[2] - padding[3] or None,
                padding[0] : output.shape[3] - padding[1] or None,
            ]
        return output

class StaticDataManager:

    def __init__(self, base_path, stat_dir=None, device="cuda"):
        stat_dir = stat_dir or os.path.join(base_path, "data")
        self.mean = torch.from_numpy(
            np.load(os.path.join(stat_dir, "mean_39.npy"))
        ).float().to(device)
        self.std = torch.from_numpy(
            np.load(os.path.join(stat_dir, "std_39.npy"))
        ).float().to(device)
        self.target_std = self.std[:25].view(1, 25, 1, 1)

        with h5py.File(
            os.path.join(base_path, "data", "west_geo_interpolate.h5"),
            "r",
        ) as handle:
            geo = (
                torch.from_numpy(handle["fields"][:])
                .float()
                .permute(0, 2, 1)
                .unsqueeze(0)
            )
        self.geo = ((geo - geo.mean()) / (geo.std() + 1e-6)).to(device)

        lats = torch.from_numpy(
            np.load(os.path.join(base_path, "data", "hrrr_west_lat.npy"))
        ).float().T
        lons = torch.from_numpy(
            np.load(os.path.join(base_path, "data", "hrrr_west_lon.npy"))
        ).float().T
        self.pos_emb = torch.stack(
            [
                torch.sin(torch.deg2rad(lats)),
                torch.cos(torch.deg2rad(lats)),
                torch.sin(torch.deg2rad(lons)),
                torch.cos(torch.deg2rad(lons)),
            ],
            dim=0,
        ).unsqueeze(0).to(device)
        self._static = torch.cat([self.geo, self.pos_emb], dim=1)

    def get_static_input(self, batch_size):
        return self._static.expand(batch_size, -1, -1, -1)

def compute_time_indices(time_list, total_steps, device):
    steps, hours, months = [], [], []
    for time_string in time_list:
        date_range = pd.date_range(
            start=pd.to_datetime(time_string),
            periods=total_steps,
            freq="h",
        )
        steps.append(np.arange(total_steps))
        hours.append(date_range.hour.values)
        months.append(date_range.month.values)
    return (
        torch.from_numpy(np.concatenate(steps)).long().to(device),
        torch.from_numpy(np.concatenate(hours)).long().to(device),
        torch.from_numpy(np.concatenate(months)).long().to(device),
    )
