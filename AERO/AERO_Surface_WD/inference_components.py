from __future__ import annotations

import os
import pathlib as _pathlib
import sys as _sys

_sys.path.insert(
    0,
    str(
        next(
            parent
            for parent in _pathlib.Path(__file__).resolve().parents
            if (parent / "common" / "paths.py").is_file()
        )
    ),
)
from common.paths import ROOT

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

class DiagnosisModel(nn.Module):

    def __init__(
        self,
        in_channels_dynamic=74,
        in_channels_static=14,
        out_channels=4,
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

LEVEL_STAT_MAPPING = {
    1000: 3,
    925: 3,
    850: 2,
    700: 2,
    500: 1,
}
TARGET_LEVELS = [1000, 925, 850, 700, 500]
NGCM_VAR_TO_HRRR_BASE = {
    0: 0,
    1: 4,
    2: 8,
    3: 12,
    4: 16,
    5: -1,
    6: -1,
}

class StaticDataManager:

    def __init__(
        self,
        base_path,
        device="cuda",
        target_levels=None,
        static_data_path=None,
    ):
        self.device = device
        self.target_levels = target_levels or TARGET_LEVELS

        if static_data_path is None:
            candidates = [
                os.path.join(base_path, "data"),
                base_path,
                str(ROOT / "AERO_Surface_WD/Hrrr_rb/data"),
            ]
            static_data_path = next(
                (
                    path
                    for path in candidates
                    if os.path.exists(os.path.join(path, "slope.npy"))
                ),
                base_path,
            )
        self.static_data_path = static_data_path

        self.mean_raw = torch.from_numpy(
            np.load(os.path.join(base_path, "2015_2023_mean_west.npy"))
        ).float().to(device)
        self.std_raw = torch.from_numpy(
            np.load(os.path.join(base_path, "2015_2023_std_west.npy"))
        ).float().to(device)
        self.mean = self.mean_raw
        self.std = self.std_raw
        self.input_mean, self.input_std = self._build_input_stats()
        self.surface_mean = self.mean_raw[20:24].view(1, 4, 1, 1)
        self.surface_std = self.std_raw[20:24].view(1, 4, 1, 1)
        self.target_std = self.std_raw[:20].view(1, 20, 1, 1)

        geo_path = os.path.join(static_data_path, "west_geo_interpolate.h5")
        if not os.path.exists(geo_path):
            geo_path = os.path.join(base_path, "west_geo_interpolate.h5")
        with h5py.File(geo_path, "r") as handle:
            geo = (
                torch.from_numpy(handle["fields"][:])
                .float()
                .permute(0, 2, 1)
                .unsqueeze(0)
            )
        self.geo = ((geo - geo.mean()) / (geo.std() + 1e-6)).to(device)

        lats_path = os.path.join(static_data_path, "hrrr_west_lat.npy")
        lons_path = os.path.join(static_data_path, "hrrr_west_lon.npy")
        if not os.path.exists(lats_path):
            lats_path = os.path.join(base_path, "hrrr_west_lat.npy")
            lons_path = os.path.join(base_path, "hrrr_west_lon.npy")
        lats = torch.from_numpy(np.load(lats_path)).float().T
        lons = torch.from_numpy(np.load(lons_path)).float().T
        self.pos_emb = torch.stack(
            [
                torch.sin(torch.deg2rad(lats)),
                torch.cos(torch.deg2rad(lats)),
                torch.sin(torch.deg2rad(lons)),
                torch.cos(torch.deg2rad(lons)),
            ],
            dim=0,
        ).unsqueeze(0).to(device)

        self._static = torch.cat(
            [
                self.geo,
                self.pos_emb,
                self._load_terrain_features(),
                self._load_landuse_features(),
            ],
            dim=1,
        )

    def _load_terrain_features(self):
        features = []
        for name in ("slope", "aspect_sin", "aspect_cos", "tpi"):
            path = os.path.join(self.static_data_path, f"{name}.npy")
            if os.path.exists(path):
                features.append(torch.from_numpy(np.load(path)).float().T)
            else:
                print(f"[WARN] missing {name}.npy; zero-filled")
                features.append(torch.zeros_like(self.geo.squeeze(0).squeeze(0)))
        return torch.stack(features, dim=0).unsqueeze(0).to(self.device)

    def _load_landuse_features(self):
        names = (
            "roughness_log_z0_norm",
            "urban_mask",
            "forest_mask",
            "cropland_mask",
            "grassland_mask",
        )
        mask_names = {"urban_mask", "forest_mask", "cropland_mask", "grassland_mask"}
        features = []
        for name in names:
            path = os.path.join(self.static_data_path, f"{name}.npy")
            if os.path.exists(path):
                data = np.load(path)
                if name in mask_names:
                    data = (data - data.mean()) / (data.std() + 1e-6)
                features.append(torch.from_numpy(data).float().T)
                continue

            if name == "roughness_log_z0_norm":
                fallback = os.path.join(self.static_data_path, "roughness_log_z0.npy")
                if os.path.exists(fallback):
                    data = np.load(fallback)
                    data = (data - data.mean()) / (data.std() + 1e-6)
                    features.append(torch.from_numpy(data).float().T)
                    continue
            print(f"[WARN] missing {name}.npy; zero-filled")
            features.append(torch.zeros_like(self.geo.squeeze(0).squeeze(0)))
        return torch.stack(features, dim=0).unsqueeze(0).to(self.device)

    def _build_input_stats(self):
        level_count = len(self.target_levels)
        input_mean = torch.zeros(7 * level_count, device=self.device)
        input_std = torch.ones(7 * level_count, device=self.device)
        for var_idx in range(7):
            hrrr_base = NGCM_VAR_TO_HRRR_BASE[var_idx]
            for level_idx, level in enumerate(self.target_levels):
                output_idx = var_idx * level_count + level_idx
                if hrrr_base < 0:
                    input_std[output_idx] = 7.538e-06 if var_idx == 5 else 1.979e-05
                else:
                    stat_idx = hrrr_base + LEVEL_STAT_MAPPING[level]
                    input_mean[output_idx] = self.mean_raw[stat_idx]
                    input_std[output_idx] = self.std_raw[stat_idx]
        shape = (1, 1, 7 * level_count, 1, 1)
        return input_mean.view(*shape), input_std.view(*shape)

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

def compute_solar_features(time_list, total_steps, lats, lons, device):
    batch_size = len(time_list)
    width, height = lats.shape
    lats_np = lats.cpu().numpy() if isinstance(lats, torch.Tensor) else np.array(lats)
    lons_np = lons.cpu().numpy() if isinstance(lons, torch.Tensor) else np.array(lons)
    solar = np.zeros(
        (batch_size, total_steps, 4, width, height),
        dtype=np.float32,
    )

    for batch_idx, time_string in enumerate(time_list):
        start = pd.to_datetime(time_string)
        for time_idx in range(total_steps):
            current = start + pd.Timedelta(hours=time_idx)
            day_of_year = current.timetuple().tm_yday
            hour_utc = current.hour + current.minute / 60.0
            declination = 23.45 * np.sin(
                np.radians(360.0 / 365.0 * (284 + day_of_year))
            )
            declination_rad = np.radians(declination)
            local_solar_time = hour_utc + lons_np / 15.0
            hour_angle = np.radians(15.0 * (local_solar_time - 12.0))
            latitude = np.radians(lats_np)
            sin_elevation = (
                np.sin(latitude) * np.sin(declination_rad)
                + np.cos(latitude) * np.cos(declination_rad) * np.cos(hour_angle)
            )
            sin_elevation = np.clip(sin_elevation, -1.0, 1.0)
            solar[batch_idx, time_idx, 0] = sin_elevation
            solar[batch_idx, time_idx, 1] = np.cos(np.arcsin(sin_elevation))
            solar[batch_idx, time_idx, 2] = np.sin(hour_angle)
            solar[batch_idx, time_idx, 3] = np.cos(hour_angle)
    return torch.from_numpy(solar).to(device)
