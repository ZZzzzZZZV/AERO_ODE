"""
    Surface variable diagnosis inference module.
"""

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


class DiagnosisModel(nn.Module):
    """Surface variable diagnosis model"""

    def __init__(self, in_channels_dynamic=74, in_channels_static=14, out_channels=4,
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
    """Manage static data: terrain, position encoding, terrain derivatives, land-use features, statistics"""

    def __init__(self, base_path, device='cuda', target_levels=None,
                 static_data_path=None):
        self.device = device
        self.target_levels = target_levels or TARGET_LEVELS

        if static_data_path is None:
            possible_paths = [
                os.path.join(base_path, 'data'),
                base_path,
                './Hrrr_rb/data'
            ]
            for p in possible_paths:
                if os.path.exists(os.path.join(p, 'slope.npy')):
                    static_data_path = p
                    break
            if static_data_path is None:
                static_data_path = base_path
        self.static_data_path = static_data_path

        self.mean_raw = torch.from_numpy(
            np.load(os.path.join(base_path, 'stat/mean_crop.npy'))
        ).float().to(device)
        self.std_raw = torch.from_numpy(
            np.load(os.path.join(base_path, 'stat/std_crop.npy'))
        ).float().to(device)

        self.mean = self.mean_raw
        self.std = self.std_raw
        self.input_mean, self.input_std = self._build_input_stats()
        self.surface_mean = self.mean_raw[20:24].view(1, 4, 1, 1)
        self.surface_std = self.std_raw[20:24].view(1, 4, 1, 1)
        self.target_std = self.std_raw[:20].view(1, 20, 1, 1)

        geo_path = os.path.join(static_data_path, 'geo.h5')
        if not os.path.exists(geo_path):
            geo_path = os.path.join(base_path, 'geo.h5')
        with h5py.File(geo_path, "r") as f:
            geo = torch.from_numpy(f["fields"][:]).float().permute(0, 2, 1).unsqueeze(0)
            self.geo = ((geo - geo.mean()) / (geo.std() + 1e-6)).to(device)

        lats_path = os.path.join(static_data_path, 'lats.npy')
        lons_path = os.path.join(static_data_path, 'lons.npy')
        if not os.path.exists(lats_path):
            lats_path = os.path.join(base_path, 'lats.npy')
            lons_path = os.path.join(base_path, 'lons.npy')
        lats = torch.from_numpy(np.load(lats_path)).float().T
        lons = torch.from_numpy(np.load(lons_path)).float().T
        self.pos_emb = torch.stack([
            torch.sin(torch.deg2rad(lats)), torch.cos(torch.deg2rad(lats)),
            torch.sin(torch.deg2rad(lons)), torch.cos(torch.deg2rad(lons))
        ], dim=0).unsqueeze(0).to(device)

        terrain_features = self._load_terrain_features()
        landuse_features = self._load_landuse_features()

        self._static = torch.cat([
            self.geo,
            self.pos_emb,
            terrain_features,
            landuse_features
        ], dim=1)

        print(f"Static feature shape: {self._static.shape}")

    def _load_terrain_features(self):
        features = []
        feature_names = ['slope', 'aspect_sin', 'aspect_cos', 'tpi']

        for name in feature_names:
            path = os.path.join(self.static_data_path, f'{name}.npy')
            if os.path.exists(path):
                data = torch.from_numpy(np.load(path)).float().T
                features.append(data)
            else:
                print(f"Warning: missing {name}.npy, using zero fill")
                features.append(torch.zeros_like(self.geo.squeeze(0).squeeze(0)))

        return torch.stack(features, dim=0).unsqueeze(0).to(self.device)

    def _load_landuse_features(self):
        features = []
        feature_names = [
            'roughness_log_z0_norm',
            'urban_mask',
            'forest_mask',
            'cropland_mask',
            'grassland_mask'
        ]
        mask_features = {'urban_mask', 'forest_mask', 'cropland_mask', 'grassland_mask'}

        for name in feature_names:
            path = os.path.join(self.static_data_path, f'{name}.npy')
            if os.path.exists(path):
                data = np.load(path)
                if name in mask_features:
                    data = (data - data.mean()) / (data.std() + 1e-6)
                data = torch.from_numpy(data).float().T
                features.append(data)
            else:
                if name == 'roughness_log_z0_norm':
                    alt_path = os.path.join(self.static_data_path, 'roughness_log_z0.npy')
                    if os.path.exists(alt_path):
                        data = np.load(alt_path)
                        data = (data - data.mean()) / (data.std() + 1e-6)
                        data = torch.from_numpy(data).float().T
                        features.append(data)
                        continue
                print(f"Warning: missing {name}.npy, using zero fill")
                features.append(torch.zeros_like(self.geo.squeeze(0).squeeze(0)))

        return torch.stack(features, dim=0).unsqueeze(0).to(self.device)

    def _build_input_stats(self):
        n_vars = 7
        n_levels = len(self.target_levels)

        input_mean = torch.zeros(n_vars * n_levels, device=self.device)
        input_std = torch.ones(n_vars * n_levels, device=self.device)

        for var_idx in range(n_vars):
            hrrr_base = NGCM_VAR_TO_HRRR_BASE[var_idx]

            for level_idx, level in enumerate(self.target_levels):
                out_idx = var_idx * n_levels + level_idx

                if hrrr_base < 0:
                    input_mean[out_idx] = 0.0
                    if var_idx == 5:
                        input_std[out_idx] = 7.538e-06
                    else:
                        input_std[out_idx] = 1.979e-05
                else:
                    stat_level_idx = LEVEL_STAT_MAPPING[level]
                    stat_idx = hrrr_base + stat_level_idx
                    input_mean[out_idx] = self.mean_raw[stat_idx]
                    input_std[out_idx] = self.std_raw[stat_idx]

        return input_mean.view(1, 1, n_vars * n_levels, 1, 1), \
               input_std.view(1, 1, n_vars * n_levels, 1, 1)

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


def compute_solar_features(time_list, total_steps, lats, lons, device):
    B = len(time_list)
    W, H = lats.shape

    if isinstance(lats, torch.Tensor):
        lats_np = lats.cpu().numpy()
        lons_np = lons.cpu().numpy()
    else:
        lats_np = np.array(lats)
        lons_np = np.array(lons)

    solar_features = np.zeros((B, total_steps, 4, W, H), dtype=np.float32)

    for b, t_str in enumerate(time_list):
        dt_start = pd.to_datetime(t_str)

        for t in range(total_steps):
            dt = dt_start + pd.Timedelta(hours=t)
            doy = dt.timetuple().tm_yday
            hour_utc = dt.hour + dt.minute / 60.0

            declination = 23.45 * np.sin(np.radians(360.0 / 365.0 * (284 + doy)))
            declination_rad = np.radians(declination)
            lst = hour_utc + lons_np / 15.0
            hour_angle_rad = np.radians(15.0 * (lst - 12.0))
            lat_rad = np.radians(lats_np)

            sin_elevation = (np.sin(lat_rad) * np.sin(declination_rad) +
                           np.cos(lat_rad) * np.cos(declination_rad) * np.cos(hour_angle_rad))
            sin_elevation = np.clip(sin_elevation, -1.0, 1.0)
            elevation_rad = np.arcsin(sin_elevation)
            cos_elevation = np.cos(elevation_rad)

            solar_features[b, t, 0] = sin_elevation
            solar_features[b, t, 1] = cos_elevation
            solar_features[b, t, 2] = np.sin(hour_angle_rad)
            solar_features[b, t, 3] = np.cos(hour_angle_rad)

    return torch.from_numpy(solar_features).to(device)
