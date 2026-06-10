import contextlib
import io
import os
import sys
from pathlib import Path

import numpy as np
import torch

from script import config as cfg

SURFACE_ROOT = cfg.SURFACE_ROOT
if os.fspath(SURFACE_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(SURFACE_ROOT))

import paths_config as pc
import AERO_v3 as NGCM
from film_inference_surface import (
    DiagnosisModel,
    StaticDataManager,
    TARGET_LEVELS,
    compute_solar_features,
    compute_time_indices,
)


class SurfaceForecaster:
    """AERO-Surface 72 h MSLP / T2m / 10 m wind forecast."""

    def __init__(self, device: str = "cuda:0", time_chunk: int = 8):
        self.device = torch.device(device)
        self.time_chunk = time_chunk
        stat_path = os.fspath(pc.hrrr_stat_root())
        ngcm_ckpt = os.fspath(pc.ngcm_checkpoint(SURFACE_ROOT))
        diagnosis_ckpt = pc.film_checkpoint_dir(SURFACE_ROOT, "checkpoints_film_v2") / "model_ep5_batch4499.pth"
        pc.ensure_exists(diagnosis_ckpt, "Surface FiLM checkpoint")
        static_data_path = os.fspath(pc.surface_static_data(SURFACE_ROOT))

        self.static_manager = StaticDataManager(
            stat_path,
            device=self.device,
            target_levels=TARGET_LEVELS,
            static_data_path=static_data_path,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            self.ngcm = NGCM.NeuralGCMInference(
                ngcm_ckpt, inner_steps=1, outer_steps=cfg.TIME_STEPS
            )
        self.diagnosis_model = DiagnosisModel(
            in_channels_dynamic=74,
            in_channels_static=14,
            out_channels=4,
            time_emb_dim=128,
            backbone="unet3plus",
        ).to(self.device)

        checkpoint = torch.load(os.fspath(diagnosis_ckpt), map_location=self.device)
        state_dict = {
            k.replace("module.", ""): v for k, v in checkpoint["model_state_dict"].items()
        }
        self.diagnosis_model.load_state_dict(state_dict)
        self.diagnosis_model.eval()

        self.region_lat = np.load(os.path.join(stat_path, "lats.npy")).T
        self.region_lon = np.load(os.path.join(stat_path, "lons.npy")).T
        self.input_mean = self.static_manager.input_mean
        self.input_std = self.static_manager.input_std

    @torch.inference_mode()
    def predict(self, input_list, time_list):
        t_pred, t_phy = self.ngcm.forward(
            input_list,
            target_levels=TARGET_LEVELS,
            include_era5_label=False,
            region_lon=self.region_lon,
            region_lat=self.region_lat,
        )
        t_pred[:, :5] /= 9.8
        t_phy[:, :5] /= 9.8
        t_pred = t_pred.permute(0, 4, 1, 2, 3).to(self.device)
        t_phy = t_phy.permute(0, 4, 1, 2, 3).to(self.device)

        bsz, t_len, _, width, height = t_phy.shape
        t_phy_norm = (t_phy - self.input_mean) / (self.input_std + 1e-9)
        corr_norm = (t_pred - t_phy) / (self.input_std + 1e-9)
        solar_feat = compute_solar_features(
            time_list, t_len, self.region_lat, self.region_lon, self.device
        )
        dyn_in = torch.cat([t_phy_norm, corr_norm, solar_feat], dim=2)
        static_in = self.static_manager.get_static_input(bsz)
        steps_all, hours_all, months_all = compute_time_indices(time_list, t_len, self.device)

        chunks = []
        for t0 in range(0, t_len, self.time_chunk):
            t1 = min(t0 + self.time_chunk, t_len)
            tc = t1 - t0
            dyn = dyn_in[:, t0:t1].reshape(bsz * tc, 74, width, height)
            n_static = static_in.shape[1]
            sta = static_in.unsqueeze(1).expand(-1, tc, -1, -1, -1).reshape(bsz * tc, n_static, width, height)
            s = steps_all.view(bsz, t_len)[:, t0:t1].reshape(-1)
            h = hours_all.view(bsz, t_len)[:, t0:t1].reshape(-1)
            m = months_all.view(bsz, t_len)[:, t0:t1].reshape(-1)

            output_norm = self.diagnosis_model(dyn, sta, s, h, m)
            pred = output_norm * (self.static_manager.surface_std + 1e-9) + self.static_manager.surface_mean
            chunks.append(pred.reshape(bsz, tc, 4, width, height))

        surface_pred = torch.cat(chunks, dim=1)
        return surface_pred.permute(0, 1, 2, 4, 3)
