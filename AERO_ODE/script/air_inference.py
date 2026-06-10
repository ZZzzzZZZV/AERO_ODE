import contextlib
import io
import os
import sys
from pathlib import Path

import numpy as np
import torch

from script import config as cfg

AIR_ROOT = cfg.AIR_ROOT
if os.fspath(AIR_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(AIR_ROOT))

import paths_config as pc
import AERO_v3 as NGCM
from film_inference_air import (
    CorrectionModel,
    DT_DYNAMICS_S,
    StaticDataManager,
    StrangSplittingIntegrator,
    compute_time_indices,
)


def _find_latest_checkpoint(checkpoint_dir: Path) -> Path:
    checkpoints = sorted(checkpoint_dir.glob("*.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"No FiLM checkpoints in {checkpoint_dir}")

    def sort_key(path: Path):
        name = path.name
        try:
            if "batch" in name:
                parts = name.replace(".pth", "").split("_")
                return int(parts[1].replace("ep", "")), int(parts[2].replace("batch", ""))
            return int(name.replace(".pth", "").split("ep")[1]), 10**9
        except Exception:
            return -1, -1

    return sorted(checkpoints, key=sort_key)[-1]


class AirForecaster:
    """AERO-AIR 72 h regional pressure-level forecast."""

    def __init__(self, device: str = "cuda:0", time_chunk: int = 4):
        self.device = torch.device(device)
        self.time_chunk = time_chunk
        stat_path = os.fspath(pc.hrrr_stat_root())
        ngcm_ckpt = os.fspath(pc.ngcm_checkpoint(AIR_ROOT))
        ckpt_dir = pc.film_checkpoint_dir(AIR_ROOT)
        ckpt_path = _find_latest_checkpoint(ckpt_dir)

        with contextlib.redirect_stdout(io.StringIO()):
            self.ngcm = NGCM.NeuralGCMInference(
                ngcm_ckpt, inner_steps=1, outer_steps=cfg.TIME_STEPS
            )
        self.region_lat = np.load(os.path.join(stat_path, "lats.npy")).T
        self.region_lon = np.load(os.path.join(stat_path, "lons.npy")).T

        self.model = CorrectionModel(time_emb_dim=128, backbone="unet3plus").to(self.device)
        checkpoint = torch.load(os.fspath(ckpt_path), map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.static_manager = StaticDataManager(stat_path, device=self.device)
        self.integrator = StrangSplittingIntegrator(dt=DT_DYNAMICS_S)

    @torch.inference_mode()
    def predict(self, input_list, time_list):
        t_pred, t_phy = self.ngcm.forward(
            input_list,
            target_levels=[50, 500, 850, 1000],
            include_era5_label=False,
            region_lon=self.region_lon,
            region_lat=self.region_lat,
        )
        t_pred[:, :4] /= 9.8
        t_phy[:, :4] /= 9.8
        t_pred = t_pred.permute(0, 4, 1, 2, 3).to(self.device)
        t_phy = t_phy.permute(0, 4, 1, 2, 3).to(self.device)

        bsz, t_len, _, width, height = t_phy.shape
        input_std = torch.ones(1, 1, 28, 1, 1, device=self.device)
        input_mean = torch.zeros(1, 1, 28, 1, 1, device=self.device)
        input_std[0, 0, :20, 0, 0] = self.static_manager.std[:20]
        input_mean[0, 0, :20, 0, 0] = self.static_manager.mean[:20]
        input_std[0, 0, 20:24, 0, 0] = 7.538e-06
        input_std[0, 0, 24:28, 0, 0] = 1.979e-05

        dyn_in = torch.cat(
            [
                (t_phy - input_mean) / (input_std + 1e-9),
                (t_pred - t_phy) / (input_std + 1e-9),
            ],
            dim=2,
        )
        static_in = self.static_manager.get_static_input(bsz)
        steps_all, hours_all, months_all = compute_time_indices(time_list, t_len, self.device)

        chunks = []
        for t0 in range(0, t_len, self.time_chunk):
            t1 = min(t0 + self.time_chunk, t_len)
            tc = t1 - t0
            dyn = dyn_in[:, t0:t1].reshape(bsz * tc, 56, width, height)
            sta = static_in.unsqueeze(1).expand(-1, tc, -1, -1, -1).reshape(bsz * tc, 5, width, height)
            s = steps_all.view(bsz, t_len)[:, t0:t1].reshape(-1)
            h = hours_all.view(bsz, t_len)[:, t0:t1].reshape(-1)
            m = months_all.view(bsz, t_len)[:, t0:t1].reshape(-1)

            sgs_increment = self.model(dyn, sta, s, h, m) * (self.static_manager.target_std + 1e-9)
            state_post = t_phy[:, t0:t1, :20].reshape(bsz * tc, 20, width, height)
            if t0 == 0:
                state_pre = state_post
            else:
                state_pre = t_phy[:, t0 - 1 : t1 - 1, :20].reshape(bsz * tc, 20, width, height)
            state_next = self.integrator.step(state_pre, state_post, sgs_increment)
            chunks.append(state_next.reshape(bsz, tc, 20, width, height))

        output = torch.cat(chunks, dim=1)
        return output.permute(0, 1, 2, 4, 3)
