"""
    Measure 72 h end-to-end forecast generation time for AERO-AIR.
"""

import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
AERO_ODE_ROOT = PROJECT_ROOT.parent
if os.fspath(AERO_ODE_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(AERO_ODE_ROOT))
import paths_config as pc

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np

# GPUs for single-process run; torchrun binds one GPU per LOCAL_RANK.
AVAILABLE_GPUS = [1]


def setup_gpu_visibility():
    """Set CUDA_VISIBLE_DEVICES before importing torch."""
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        if local_rank < len(AVAILABLE_GPUS):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(AVAILABLE_GPUS[local_rank])
        else:
            raise RuntimeError(f"LOCAL_RANK={local_rank} out of AVAILABLE_GPUS range")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, AVAILABLE_GPUS))


setup_gpu_visibility()

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".30"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from film_inference_air import (
    CorrectionModel, StaticDataManager, compute_time_indices,
    StrangSplittingIntegrator, DT_DYNAMICS_S,
)
import AERO_Dataset_v3 as N_Dataset
import AERO_v3 as NGCM


def setup_ddp():
    """Initialize DDP; falls back to single GPU when not launched via torchrun."""
    if "RANK" not in os.environ:
        return 0, torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), 1

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    from datetime import timedelta

    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timedelta(minutes=30))
    return rank, device, world_size


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def sync_device(device):
    """Sync CUDA before/after timing (async kernels skew wall-clock)."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def filter_00z_indices(dataset):
    """Keep 00 UTC init samples only (matches paper evaluation)."""
    filtered_indices = []
    for i, global_idx in enumerate(dataset.valid_indices):
        timestamp = dataset.timeline[global_idx]["timestamp"]
        if timestamp.hour == 0:
            filtered_indices.append(i)
    return filtered_indices


class Subset00Z(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def find_latest_checkpoint(checkpoint_dir=None):
    """Pick latest checkpoint; default to model_ep0.pth if dir missing."""
    if checkpoint_dir is None:
        checkpoint_dir = os.fspath(pc.film_checkpoint_dir(PROJECT_ROOT))
    checkpoint_path = os.path.join(checkpoint_dir, "model_ep0.pth")
    if not os.path.exists(checkpoint_dir):
        return checkpoint_path

    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    if not checkpoints:
        return checkpoint_path

    def parse_checkpoint(fname):
        try:
            if "batch" in fname:
                parts = fname.replace(".pth", "").split("_")
                ep = int(parts[1].replace("ep", ""))
                batch = int(parts[2].replace("batch", ""))
                return ep, batch
            ep = int(fname.replace(".pth", "").split("ep")[1])
            return ep, float("inf")
        except Exception:
            return -1, -1

    return os.path.join(checkpoint_dir, sorted(checkpoints, key=parse_checkpoint)[-1])


def main():
    rank, device, world_size = setup_ddp()
    is_main = is_main_process()

    batch_size = 1
    forecast_hours = 72
    time_steps = forecast_hours + 1  # includes t=0 IC -> 72 h forecast uses 73 steps
    time_chunk = 4
    # First batch may include JAX/PyTorch init/compile; excluded from steady-state timing.
    warmup_batches = int(os.environ.get("TIMING_WARMUP_BATCHES", "1"))
    # MAX_TIMED_BATCHES=0 means all 00Z samples after warmup are timed.
    max_timed_batches = int(os.environ.get("MAX_TIMED_BATCHES", "0"))

    hrrr_stat_path = os.fspath(pc.hrrr_stat_root())
    ngcm_checkpoint = os.fspath(pc.ngcm_checkpoint(PROJECT_ROOT))
    test_data_root = os.fspath(pc.era5_root())
    test_hrrr_root = os.fspath(pc.hrrr_truth_root())

    if is_main:
        print(f"[AIR] batch={batch_size}, world_size={world_size}, forecast_horizon={forecast_hours}h")
        print("[AIR] Timing excludes data loading, target transfer, metrics, plotting and saving.")

    # outer_steps matches 72 h output so NeuralGCM global prior spans 72 h.
    ngcm = NGCM.NeuralGCMInference(ngcm_checkpoint, inner_steps=1, outer_steps=time_steps)
    region_lat = np.load(os.path.join(hrrr_stat_path, "lats.npy")).T
    region_lon = np.load(os.path.join(hrrr_stat_path, "lons.npy")).T

    model = CorrectionModel(time_emb_dim=128, backbone="unet3plus").to(device)
    checkpoint_path = find_latest_checkpoint()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    static_manager = StaticDataManager(hrrr_stat_path, device=device)
    integrator = StrangSplittingIntegrator(dt=DT_DYNAMICS_S)

    # Dataset still loads HRRR truth, but this script does not move it to GPU or compute RMSE.
    ds_full = N_Dataset.ERA5XarrayDataset(
        test_data_root, hrrr_root_dirs=[test_hrrr_root], predict_lead_time=time_steps
    )
    ds = Subset00Z(ds_full, filter_00z_indices(ds_full))
    is_ddp = world_size > 1
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=False) if is_ddp else None
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=2,
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=True,
        collate_fn=N_Dataset.xarray_collate_fn,
        sampler=sampler,
    )

    # Only forecast generation time; DataLoader wait is outside the timed region.
    compute_time_sum = torch.tensor(0.0, device=device)
    compute_time_sq_sum = torch.tensor(0.0, device=device)
    compute_time_min = torch.tensor(float("inf"), device=device)
    compute_time_max = torch.tensor(0.0, device=device)
    timed_samples = torch.tensor(0.0, device=device)
    timed_batches = 0

    with torch.inference_mode():
        for batch_idx, (input_list, time_list, _hrrr_tensor) in enumerate(loader):
            if input_list is None:
                continue

            # Timing starts: NeuralGCM global forward + AERO-AIR regional refinement.
            sync_device(device)
            start = time.perf_counter()

            # 1) NeuralGCM forward: full model + physics core.
            t_pred, t_phy = ngcm.forward(
                input_list,
                target_levels=[50, 500, 850, 1000],
                include_era5_label=False,
                region_lon=region_lon,
                region_lat=region_lat,
            )
            # First 4 Z channels are geopotential -> geopotential height.
            t_pred[:, :4] /= 9.8
            t_phy[:, :4] /= 9.8
            t_pred = t_pred.permute(0, 4, 1, 2, 3).to(device)
            t_phy = t_phy.permute(0, 4, 1, 2, 3).to(device)

            bsz, t_len, _channels, width, height = t_phy.shape
            # 2) Build AERO-AIR input: physics core + correction + static features + time codes.
            input_std = torch.ones(1, 1, 28, 1, 1, device=device)
            input_mean = torch.zeros(1, 1, 28, 1, 1, device=device)
            input_std[0, 0, :20, 0, 0] = static_manager.std[:20]
            input_mean[0, 0, :20, 0, 0] = static_manager.mean[:20]
            input_std[0, 0, 20:24, 0, 0] = 7.538e-06
            input_std[0, 0, 24:28, 0, 0] = 1.979e-05

            t_phy_norm = (t_phy - input_mean) / (input_std + 1e-9)
            correction_norm = (t_pred - t_phy) / (input_std + 1e-9)
            dyn_in = torch.cat([t_phy_norm, correction_norm], dim=2)
            static_in = static_manager.get_static_input(bsz)
            steps_all, hours_all, months_all = compute_time_indices(time_list, t_len, device)

            # 3) Chunked regional network (VRAM); chunks form one 72 h forecast.
            output_chunks = []
            for t0 in range(0, t_len, time_chunk):
                t1 = min(t0 + time_chunk, t_len)
                tc = t1 - t0
                dyn = dyn_in[:, t0:t1].reshape(bsz * tc, 56, width, height)
                sta = static_in.unsqueeze(1).expand(-1, tc, -1, -1, -1).reshape(bsz * tc, 5, width, height)
                s = steps_all.view(bsz, t_len)[:, t0:t1].reshape(-1)
                h = hours_all.view(bsz, t_len)[:, t0:t1].reshape(-1)
                m = months_all.view(bsz, t_len)[:, t0:t1].reshape(-1)

                # Network outputs normalized tendency correction; scale and integrate.
                sgs_increment = model(dyn, sta, s, h, m) * (static_manager.target_std + 1e-9)
                state_post = t_phy[:, t0:t1, :20].reshape(bsz * tc, 20, width, height)
                if t0 == 0:
                    state_pre = state_post
                else:
                    state_pre = t_phy[:, t0 - 1 : t1 - 1, :20].reshape(bsz * tc, 20, width, height)
                state_next = integrator.step(state_pre, state_post, sgs_increment)
                output_chunks.append(state_next.reshape(bsz, tc, 20, width, height))

            # Timing ends here (no RMSE/plot/save after this).
            _output = torch.cat(output_chunks, dim=1)
            sync_device(device)
            elapsed = time.perf_counter() - start

            # Skip warmup batches; count steady-state samples only.
            if batch_idx >= warmup_batches:
                compute_time_sum += elapsed
                compute_time_sq_sum += elapsed * elapsed
                compute_time_min = torch.minimum(compute_time_min, torch.tensor(elapsed, device=device))
                compute_time_max = torch.maximum(compute_time_max, torch.tensor(elapsed, device=device))
                timed_samples += bsz
                timed_batches += 1
                if is_main:
                    print(f"[AIR] timed batch {timed_batches}: {elapsed:.3f} s")
                if max_timed_batches > 0 and timed_batches >= max_timed_batches:
                    break

    # DDP: each rank different samples; mean is sample-weighted across ranks;
    # wall_time uses MAX (slowest rank sets parallel wait).
    wall_time = compute_time_sum.clone()
    if is_ddp:
        dist.all_reduce(wall_time, op=dist.ReduceOp.MAX)
        dist.all_reduce(compute_time_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(compute_time_sq_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(compute_time_min, op=dist.ReduceOp.MIN)
        dist.all_reduce(compute_time_max, op=dist.ReduceOp.MAX)
        dist.all_reduce(timed_samples, op=dist.ReduceOp.SUM)

    if is_main:
        if timed_samples.item() > 0:
            avg_time_tensor = compute_time_sum / timed_samples
            variance = torch.clamp(compute_time_sq_sum / timed_samples - avg_time_tensor * avg_time_tensor, min=0.0)
            avg_time = avg_time_tensor.item()
            std_time = torch.sqrt(variance).item()
            min_time = compute_time_min.item()
            max_time = compute_time_max.item()
        else:
            avg_time = float("nan")
            std_time = float("nan")
            min_time = float("nan")
            max_time = float("nan")
        result = {
            "component": "AERO-AIR",
            "forecast_hours": forecast_hours,
            "timed_samples": int(timed_samples.item()),
            "average_seconds_per_72h_forecast": avg_time,
            "min_seconds_per_72h_forecast": min_time,
            "max_seconds_per_72h_forecast": max_time,
            "std_seconds_per_72h_forecast": std_time,
            "ddp_wall_seconds_timed_subset": wall_time.item(),
            "warmup_batches_per_rank": warmup_batches,
        }
        print(
            "[AIR] 72h forecast time summary: "
            f"avg={avg_time:.3f}s, min={min_time:.3f}s, max={max_time:.3f}s, std={std_time:.3f}s"
        )
        print(json.dumps(result, ensure_ascii=False))

    cleanup_ddp()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        cleanup_ddp()
        raise
