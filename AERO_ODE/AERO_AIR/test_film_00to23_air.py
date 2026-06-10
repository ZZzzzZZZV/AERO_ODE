"""
    python test_film_00to23_air.py
"""

import gc
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
AERO_ODE_ROOT = PROJECT_ROOT.parent
if os.fspath(AERO_ODE_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(AERO_ODE_ROOT))
import paths_config as pc

import numpy as np

# ============================================
# GPU config (edit here; before torch import)
# ============================================
# Physical GPU IDs (match nvidia-smi):
#   [0]     -> GPU0 only
#   [1]     -> GPU1 only
#   [0, 1]  -> 2-GPU DDP (torchrun --nproc_per_node=2)
AVAILABLE_GPUS = [1]


def setup_gpu_visibility():
    """Set GPU visibility for DDP/torchrun"""
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        if local_rank < len(AVAILABLE_GPUS):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(AVAILABLE_GPUS[local_rank])
        else:
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} out of AVAILABLE_GPUS range "
                f"({len(AVAILABLE_GPUS)} GPUs: {AVAILABLE_GPUS})"
            )
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, AVAILABLE_GPUS))


setup_gpu_visibility()

# JAX memory settings
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
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
from l2 import RelativeL2Loss
import AERO_Dataset_v3 as N_Dataset
import AERO_v3 as NGCM


# ============================================
# DDP utilities
# ============================================
def setup_ddp():
    """Initialize DDP environment"""
    if 'RANK' not in os.environ:
        return 0, torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), 1

    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    device = torch.device('cuda:0')
    torch.cuda.set_device(0)

    from datetime import timedelta
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timedelta(minutes=30))
    return rank, device, world_size


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


INIT_HOURS = [0, 6, 12, 18]  # UTC init hours; use [0,6,12,18] for 00/06/12/18 only
MODEL_TAG = "aero_ode"  # AERO-AIR correction results


def aero_ode_rmse_npy_path(save_dir: str, init_hour=None) -> str:
    """AERO-ODE RMSE save path; init_hour=None pools all INIT_HOURS"""
    if init_hour is None:
        return os.path.join(save_dir, f"rmse_{MODEL_TAG}_inithour_mean.npy")
    return os.path.join(save_dir, f"rmse_{MODEL_TAG}_inithour_{init_hour:02d}Z.npy")


def release_gpu_between_inithours(device):
    """Release GPU memory after each init hour (avoid JAX+DataLoader CUDA crash)"""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def build_eval_loader(ds, batch_size, sampler, num_workers):
    """Use num_workers=0 with JAX(NeuralGCM)+CUDA to avoid CUDA_ERROR_NOT_INITIALIZED after fork"""
    kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=N_Dataset.xarray_collate_fn,
        sampler=sampler,
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
        kwargs["persistent_workers"] = False
    return DataLoader(ds, **kwargs)


def filter_inithour_indices(dataset: N_Dataset.ERA5XarrayDataset, init_hour: int):
    """Filter sample indices for given UTC init hour"""
    filtered_indices = []
    for i, global_idx in enumerate(dataset.valid_indices):
        timestamp = dataset.timeline[global_idx]['timestamp']
        if timestamp.hour == init_hour:
            filtered_indices.append(i)
    return filtered_indices


class SubsetByInitHour(torch.utils.data.Dataset):
    """Dataset wrapper for one init-hour subset (DDP sharding)"""

    def __init__(self, dataset: N_Dataset.ERA5XarrayDataset, indices: list):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def evaluate():
    # --- DDP init ---
    rank, device, world_size = setup_ddp()
    is_main = is_main_process()

    # --- Config ---
    BATCH_SIZE = 1
    TIME_STEPS = 73  # forecast lead steps (~48 h)
    TIME_CHUNK = 8
    # With NeuralGCM(JAX)+PyTorch CUDA, DataLoader workers abort on the second init hour
    DATALOADER_NUM_WORKERS = 0
    HRRR_STAT_PATH = os.fspath(pc.hrrr_stat_root())
    NGCM_CHECKPOINT = os.fspath(pc.ngcm_checkpoint(PROJECT_ROOT))
    TEST_DATA_ROOT = os.fspath(pc.era5_root())
    TEST_HRRR_ROOT = os.fspath(pc.hrrr_truth_root())

    RESULT_SAVE_DIR = os.fspath(PROJECT_ROOT / 'Test_Data_Rmse_by_inithour')
    if is_main:
        os.makedirs(RESULT_SAVE_DIR, exist_ok=True)
        print(f"Config: batch={BATCH_SIZE}, world_size={world_size}, GPUs={AVAILABLE_GPUS}")
        hours_str = ",".join(f"{h:02d}" for h in INIT_HOURS)
        print(f"Mode: init hours [{hours_str}] UTC, output {len(INIT_HOURS) + 1} AERO-ODE npy files")

    CHECKPOINT_DIR = os.fspath(pc.film_checkpoint_dir(PROJECT_ROOT))
    CHECKPOINT_PATH = None
    if os.path.exists(CHECKPOINT_DIR):
        checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pth')]
        if checkpoints:
            def parse_checkpoint(fname):
                try:
                    if 'batch' in fname:
                        parts = fname.replace('.pth', '').split('_')
                        ep = int(parts[1].replace('ep', ''))
                        batch = int(parts[2].replace('batch', ''))
                        return (ep, batch)
                    else:
                        ep = int(fname.replace('.pth', '').split('ep')[1])
                        return (ep, float('inf'))
                except Exception:
                    return (-1, -1)

            latest = sorted(checkpoints, key=parse_checkpoint)[-1]
            CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, latest)
            if is_main:
                print(f"Using checkpoint: {CHECKPOINT_PATH}")

    if CHECKPOINT_PATH is None:
        raise FileNotFoundError(f"No checkpoint found in {CHECKPOINT_DIR}")

    # --- Init models ---
    if is_main:
        print("Initializing NeuralGCM...")
    ngcm = NGCM.NeuralGCMInference(NGCM_CHECKPOINT, inner_steps=1, outer_steps=TIME_STEPS)
    region_lat = np.load(os.path.join(HRRR_STAT_PATH, 'lats.npy')).T
    region_lon = np.load(os.path.join(HRRR_STAT_PATH, 'lons.npy')).T

    if is_main:
        print("Loading Correction Model...")
    model = CorrectionModel(time_emb_dim=128, backbone='unet3plus').to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    static_manager = StaticDataManager(HRRR_STAT_PATH, device=device)
    integrator = StrangSplittingIntegrator(dt=DT_DYNAMICS_S)
    loss_fn = RelativeL2Loss()
    LOSS_SCALE = 1.0  # 1000.0

    ds_full = N_Dataset.ERA5XarrayDataset(TEST_DATA_ROOT, hrrr_root_dirs=[TEST_HRRR_ROOT], predict_lead_time=TIME_STEPS)
    if is_main:
        print(f"Full dataset samples: {len(ds_full)}")

    is_ddp = world_size > 1
    evaluated_hours = []
    global_mse_aero_ode_sum = torch.zeros(TIME_STEPS, 20).to(device)  # [lead time, channel]
    global_count = torch.tensor(0.0).to(device)

    for init_hour in INIT_HOURS:
        hour_tag = f"{init_hour:02d}"
        indices = filter_inithour_indices(ds_full, init_hour)
        ds = SubsetByInitHour(ds_full, indices)

        if is_main:
            n = len(ds)
            pct = n / len(ds_full) * 100 if len(ds_full) else 0.0
            print(f"\n{'=' * 60}")
            print(f"Init hour {hour_tag}Z | samples: {n} ({pct:.1f}%)")
            if n == 0:
                print(f"Skip {hour_tag}Z: no samples")
            print(f"{'=' * 60}")

        if len(ds) == 0:
            continue

        sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=False) if is_ddp else None
        loader = build_eval_loader(ds, BATCH_SIZE, sampler, DATALOADER_NUM_WORKERS)

        mse_aero_ode_sum = torch.zeros(TIME_STEPS, 20).to(device)  # [lead time, channel]
        total_loss = torch.tensor(0.0).to(device)
        count = torch.tensor(0).to(device)

        if is_main:
            print(f"Start Evaluation (Init {hour_tag}Z)...")

        with torch.inference_mode():
            for batch_idx, (input_list, time_list, hrrr_tensor) in enumerate(loader):
                if input_list is None:
                    continue

                t_pred, t_phy = ngcm.forward(input_list, target_levels=[50, 500, 850, 1000],
                                             include_era5_label=False, region_lon=region_lon, region_lat=region_lat)
                t_pred[:, :4] /= 9.8
                t_phy[:, :4] /= 9.8
                t_pred = t_pred.permute(0, 4, 1, 2, 3).to(device)
                t_phy = t_phy.permute(0, 4, 1, 2, 3).to(device)
                gt_hrrr = hrrr_tensor.permute(0, 1, 2, 4, 3).to(device)[:, :, :20, :, :]

                B, T, C, W, H = t_phy.shape

                input_std = torch.ones(1, 1, 28, 1, 1, device=device)
                input_mean = torch.zeros(1, 1, 28, 1, 1, device=device)
                input_std[0, 0, :20, 0, 0] = static_manager.std[:20]
                input_mean[0, 0, :20, 0, 0] = static_manager.mean[:20]
                input_std[0, 0, 20:24, 0, 0] = 7.538e-06
                input_std[0, 0, 24:28, 0, 0] = 1.979e-05

                t_phy_norm = (t_phy - input_mean) / (input_std + 1e-9)
                correction_norm = (t_pred - t_phy) / (input_std + 1e-9)
                dyn_in = torch.cat([t_phy_norm, correction_norm], dim=2)
                static_in = static_manager.get_static_input(B)
                steps_all, hours_all, months_all = compute_time_indices(time_list, T, device)

                output_chunks = []
                for t0 in range(0, T, TIME_CHUNK):
                    t1 = min(t0 + TIME_CHUNK, T)
                    tc = t1 - t0

                    dyn = dyn_in[:, t0:t1].reshape(B * tc, 56, W, H)
                    sta = static_in.unsqueeze(1).expand(-1, tc, -1, -1, -1).reshape(B * tc, 5, W, H)
                    s = steps_all.view(B, T)[:, t0:t1].reshape(-1)
                    h = hours_all.view(B, T)[:, t0:t1].reshape(-1)
                    m = months_all.view(B, T)[:, t0:t1].reshape(-1)

                    sgs_increment = model(dyn, sta, s, h, m) * (static_manager.target_std + 1e-9)

                    state_post = t_phy[:, t0:t1, :20].reshape(B * tc, 20, W, H)
                    if t0 == 0:
                        state_pre = state_post
                    else:
                        state_pre = t_phy[:, t0 - 1:t1 - 1, :20].reshape(B * tc, 20, W, H)

                    state_next = integrator.step(state_pre, state_post, sgs_increment)
                    output_chunks.append(state_next.reshape(B, tc, 20, W, H))

                corrected = torch.cat(output_chunks, dim=1)

                corrected_flat = corrected.reshape(B * T, 20, W, H)
                gt_flat = gt_hrrr.reshape(B * T, 20, W, H)
                pred_list = [corrected_flat[i::T] for i in range(T)]
                gt_list = [gt_flat[i::T] for i in range(T)]
                batch_loss = loss_fn(pred_list, gt_list).item() * LOSS_SCALE
                total_loss += batch_loss / T

                mse_aero_ode = ((corrected - gt_hrrr) ** 2).mean(dim=[0, 3, 4])
                mse_aero_ode_sum += mse_aero_ode
                count += 1

                if is_main and batch_idx % 10 == 0:
                    init_time = time_list[0] if time_list else "N/A"
                    print(f"Rank {rank} | {hour_tag}Z | Batch {batch_idx}/{len(loader)} | Init: {init_time} | Loss: {batch_loss / T:.5f}")

        if is_ddp:
            dist.all_reduce(mse_aero_ode_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(count, op=dist.ReduceOp.SUM)

        global_mse_aero_ode_sum += mse_aero_ode_sum
        global_count += count

        if is_main and count.item() > 0:
            avg_rmse_aero_ode = torch.sqrt(mse_aero_ode_sum / count).cpu().numpy()
            avg_loss = (total_loss / count).item()

            print(f"\n=== Init {hour_tag}Z Complete ({count.item()} batches, {world_size} GPUs) ===")
            print(f"Average Loss: {avg_loss:.5f}")

            np.save(aero_ode_rmse_npy_path(RESULT_SAVE_DIR, init_hour), avg_rmse_aero_ode)
            evaluated_hours.append(init_hour)

            levels = [50, 500, 850, 1000]
            var_names = ['Z', 'T', 'Q', 'U', 'V']
            print(f"\nAERO-ODE RMSE [Init {hour_tag}Z] -> {os.path.basename(aero_ode_rmse_npy_path(RESULT_SAVE_DIR, init_hour))}")
            print(f"{'Channel':<12} | {'AERO-ODE RMSE':<14}")
            print("-" * 30)
            for v_idx, vname in enumerate(var_names):
                for l_idx, lev in enumerate(levels):
                    ch = v_idx * len(levels) + l_idx
                    print(f"{vname}@{lev:<4}    | {avg_rmse_aero_ode[:, ch].mean():.4f}")
        elif is_main:
            print(f"\nWarning: Init {hour_tag}Z has no valid batches; skip saving npy")

        del loader
        release_gpu_between_inithours(device)
        if is_main:
            print(f"GPU cache released; next init hour")

    if is_main and global_count > 0:
        mean_rmse_aero_ode = torch.sqrt(global_mse_aero_ode_sum / global_count).cpu().numpy()
        mean_path = aero_ode_rmse_npy_path(RESULT_SAVE_DIR, None)
        np.save(mean_path, mean_rmse_aero_ode)
        np.save(os.path.join(RESULT_SAVE_DIR, 'inithour_labels.npy'), np.array(evaluated_hours, dtype=np.int32))

        saved_files = [os.path.basename(aero_ode_rmse_npy_path(RESULT_SAVE_DIR, h)) for h in evaluated_hours]
        saved_files.append(os.path.basename(mean_path))
        saved_files.append('inithour_labels.npy')

        print(f"\nAll init hours done ({len(evaluated_hours)} hours + 1 mean, {len(saved_files)} npy total)")
        print(f"Directory: {RESULT_SAVE_DIR}")
        print("AERO-ODE (AERO-AIR) RMSE files:")
        for fname in sorted(saved_files):
            print(f"  - {fname}")

        levels = [50, 500, 850, 1000]
        var_names = ['Z', 'T', 'Q', 'U', 'V']
        print(f"\nAERO-ODE RMSE [Init Hours {','.join(f'{h:02d}' for h in evaluated_hours)} Mean]:")
        print(f"{'Channel':<12} | {'AERO-ODE RMSE':<14}")
        print("-" * 30)
        for v_idx, vname in enumerate(var_names):
            for l_idx, lev in enumerate(levels):
                ch = v_idx * len(levels) + l_idx
                print(f"{vname}@{lev:<4}    | {mean_rmse_aero_ode[:, ch].mean():.4f}")

    cleanup_ddp()


if __name__ == "__main__":
    try:
        evaluate()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        cleanup_ddp()
