"""
    python test_film_rb.py
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
AERO_ODE_ROOT = PROJECT_ROOT.parent
if os.fspath(AERO_ODE_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(AERO_ODE_ROOT))
import paths_config as pc

os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# device_ids = [0]

import numpy as np
import matplotlib.pyplot as plt

# ============================================
# GPU visibility (before torch import)
# ============================================
AVAILABLE_GPUS = [0]  # edit to change GPUs


def setup_gpu_visibility():
    """Set GPU visibility for DDP/torchrun"""
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        if local_rank < len(AVAILABLE_GPUS):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(AVAILABLE_GPUS[local_rank])
        else:
            raise RuntimeError(f"LOCAL_RANK={local_rank} out of AVAILABLE_GPUS range")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, AVAILABLE_GPUS))


setup_gpu_visibility()

# JAX memory settings
# os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".30"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Subset

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


def filter_00z_indices(dataset: N_Dataset.ERA5XarrayDataset):
    """
    Filter indices for 00Z init samples only.

    Args:
        dataset: ERA5XarrayDataset instance

    Returns:
        list: position indices into valid_indices for 00Z inits only
    """
    filtered_indices = []
    for i, global_idx in enumerate(dataset.valid_indices):
        timestamp = dataset.timeline[global_idx]['timestamp']
        if timestamp.hour == 0:  # 00Z only
            filtered_indices.append(i)
    return filtered_indices


class Subset00Z(torch.utils.data.Dataset):
    """
    Dataset wrapper containing 00Z init samples only.

    Ensures correct DDP sharding across ranks.
    """

    def __init__(self, dataset: N_Dataset.ERA5XarrayDataset, indices: list):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def plot_rmse_comparison(rmse_phy, rmse_ngcm, rmse_split, target_levels, save_path):
    """Plot RMSE comparison figure"""
    var_names = ['Geopotential Height', 'Temperature', 'Specific Humidity', 'U Component', 'V Component']
    num_vars, num_levels = len(var_names), len(target_levels)

    fig, axes = plt.subplots(nrows=num_vars, ncols=num_levels, figsize=(4 * num_levels, 3 * num_vars))
    times = np.arange(rmse_phy.shape[1])

    for v_idx, var_name in enumerate(var_names):
        for l_idx, level in enumerate(target_levels):
            ax = axes[v_idx, l_idx]
            ch_idx = v_idx * num_levels + l_idx

            y_phy = rmse_phy[ch_idx, :]
            y_ngcm = rmse_ngcm[ch_idx, :]
            y_corr = rmse_split[ch_idx, :]

            ax.plot(times, y_phy, label='Physics', color='red', linestyle='--', alpha=0.6)
            ax.plot(times, y_ngcm, label='NeuralGCM', color='green', linestyle='-.', alpha=0.6)
            ax.plot(times, y_corr, label='Strang', color='blue', linewidth=1.5)

            ax.set_title(f'{var_name} @ {level}hPa', fontsize=10)
            ax.grid(True, linestyle=':', alpha=0.6)

            imp = (y_phy.mean() - y_corr.mean()) / y_phy.mean() * 100
            ax.text(0.05, 0.95, f'Imp: {imp:.1f}%', transform=ax.transAxes, fontsize=8,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            if l_idx == 0:
                ax.set_ylabel('RMSE')
            if v_idx == num_vars - 1:
                ax.set_xlabel('Time Steps (h)')
            if v_idx == 0 and l_idx == num_levels - 1:
                ax.legend(loc='best', fontsize='small')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Figure saved to {save_path}")


def evaluate():
    # --- DDP init ---
    rank, device, world_size = setup_ddp()
    is_main = is_main_process()

    # --- Config ---
    BATCH_SIZE = 1

    TIME_STEPS = 73
    TIME_CHUNK = 4
    HRRR_STAT_PATH = os.fspath(pc.hrrr_stat_root())
    NGCM_CHECKPOINT = os.fspath(pc.ngcm_checkpoint(PROJECT_ROOT))
    TEST_DATA_ROOT = os.fspath(pc.era5_root())
    TEST_HRRR_ROOT = os.fspath(pc.hrrr_truth_root())

    RESULT_SAVE_DIR = os.fspath(PROJECT_ROOT / 'Test_Data_Rmse_00z')
    if is_main:
        os.makedirs(RESULT_SAVE_DIR, exist_ok=True)
        print(f"Config: batch={BATCH_SIZE}, world_size={world_size}")
        print(f"Mode: 00Z init only")

    # Find latest checkpoint
    CHECKPOINT_DIR = pc.film_checkpoint_dir(PROJECT_ROOT)
    CHECKPOINT_PATH = os.fspath(CHECKPOINT_DIR / 'model_ep0.pth')
    if CHECKPOINT_DIR.exists():
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
                except:
                    return (-1, -1)

            latest = sorted(checkpoints, key=parse_checkpoint)[-1]
            CHECKPOINT_PATH = os.fspath(CHECKPOINT_DIR / latest)
            if is_main:
                print(f"Using checkpoint: {CHECKPOINT_PATH}")

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

    # Dataset and distributed sampler
    ds_full = N_Dataset.ERA5XarrayDataset(TEST_DATA_ROOT, hrrr_root_dirs=[TEST_HRRR_ROOT], predict_lead_time=TIME_STEPS)

    # Keep 00Z init samples only
    indices_00z = filter_00z_indices(ds_full)
    ds = Subset00Z(ds_full, indices_00z)

    if is_main:
        print(f"Full dataset samples: {len(ds_full)}")
        print(f"00Z samples: {len(ds)} (filtered {len(ds) / len(ds_full) * 100:.1f}%)")

    # DistributedSampler for DDP
    is_ddp = world_size > 1
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=False) if is_ddp else None
    loader = DataLoader(
        ds, batch_size=BATCH_SIZE, num_workers=2, prefetch_factor=2,
        persistent_workers=True, pin_memory=True,
        collate_fn=N_Dataset.xarray_collate_fn, sampler=sampler
    )
    # loader = DataLoader(
    #     ds, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True,
    #     collate_fn=N_Dataset.xarray_collate_fn, sampler=sampler
    # )
    # --- Evaluation ---
    # Accumulate MSE, sqrt at end for RMSE
    mse_phy_sum = torch.zeros(TIME_STEPS, 20).to(device)
    mse_ngcm_sum = torch.zeros(TIME_STEPS, 20).to(device)
    mse_split_sum = torch.zeros(TIME_STEPS, 20).to(device)
    total_loss = torch.tensor(0.0).to(device)
    count = torch.tensor(0).to(device)

    if is_main:
        print("Start Evaluation (00Z only)...")

    with torch.inference_mode():
        for batch_idx, (input_list, time_list, hrrr_tensor) in enumerate(loader):
            if input_list is None:
                continue

            # NeuralGCM inference
            t_pred, t_phy = ngcm.forward(input_list, target_levels=[50, 500, 850, 1000],
                                         include_era5_label=False, region_lon=region_lon, region_lat=region_lat)
            t_pred[:, :4] /= 9.8
            t_phy[:, :4] /= 9.8
            t_pred = t_pred.permute(0, 4, 1, 2, 3).to(device)
            t_phy = t_phy.permute(0, 4, 1, 2, 3).to(device)
            gt_hrrr = hrrr_tensor.permute(0, 1, 2, 4, 3).to(device)[:, :, :20, :, :]

            B, T, C, W, H = t_phy.shape

            # Input normalization
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

            # Chunked time integration (Strang splitting)
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

            output = torch.cat(output_chunks, dim=1)

            # Loss
            output_flat = output.reshape(B * T, 20, W, H)
            gt_flat = gt_hrrr.reshape(B * T, 20, W, H)
            pred_list = [output_flat[i::T] for i in range(T)]
            gt_list = [gt_flat[i::T] for i in range(T)]
            batch_loss = loss_fn(pred_list, gt_list).item() * LOSS_SCALE
            total_loss += batch_loss / T

            # MSE accumulation -> RMSE = sqrt(mean(MSE))
            mse_phy = ((t_phy[:, :, :20, :, :] - gt_hrrr) ** 2).mean(dim=[0, 3, 4])
            mse_ngcm = ((t_pred[:, :, :20, :, :] - gt_hrrr) ** 2).mean(dim=[0, 3, 4])
            mse_split = ((output - gt_hrrr) ** 2).mean(dim=[0, 3, 4])

            mse_phy_sum += mse_phy
            mse_ngcm_sum += mse_ngcm
            mse_split_sum += mse_split
            count += 1

            # Main process only
            if is_main and batch_idx % 10 == 0:
                init_time = time_list[0] if time_list else "N/A"
                print(f"Rank {rank} | Batch {batch_idx}/{len(loader)} | Init: {init_time} | Loss: {batch_loss / T:.5f}")

            # Plot every 100 batches (main process)
            if is_main and (batch_idx + 1) % 100 == 0:
                avg_rmse_phy_tmp = torch.sqrt(mse_phy_sum / count).cpu().numpy()
                avg_rmse_ngcm_tmp = torch.sqrt(mse_ngcm_sum / count).cpu().numpy()
                avg_rmse_split_tmp = torch.sqrt(mse_split_sum / count).cpu().numpy()
                plot_rmse_comparison(
                    avg_rmse_phy_tmp.T, avg_rmse_ngcm_tmp.T, avg_rmse_split_tmp.T,
                    [50, 500, 850, 1000],
                    os.path.join(RESULT_SAVE_DIR, 'rmse_comparison_00z.png')
                )

    # --- Reduce across ranks ---
    if is_ddp:
        dist.all_reduce(mse_phy_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(mse_ngcm_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(mse_split_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)

    # --- Summary (main process) ---
    # RMSE = sqrt(mean(MSE))
    if is_main:
        avg_rmse_phy = torch.sqrt(mse_phy_sum / count).cpu().numpy()
        avg_rmse_ngcm = torch.sqrt(mse_ngcm_sum / count).cpu().numpy()
        avg_rmse_split = torch.sqrt(mse_split_sum / count).cpu().numpy()
        avg_loss = (total_loss / count).item()

        print(f"\n=== Evaluation Complete [00Z Only] ({count.item()} batches, {world_size} GPUs) ===")
        print(f"Average Loss: {avg_loss:.5f}")

        np.save(os.path.join(RESULT_SAVE_DIR, 'rmse_phy_00z.npy'), avg_rmse_phy)
        np.save(os.path.join(RESULT_SAVE_DIR, 'rmse_ngcm_00z.npy'), avg_rmse_ngcm)
        np.save(os.path.join(RESULT_SAVE_DIR, 'rmse_aero_00z.npy'), avg_rmse_split)

        plot_rmse_comparison(avg_rmse_phy.T, avg_rmse_ngcm.T, avg_rmse_split.T, [50, 500, 850, 1000],
                             os.path.join(RESULT_SAVE_DIR, 'rmse_comparison_00z.png'))

        # Print improvement vs physics
        var_names = ['Z50', 'Z500', 'Z850', 'Z1000', 'T50', 'T500', 'T850', 'T1000']
        print("\nRMSE Improvement (vs Physics) [00Z Only]:")
        print(f"{'Variable':<10} | {'Physics':<8} | {'NeuralGCM':<9} | {'Strang':<9} | {'Imp(%)':<6}")
        print("-" * 60)
        for i, name in enumerate(var_names):
            v_phy = avg_rmse_phy[:, i].mean()
            v_ngcm = avg_rmse_ngcm[:, i].mean()
            v_split = avg_rmse_split[:, i].mean()
            imp = (v_phy - v_split) / v_phy * 100
            print(f"{name:<10} | {v_phy:.4f}   | {v_ngcm:.4f}    | {v_split:.4f}    | {imp:.1f}%")

    cleanup_ddp()


if __name__ == "__main__":
    try:
        evaluate()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        cleanup_ddp()
