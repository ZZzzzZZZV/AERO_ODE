"""
    python test_film_00z_RMSE.py
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
AERO_ODE_ROOT = PROJECT_ROOT.parent
if os.fspath(AERO_ODE_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(AERO_ODE_ROOT))
import paths_config as pc

# GPU visibility (must run before importing torch)
AVAILABLE_GPUS = [0]  # Change this to select GPUs


def setup_gpu_visibility():
    """Set GPU visibility for DDP/torchrun."""
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        if local_rank < len(AVAILABLE_GPUS):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(AVAILABLE_GPUS[local_rank])
        else:
            raise RuntimeError(f"LOCAL_RANK={local_rank} exceeds available GPU count")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, AVAILABLE_GPUS))


setup_gpu_visibility()

# JAX memory settings
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".30"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from film_inference_surface import DiagnosisModel, StaticDataManager, compute_time_indices, compute_solar_features, TARGET_LEVELS
from l2 import RelativeL2Loss
import AERO_Dataset_v3 as N_Dataset
import AERO_v3 as NGCM


def setup_ddp():
    """Initialize DDP environment."""
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
    """Return indices of 00Z initialization samples only."""
    filtered_indices = []
    for i, global_idx in enumerate(dataset.valid_indices):
        timestamp = dataset.timeline[global_idx]['timestamp']
        if timestamp.hour == 0:
            filtered_indices.append(i)
    return filtered_indices


class Subset00Z(torch.utils.data.Dataset):
    """Dataset wrapper containing 00Z initialization samples only."""

    def __init__(self, dataset: N_Dataset.ERA5XarrayDataset, indices: list):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def plot_surface_rmse(rmse_pred, save_path, time_steps):
    """Plot surface-variable RMSE vs lead time."""
    var_names = ['MSLP (hPa)', 'U10 (m/s)', 'V10 (m/s)', 'T2m (K)']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    times = np.arange(time_steps)

    for idx, (ax, var_name) in enumerate(zip(axes.flat, var_names)):
        y_pred = rmse_pred[idx, :]

        ax.plot(times, y_pred, color='blue', linewidth=2, marker='o', markersize=3)

        ax.set_title(f'{var_name}', fontsize=12)
        ax.set_xlabel('Lead Time (hours)')
        ax.set_ylabel('RMSE')
        ax.grid(True, linestyle=':', alpha=0.6)

        avg_rmse = y_pred.mean()
        ax.axhline(y=avg_rmse, color='red', linestyle='--', alpha=0.7, label=f'Mean: {avg_rmse:.4f}')
        ax.legend(loc='best')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Figure saved to {save_path}")


def evaluate():
    rank, device, world_size = setup_ddp()
    is_main = is_main_process()

    BATCH_SIZE = 1
    TIME_STEPS = 73
    TIME_CHUNK = 8 # 8

    HRRR_STAT_PATH = os.fspath(pc.hrrr_stat_root())
    STATIC_DATA_PATH = os.fspath(pc.surface_static_data(PROJECT_ROOT))
    NGCM_CHECKPOINT = os.fspath(pc.ngcm_checkpoint(PROJECT_ROOT))
    TEST_DATA_ROOT = os.fspath(pc.era5_root())
    TEST_HRRR_ROOT = os.fspath(pc.hrrr_truth_root())

    RESULT_SAVE_DIR = os.fspath(PROJECT_ROOT / 'Test_Surface_Rmse_00z')
    if is_main:
        os.makedirs(RESULT_SAVE_DIR, exist_ok=True)
        print(f"Config: batch={BATCH_SIZE}, world_size={world_size}")
        print("Mode: 00Z initialization only, surface variable diagnosis")

    CHECKPOINT_DIR = os.fspath(pc.film_checkpoint_dir(PROJECT_ROOT, 'checkpoints_film_v2'))
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
                except:
                    return (-1, -1)

            latest = sorted(checkpoints, key=parse_checkpoint)[-1]
            CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, latest)
            if is_main:
                print(f"Using checkpoint: {CHECKPOINT_PATH}")

    if CHECKPOINT_PATH is None:
        raise FileNotFoundError(f"No checkpoint found in {CHECKPOINT_DIR}")

    if is_main:
        print("Initializing NeuralGCM...")
    ngcm = NGCM.NeuralGCMInference(NGCM_CHECKPOINT, inner_steps=1, outer_steps=TIME_STEPS)

    region_lat = np.load(os.path.join(HRRR_STAT_PATH, 'lats.npy')).T
    region_lon = np.load(os.path.join(HRRR_STAT_PATH, 'lons.npy')).T

    if is_main:
        print("Loading Diagnosis Model...")

    model = DiagnosisModel(
        in_channels_dynamic=74,
        in_channels_static=14,
        out_channels=4,
        time_emb_dim=128,
        backbone='unet3plus'
    ).to(device)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    state_dict = checkpoint['model_state_dict']
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()

    static_manager = StaticDataManager(
        HRRR_STAT_PATH,
        device=device,
        target_levels=TARGET_LEVELS,
        static_data_path=STATIC_DATA_PATH
    )

    loss_fn = RelativeL2Loss(climatological_std=static_manager.std[20:24], humidity_weight=1.0)

    ds_full = N_Dataset.ERA5XarrayDataset(
        TEST_DATA_ROOT,
        hrrr_root_dirs=[TEST_HRRR_ROOT],
        predict_lead_time=TIME_STEPS
    )

    indices_00z = filter_00z_indices(ds_full)
    ds = Subset00Z(ds_full, indices_00z)

    if is_main:
        print(f"Full dataset samples: {len(ds_full)}")
        print(f"00Z samples: {len(ds)} (filter rate: {len(ds) / len(ds_full) * 100:.1f}%)")

    is_ddp = world_size > 1
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=False) if is_ddp else None
    loader = DataLoader(
        ds, batch_size=BATCH_SIZE, num_workers=2, prefetch_factor=2,
        persistent_workers=True, pin_memory=True,
        collate_fn=N_Dataset.xarray_collate_fn, sampler=sampler
    )

    mse_sum = torch.zeros(4, TIME_STEPS).to(device)
    total_loss = torch.tensor(0.0).to(device)
    count = torch.tensor(0).to(device)

    if is_main:
        print("Start Evaluation (00Z only, Surface Variables)...")

    with torch.inference_mode():
        for batch_idx, (input_list, time_list, hrrr_tensor) in enumerate(loader):
            if input_list is None:
                continue

            t_pred, t_phy = ngcm.forward(
                input_list,
                target_levels=TARGET_LEVELS,
                include_era5_label=False,
                region_lon=region_lon,
                region_lat=region_lat
            )

            t_pred[:, :5] /= 9.8
            t_phy[:, :5] /= 9.8

            t_pred = t_pred.permute(0, 4, 1, 2, 3).to(device)
            t_phy = t_phy.permute(0, 4, 1, 2, 3).to(device)

            gt = hrrr_tensor.permute(0, 1, 2, 4, 3).to(device)[:, :, 20:24]

            B, T, C, W, H = t_phy.shape

            t_phy_norm = (t_phy - static_manager.input_mean) / (static_manager.input_std + 1e-9)
            corr_norm = (t_pred - t_phy) / (static_manager.input_std + 1e-9)

            solar_feat = compute_solar_features(time_list, T, region_lat, region_lon, device)

            dyn_in = torch.cat([t_phy_norm, corr_norm, solar_feat], dim=2)
            static_in = static_manager.get_static_input(B)
            steps, hours, months = compute_time_indices(time_list, T, device)

            pred_chunks = []
            for t0 in range(0, T, TIME_CHUNK):
                t1 = min(t0 + TIME_CHUNK, T)
                tc = t1 - t0

                dyn = dyn_in[:, t0:t1].reshape(B * tc, 74, W, H)
                n_static = static_in.shape[1]
                sta = static_in.unsqueeze(1).expand(-1, tc, -1, -1, -1).reshape(B * tc, n_static, W, H)
                s = steps.view(B, T)[:, t0:t1].reshape(-1)
                h = hours.view(B, T)[:, t0:t1].reshape(-1)
                m = months.view(B, T)[:, t0:t1].reshape(-1)

                output_norm = model(dyn, sta, s, h, m)
                pred = output_norm * (static_manager.surface_std + 1e-9) + static_manager.surface_mean
                pred_chunks.append(pred.reshape(B, tc, 4, W, H))

            pred_all = torch.cat(pred_chunks, dim=1)

            pred_flat = pred_all.reshape(B * T, 4, W, H)
            gt_flat = gt.reshape(B * T, 4, W, H)
            pred_list = [pred_flat[i::T] for i in range(T)]
            gt_list = [gt_flat[i::T] for i in range(T)]
            batch_loss = loss_fn(pred_list, gt_list).item()
            total_loss += batch_loss / T

            mse = ((pred_all - gt) ** 2).mean(dim=[0, 3, 4])
            mse_sum += mse.T
            count += 1

            if is_main and batch_idx % 10 == 0:
                init_time = time_list[0] if time_list else "N/A"
                print(f"Batch {batch_idx}/{len(loader)} | Init: {init_time} | Loss: {batch_loss / T:.5f}")

            if is_main and (batch_idx + 1) % 50 == 0:
                avg_rmse = torch.sqrt(mse_sum / count).cpu().numpy()
                plot_surface_rmse(avg_rmse, os.path.join(RESULT_SAVE_DIR, 'rmse_surface_00z.png'), TIME_STEPS)

    if is_ddp:
        dist.all_reduce(mse_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)

    if is_main:
        avg_rmse = torch.sqrt(mse_sum / count).cpu().numpy()
        avg_loss = (total_loss / count).item()

        print(f"\n=== Evaluation Complete [00Z Only, Surface Variables] ({count.item()} batches, {world_size} GPUs) ===")
        print(f"Average RelativeL2 Loss: {avg_loss:.5f}")

        np.save(os.path.join(RESULT_SAVE_DIR, 'rmse_surface_00z.npy'), avg_rmse)

        plot_surface_rmse(avg_rmse, os.path.join(RESULT_SAVE_DIR, 'rmse_surface_00z.png'), TIME_STEPS)

        var_names = ['MSLP', 'U10', 'V10', 'T2m']
        var_units = ['hPa', 'm/s', 'm/s', 'K']

        print("\n" + "=" * 50)
        print("Surface Variable RMSE Summary [00Z Only]")
        print("=" * 50)
        print(f"{'Variable':<10} | {'Mean RMSE':<15} | {'Unit':<8}")
        print("-" * 50)

        for i, (name, unit) in enumerate(zip(var_names, var_units)):
            mean_rmse = avg_rmse[i, :].mean()
            print(f"{name:<10} | {mean_rmse:<15.4f} | {unit:<8}")

        print("=" * 50)

        print("\nRMSE by Lead Time (selected hours):")
        print(f"{'Variable':<10} | {'6h':<8} | {'12h':<8} | {'24h':<8} | {'36h':<8} | {'48h':<8}")
        print("-" * 60)
        for i, name in enumerate(var_names):
            rmse_6h = avg_rmse[i, 5] if TIME_STEPS > 5 else 0
            rmse_12h = avg_rmse[i, 11] if TIME_STEPS > 11 else 0
            rmse_24h = avg_rmse[i, 23] if TIME_STEPS > 23 else 0
            rmse_36h = avg_rmse[i, 35] if TIME_STEPS > 35 else 0
            rmse_48h = avg_rmse[i, 47] if TIME_STEPS > 47 else 0
            print(f"{name:<10} | {rmse_6h:<8.4f} | {rmse_12h:<8.4f} | {rmse_24h:<8.4f} | {rmse_36h:<8.4f} | {rmse_48h:<8.4f}")

    cleanup_ddp()


if __name__ == "__main__":
    try:
        evaluate()
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        cleanup_ddp()
