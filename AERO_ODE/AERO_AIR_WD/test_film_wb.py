import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(next(_p for _p in _pathlib.Path(__file__).resolve().parents
                             if (_p / "common" / "paths.py").is_file())))
from common.paths import ROOT

import os

import numpy as np
import matplotlib.pyplot as plt

AVAILABLE_GPUS = [2, 3]
DEFAULT_RESULT_DIR = str(ROOT / "AERO_AIR_WD/Test_Data_Rmse_00z_wb")

def setup_gpu_visibility():
    ddp = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if not ddp and os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        return
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        if local_rank < len(AVAILABLE_GPUS):
            os.environ["CUDA_VISIBLE_DEVICES"] = str(AVAILABLE_GPUS[local_rank])
        else:
            raise RuntimeError(f"LOCAL_RANK={local_rank} is outside the available GPU list")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, AVAILABLE_GPUS))

setup_gpu_visibility()

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".30"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from inference_components import CorrectionModel, StaticDataManager, compute_time_indices
from l2 import RelativeL2Loss
import AERO_ODE_Dataset_v3 as AERO_Dataset
import AERO_ODE_v3 as AERO

DT_DYNAMICS_S: float = 3600.0

def _resolved_dynamics_tendency(state_pre, state_post, dt):
    return (state_post - state_pre) / dt

def _subgrid_tendency(increment, dt):
    return increment / dt

class StrangSplittingIntegrator:

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

def setup_ddp():
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

def filter_00z_indices(dataset: AERO_Dataset.ERA5XarrayDataset):
    filtered_indices = []
    for i, global_idx in enumerate(dataset.valid_indices):
        timestamp = dataset.timeline[global_idx]['timestamp']
        if timestamp.hour == 0:
            filtered_indices.append(i)
    return filtered_indices

class Subset00Z(torch.utils.data.Dataset):

    def __init__(self, dataset: AERO_Dataset.ERA5XarrayDataset, indices: list):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

TARGET_LEVELS = [300, 500, 700, 850, 925]

def plot_aero_rmse(rmse_aero, target_levels, save_path):
    var_names = ['Geopotential Height', 'Temperature', 'Specific Humidity', 'U Component', 'V Component']
    num_vars, num_levels = len(var_names), len(target_levels)

    fig, axes = plt.subplots(nrows=num_vars, ncols=num_levels, figsize=(4 * num_levels, 3 * num_vars))
    n_lead = rmse_aero.shape[1]
    times = np.arange(1, n_lead + 1)

    for v_idx, var_name in enumerate(var_names):
        for l_idx, level in enumerate(target_levels):
            ax = axes[v_idx, l_idx]
            ch_idx = v_idx * num_levels + l_idx

            ax.plot(times, rmse_aero[ch_idx, :], label='AERO-ODE', color='blue', linewidth=1.5)

            ax.set_title(f'{var_name} @ {level}hPa', fontsize=10)
            ax.grid(True, linestyle=':', alpha=0.6)

            if l_idx == 0:
                ax.set_ylabel('RMSE')
            if v_idx == num_vars - 1:
                ax.set_xlabel('Lead Time (hours)')
            if v_idx == 0 and l_idx == num_levels - 1:
                ax.legend(loc='best', fontsize='small')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"Figure saved to {save_path}")


def compute_rmse(*, write_dir=None, verbose=True):
    rank, device, world_size = setup_ddp()
    is_main = is_main_process()
    log = verbose and is_main

    BATCH_SIZE = 1
    TIME_STEPS = int(os.environ.get("QUICK_TIME_STEPS", "73"))
    TIME_CHUNK = int(os.environ.get("QUICK_TIME_CHUNK", "4"))
    MAX_BATCHES = int(os.environ.get("QUICK_MAX_BATCHES", "0"))
    HRRR_STAT_PATH = os.environ.get(
        "TEST_HRRR_STAT", str(ROOT / "AERO_AIR_WD/Hrrr_rb")
    )
    STAT_PATH = os.environ.get(
        "TEST_STAT_PATH", str(ROOT / "data/hrrr_stat")
    )
    NGCM_CHECKPOINT = os.environ.get(
        "TEST_NGCM_CHECKPOINT",
        str(
            ROOT / "shared_assets/NeuralGCM_Weights"
            / "neuralgcm_04_30_2024_neural_gcm_dynamic_forcing_deterministic_1_4_deg.pkl"
        ),
    )
    TEST_DATA_ROOT = os.environ.get(
        "TEST_DATA_ROOT", str(ROOT / "data/era5_test"),
    )
    TEST_HRRR_ROOT = os.environ.get(
        "TEST_HRRR_ROOT", str(ROOT / "data/hrrr_test"),
    )

    try:
        if write_dir and is_main:
            os.makedirs(write_dir, exist_ok=True)

        if log:
            print(f"config: batch={BATCH_SIZE}, world_size={world_size}")
            print(f"mode: 00 UTC inits only")

        CHECKPOINT_PATH = os.environ.get('TEST_CHECKPOINT', '').strip() or str(
            ROOT / "AERO_AIR_WD/checkpoints_film_25/model_ep5.pth"
        )
        if not os.path.isfile(CHECKPOINT_PATH):
            raise FileNotFoundError(f"checkpoint not found: {CHECKPOINT_PATH}")
        if log:
            print(f"Using checkpoint: {CHECKPOINT_PATH}")

        ngcm = AERO.NeuralGCMInference(NGCM_CHECKPOINT, inner_steps=1, outer_steps=TIME_STEPS)
        region_lat = np.load(os.path.join(HRRR_STAT_PATH, 'hrrr_west_lat.npy')).T
        region_lon = np.load(os.path.join(HRRR_STAT_PATH, 'hrrr_west_lon.npy')).T

        model = CorrectionModel(time_emb_dim=128, backbone='unet3plus').to(device)
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
        except Exception:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        static_manager = StaticDataManager(HRRR_STAT_PATH, stat_dir=STAT_PATH, device=device)
        integrator = StrangSplittingIntegrator(dt=DT_DYNAMICS_S)
        loss_fn = RelativeL2Loss()
        LOSS_SCALE = 1.0

        ds_full = AERO_Dataset.ERA5XarrayDataset(
            TEST_DATA_ROOT,
            hrrr_root_dirs=[TEST_HRRR_ROOT],
            predict_lead_time=TIME_STEPS,
        )

        indices_00z = filter_00z_indices(ds_full)
        ds = Subset00Z(ds_full, indices_00z)

        is_ddp = world_size > 1
        sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=False) if is_ddp else None
        loader = DataLoader(
            ds, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True,
            collate_fn=AERO_Dataset.xarray_collate_fn, sampler=sampler
        )

        mse_aero_sum = torch.zeros(TIME_STEPS, 25).to(device)
        total_loss = torch.tensor(0.0).to(device)
        count = torch.tensor(0).to(device)

        if log:
            print("Start Evaluation (00Z only)...")

        with torch.inference_mode():
            for batch_idx, (input_list, time_list, hrrr_tensor) in enumerate(loader):
                if MAX_BATCHES > 0 and batch_idx >= MAX_BATCHES:
                    if log:
                        print(f"[QUICK] reached MAX_BATCHES={MAX_BATCHES}, stopping early")
                    break
                if input_list is None:
                    continue

                t_pred, t_phy = ngcm.forward(input_list, target_levels=[300, 500, 700, 850, 925],
                                             include_era5_label=False, region_lon=region_lon, region_lat=region_lat)
                t_pred[:, :5] /= 9.8
                t_phy[:, :5] /= 9.8
                t_pred = t_pred.permute(0, 4, 1, 2, 3).to(device)
                t_phy = t_phy.permute(0, 4, 1, 2, 3).to(device)
                gt_hrrr = hrrr_tensor.permute(0, 1, 2, 4, 3).to(device)[:, :, :25, :, :]

                B, T, C, W, H = t_phy.shape

                input_std = torch.ones(1, 1, 35, 1, 1, device=device)
                input_mean = torch.zeros(1, 1, 35, 1, 1, device=device)
                input_std[0, 0, :25, 0, 0] = static_manager.std[:25]
                input_mean[0, 0, :25, 0, 0] = static_manager.mean[:25]
                input_std[0, 0, 25:30, 0, 0] = 7.538e-06
                input_std[0, 0, 30:35, 0, 0] = 1.979e-05

                t_phy_norm = (t_phy - input_mean) / (input_std + 1e-9)
                correction_norm = (t_pred - t_phy) / (input_std + 1e-9)
                dyn_in = torch.cat([t_phy_norm, correction_norm], dim=2)
                static_in = static_manager.get_static_input(B)
                steps_all, hours_all, months_all = compute_time_indices(time_list, T, device)

                output_chunks = []
                for t0 in range(0, T, TIME_CHUNK):
                    t1 = min(t0 + TIME_CHUNK, T)
                    tc = t1 - t0

                    dyn = dyn_in[:, t0:t1].reshape(B * tc, 70, W, H)
                    sta = static_in.unsqueeze(1).expand(-1, tc, -1, -1, -1).reshape(B * tc, 5, W, H)
                    s = steps_all.view(B, T)[:, t0:t1].reshape(-1)
                    h = hours_all.view(B, T)[:, t0:t1].reshape(-1)
                    m = months_all.view(B, T)[:, t0:t1].reshape(-1)

                    sgs_increment = model(dyn, sta, s, h, m) * (static_manager.target_std + 1e-9)

                    state_post = t_phy[:, t0:t1, :25].reshape(B * tc, 25, W, H)
                    if t0 == 0:
                        state_pre = state_post
                    else:
                        state_pre = t_phy[:, t0 - 1:t1 - 1, :25].reshape(B * tc, 25, W, H)

                    state_next = integrator.step(state_pre, state_post, sgs_increment)
                    output_chunks.append(state_next.reshape(B, tc, 25, W, H))

                output = torch.cat(output_chunks, dim=1)

                output_flat = output.reshape(B * T, 25, W, H)
                gt_flat = gt_hrrr.reshape(B * T, 25, W, H)
                pred_list = [output_flat[i::T] for i in range(T)]
                gt_list = [gt_flat[i::T] for i in range(T)]
                batch_loss = loss_fn(pred_list, gt_list).item() * LOSS_SCALE
                total_loss += batch_loss / T

                mse_aero_sum += ((output - gt_hrrr) ** 2).mean(dim=[0, 3, 4])
                count += 1

                if log and (batch_idx % 10 == 0 or batch_idx + 10 >= len(loader)):
                    init_time = time_list[0] if time_list else "N/A"
                    print(
                        f"Rank {rank} | Batch {batch_idx}/{len(loader)} | Init: {init_time} | "
                        f"Loss: {batch_loss / T:.5f}",
                        flush=True,
                    )

                if write_dir and is_main and (batch_idx + 1) % 100 == 0:
                    avg_rmse_aero_tmp = torch.sqrt(mse_aero_sum / count).cpu().numpy()
                    plot_aero_rmse(
                        avg_rmse_aero_tmp[1:].T,
                        [300, 500, 700, 850, 925],
                        os.path.join(write_dir, 'rmse_aero_00z.png')
                    )

        if log:
            print(f"Rank {rank} finished loader loop (local batches={int(count.item())})", flush=True)

        if is_ddp:
            if log:
                print(f"Rank {rank} all_reduce start...", flush=True)
            for t in (mse_aero_sum, total_loss, count):
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
            if log:
                print(f"Rank {rank} all_reduce done", flush=True)

        avg_rmse_aero = torch.sqrt(mse_aero_sum / count).cpu().numpy()
        if log:
            avg_loss = (total_loss / count).item()
            print(f"\n=== Evaluation Complete [00Z Only] ({count.item()} batches, {world_size} GPUs) ===")
            print(f"Average Loss: {avg_loss:.5f}")
        return avg_rmse_aero if is_main else None
    finally:
        cleanup_ddp()


def save_rmse_outputs(avg_rmse_aero, result_save_dir):
    os.makedirs(result_save_dir, exist_ok=True)
    np.save(os.path.join(result_save_dir, 'rmse_aero_00z.npy'), avg_rmse_aero)
    plot_aero_rmse(
        avg_rmse_aero[1:].T,
        TARGET_LEVELS,
        os.path.join(result_save_dir, 'rmse_aero_00z.png'),
    )

    var_names = ['Z300', 'Z500', 'Z700', 'Z850', 'Z925', 'T300', 'T500', 'T700', 'T850', 'T925']

    def _print_table(title, aero):
        print(f"\n{title}")
        print(f"{'Variable':<10} | {'AERO-ODE':<10}")
        print("-" * 24)
        for i, name in enumerate(var_names):
            print(f"{name:<10} | {aero[:, i].mean():.4f}")
        print(f"{'MEAN':<10} | {aero.mean():.4f}")

    _print_table("AERO-ODE RMSE:", avg_rmse_aero[1:])


def evaluate():
    result_dir = os.environ.get("QUICK_RESULT_DIR", DEFAULT_RESULT_DIR)
    avg_rmse_aero = compute_rmse(write_dir=result_dir, verbose=True)
    if avg_rmse_aero is not None:
        save_rmse_outputs(avg_rmse_aero, result_dir)
    return avg_rmse_aero

if __name__ == "__main__":
    try:
        evaluate()
    except Exception as e:
        print(f"test failed: {e}")
        import traceback
        traceback.print_exc()
        cleanup_ddp()
