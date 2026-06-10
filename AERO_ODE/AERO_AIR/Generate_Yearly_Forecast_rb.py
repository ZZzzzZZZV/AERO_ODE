"""
    python Generate_Yearly_Forecast_rb.py
"""

import os
import sys

# ============================================
# GPU setup (before any imports)
# ============================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["LOCAL_RANK"] = "0"

# JAX memory settings
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".30"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
AERO_ODE_ROOT = PROJECT_ROOT.parent
sys.path.append(os.fspath(PROJECT_ROOT))
if os.fspath(AERO_ODE_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(AERO_ODE_ROOT))
import paths_config as pc

import torch
import numpy as np
import h5py
import xarray as xr
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
from tqdm import tqdm

from film_inference_air import (
    CorrectionModel, StaticDataManager, compute_time_indices,
    StrangSplittingIntegrator, DT_DYNAMICS_S,
)
import AERO_v3 as NGCM


# ============================================
# Config
# ============================================
class Config:
    """Forecast generation config."""

    TIME_STEPS = 73  # includes IC; saved output drops t=0

    NGCM_CHECKPOINT = os.fspath(pc.ngcm_checkpoint(PROJECT_ROOT))
    CORRECTION_CKPT = os.fspath(pc.film_checkpoint_dir(PROJECT_ROOT) / 'model_ep3.pth')

    HRRR_STAT_PATH = os.fspath(pc.hrrr_stat_root())
    ERA5_ROOT = os.fspath(pc.era5_root())

    OUTPUT_ROOT = os.fspath(pc.predicted_output_root(PROJECT_ROOT))

    TIME_CHUNK = 8  # chunk time steps to save VRAM
    DEVICE = 'cuda:0'

    SKIP_EXISTING = True


def find_era5_00z_files(era5_root: str) -> List[Tuple[str, datetime]]:
    """
    Scan ERA5 tree for daily nc files.

    Args:
        era5_root: ERA5 root, layout test/YYYY/MM/DD/YYYYMMDD.nc

    Returns:
        list of tuples: [(file_path, date), ...] sorted by date
    """
    file_dates = []

    for root, _, files in os.walk(era5_root):
        for fname in files:
            if not fname.endswith('.nc'):
                continue

            date_str = fname.replace('.nc', '')
            if len(date_str) != 8 or not date_str.isdigit():
                continue

            try:
                dt = datetime.strptime(date_str, "%Y%m%d")
                fpath = os.path.join(root, fname)
                file_dates.append((fpath, dt))
            except ValueError:
                continue

    file_dates.sort(key=lambda x: x[1])

    return file_dates


def load_era5_00z_initial_state(era5_file: str) -> Tuple[Optional[xr.Dataset], Optional[str], bool]:
    """
    Load ERA5 00Z initial condition.

    Args:
        era5_file: ERA5 nc path

    Returns:
        ds0: xarray.Dataset at time=0 (time dim length=1)
        time_str: 'YYYY/MM/DD/HH'
        valid: whether IC is valid
    """
    ds = None
    try:
        ds = xr.open_dataset(era5_file, decode_timedelta=False)

        times = ds['time'].values
        if len(times) == 0:
            ds.close()
            return None, None, False

        t0 = np.datetime64(times[0], 's')
        dt = datetime.utcfromtimestamp(int(t0.astype('int64')))

        if dt.hour != 0:
            print(f"Warning: {era5_file} first time is not 00Z (hour={dt.hour}), skip")
            ds.close()
            return None, None, False

        ds0 = ds.isel(time=slice(0, 1))
        time_str = dt.strftime("%Y/%m/%d/%H")

        return ds0, time_str, True

    except Exception as e:
        print(f"Error loading {era5_file}: {e}")
        if ds is not None:
            try:
                ds.close()
            except:
                pass
        return None, None, False


class YearlyForecaster:
    """
    Yearly forecast generator.

    NeuralGCM + CorrectionModel inference pipeline.
    """

    def __init__(self, config: Config):
        self.cfg = config
        self.device = torch.device(config.DEVICE)

        print(">>> Loading Static Data...")
        self.static_manager = StaticDataManager(config.HRRR_STAT_PATH, device=self.device)

        self.region_lat = np.load(os.path.join(config.HRRR_STAT_PATH, 'lats.npy')).T
        self.region_lon = np.load(os.path.join(config.HRRR_STAT_PATH, 'lons.npy')).T

        print(">>> Loading NeuralGCM...")
        self.ngcm = NGCM.NeuralGCMInference(
            config.NGCM_CHECKPOINT,
            inner_steps=1,
            outer_steps=config.TIME_STEPS
        )

        print(">>> Loading Correction Model...")
        self.correction_model = CorrectionModel(
            time_emb_dim=128,
            backbone='unet3plus'
        ).to(self.device)

        checkpoint = torch.load(config.CORRECTION_CKPT, map_location=self.device)
        state_dict = checkpoint['model_state_dict']

        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k.replace('module.', '')] = v
        self.correction_model.load_state_dict(new_state_dict)
        self.correction_model.eval()

        self.input_std = torch.ones(1, 1, 28, 1, 1, device=self.device)
        self.input_mean = torch.zeros(1, 1, 28, 1, 1, device=self.device)
        self.input_std[0, 0, :20, 0, 0] = self.static_manager.std[:20]
        self.input_mean[0, 0, :20, 0, 0] = self.static_manager.mean[:20]
        self.input_std[0, 0, 20:24, 0, 0] = 7.538e-06
        self.input_std[0, 0, 24:28, 0, 0] = 1.979e-05

        self.integrator = StrangSplittingIntegrator(dt=DT_DYNAMICS_S)

        print(">>> Model initialization complete!")

    def inference(self, input_list: List[xr.Dataset], time_list: List[str]) -> torch.Tensor:
        """
        Full pipeline: NeuralGCM physics core -> Strang splitting -> output.

        Args:
            input_list: [ds0] ERA5 IC
            time_list: ['YYYY/MM/DD/HH'] init time strings

        Returns:
            output: (1, T, 20, 440, 408) forecast
        """
        B = len(input_list)
        time_chunk = self.cfg.TIME_CHUNK

        with torch.inference_mode():
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

            B, T, C, W, H = t_phy.shape

            t_phy_norm = (t_phy - self.input_mean) / (self.input_std + 1e-9)
            correction_norm = (t_pred - t_phy) / (self.input_std + 1e-9)
            dyn_in = torch.cat([t_phy_norm, correction_norm], dim=2)

            static_in = self.static_manager.get_static_input(B)
            steps_all, hours_all, months_all = compute_time_indices(time_list, T, self.device)

            output_chunks = []
            for t0 in range(0, T, time_chunk):
                t1 = min(t0 + time_chunk, T)
                tc = t1 - t0

                dyn = dyn_in[:, t0:t1].reshape(B * tc, 56, W, H)
                sta = static_in.unsqueeze(1).expand(-1, tc, -1, -1, -1).reshape(B * tc, 5, W, H)
                s = steps_all.view(B, T)[:, t0:t1].reshape(-1)
                h = hours_all.view(B, T)[:, t0:t1].reshape(-1)
                m = months_all.view(B, T)[:, t0:t1].reshape(-1)

                sgs_increment = self.correction_model(dyn, sta, s, h, m) * (self.static_manager.target_std + 1e-9)

                state_post = t_phy[:, t0:t1, :20].reshape(B * tc, 20, W, H)
                if t0 == 0:
                    state_pre = state_post
                else:
                    state_pre = t_phy[:, t0 - 1:t1 - 1, :20].reshape(B * tc, 20, W, H)

                state_next = self.integrator.step(state_pre, state_post, sgs_increment)
                output_chunks.append(state_next.reshape(B, tc, 20, W, H))

            output = torch.cat(output_chunks, dim=1)

            output = output.permute(0, 1, 2, 4, 3)

            return output

    def save_forecast(self, forecast_tensor: torch.Tensor, date: datetime, output_root: str) -> str:
        """
        Save forecast to h5 (drop IC; keep t=1..48).

        Args:
            forecast_tensor: (1, T, 20, 440, 408), T includes IC
            date: init date
            output_root: output root

        Returns:
            output_path: saved file path
        """
        year = date.strftime('%Y')
        month = date.strftime('%m')
        day = date.strftime('%d')

        output_dir = os.path.join(output_root, year, month)
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, f'{day}.h5')

        data = forecast_tensor[0, 1:, :, :, :].cpu().numpy()  # drop t=0 IC

        with h5py.File(output_path, 'w') as f:
            f.create_dataset(
                'fields',
                data=data,
                compression='gzip',
                compression_opts=4
            )
            f.attrs['init_time'] = date.strftime('%Y-%m-%d %H:%M:%S UTC')
            f.attrs['forecast_hours'] = 48
            f.attrs['variables'] = [
                'Z50', 'Z500', 'Z850', 'Z1000',
                'T50', 'T500', 'T850', 'T1000',
                'S50', 'S500', 'S850', 'S1000',
                'U50', 'U500', 'U850', 'U1000',
                'V50', 'V500', 'V850', 'V1000'
            ]
            f.attrs['shape'] = '(time_steps, variables, latitude, longitude)'
            f.attrs['note'] = 'time_steps=1~48h forecast (initial state t=0 excluded)'

        return output_path


def main():
    """Main entry."""
    cfg = Config()

    print("=" * 70)
    print("AERO-AIR yearly forecast generator")
    print("=" * 70)
    print(f"Inference steps: {cfg.TIME_STEPS} (00Z + 72 h)")
    print(f"Saved steps: 72 (drop IC t=0)")
    print(f"Output shape: (72, 20, 440, 408)")
    print(f"Input: {cfg.ERA5_ROOT}")
    print(f"Output: {cfg.OUTPUT_ROOT}")
    print(f"Skip existing: {cfg.SKIP_EXISTING}")
    print("=" * 70)

    print("\n>>> Scanning ERA5 initial conditions...")
    file_dates = find_era5_00z_files(cfg.ERA5_ROOT)
    print(f"Found {len(file_dates)} ERA5 files")

    if len(file_dates) == 0:
        print("Error: no ERA5 files found")
        print(f"Check path: {cfg.ERA5_ROOT}")
        return

    first_date = file_dates[0][1]
    last_date = file_dates[-1][1]
    print(f"Date range: {first_date.strftime('%Y-%m-%d')} ~ {last_date.strftime('%Y-%m-%d')}")

    print("\n>>> Initializing forecast model...")
    forecaster = YearlyForecaster(cfg)

    success_count = 0
    skip_count = 0
    error_count = 0

    print("\n>>> Generating forecasts...")
    pbar = tqdm(file_dates, desc="Forecasting", unit="day")

    for era5_file, date in pbar:
        date_str = date.strftime('%Y-%m-%d')
        pbar.set_postfix_str(f"Processing: {date_str}")

        try:
            year = date.strftime('%Y')
            month = date.strftime('%m')
            day = date.strftime('%d')
            output_path = os.path.join(cfg.OUTPUT_ROOT, year, month, f'{day}.h5')

            if cfg.SKIP_EXISTING and os.path.exists(output_path):
                skip_count += 1
                continue

            ds0, time_str, valid = load_era5_00z_initial_state(era5_file)

            if not valid:
                print(f"\nSkip {date_str}: invalid IC")
                skip_count += 1
                continue

            input_list = [ds0]
            time_list = [time_str]

            forecast = forecaster.inference(input_list, time_list)

            if torch.isnan(forecast).any():
                print(f"\nWarning: {date_str} forecast has NaN, skip")
                ds0.close()
                error_count += 1
                continue

            expected_shape = (1, cfg.TIME_STEPS, 20, 440, 408)
            if forecast.shape != expected_shape:
                print(f"\nWarning: {date_str} shape {forecast.shape} != expected {expected_shape}")

            saved_path = forecaster.save_forecast(forecast, date, cfg.OUTPUT_ROOT)
            success_count += 1

            ds0.close()

            if success_count % 10 == 0:
                torch.cuda.empty_cache()

            if success_count % 30 == 0:
                pbar.write(f"Progress: ok {success_count}, skip {skip_count}, err {error_count}")

        except Exception as e:
            print(f"\nError on {date_str}: {e}")
            import traceback
            traceback.print_exc()
            error_count += 1
            continue

    print("\n" + "=" * 70)
    print("Forecast generation complete")
    print("=" * 70)
    print(f"Success: {success_count} days")
    print(f"Skipped: {skip_count} days (exists or invalid)")
    print(f"Errors: {error_count} days")
    print(f"Total: {len(file_dates)} days")
    print("=" * 70)

    if success_count > 0:
        print(f"\nForecasts saved to: {cfg.OUTPUT_ROOT}")
        print(f"Format: YYYY/MM/DD.h5")
        print(f"Shape: (48, 20, 440, 408)  # t=1..48 h, no IC")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nProgram failed: {e}")
        import traceback
        traceback.print_exc()
