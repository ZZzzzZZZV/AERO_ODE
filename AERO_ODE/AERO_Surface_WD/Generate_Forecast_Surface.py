import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(next(_p for _p in _pathlib.Path(__file__).resolve().parents
                             if (_p / "common" / "paths.py").is_file())))
from common.paths import ROOT, iter_era5_daily_files

import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")
os.environ.setdefault("LOCAL_RANK", "0")

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".30"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import h5py
import xarray as xr
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
from tqdm import tqdm

from inference_components import (
    DiagnosisModel,
    StaticDataManager,
    TARGET_LEVELS,
    compute_solar_features,
    compute_time_indices,
)
import AERO_ODE_v3 as AERO

class Config:

    TIME_STEPS = 73

    NGCM_CHECKPOINT = str(
        ROOT / "shared_assets/NeuralGCM_Weights"
        / "neuralgcm_04_30_2024_neural_gcm_dynamic_forcing_deterministic_1_4_deg.pkl"
    )
    DIAGNOSIS_CKPT = str(
        ROOT / "AERO_Surface_WD/checkpoints_film_v2/model_ep5.pth"
    )

    HRRR_STAT_PATH = str(ROOT / "AERO_Surface_WD/Hrrr_rb/data")
    STATIC_DATA_PATH = str(ROOT / "AERO_Surface_WD/Hrrr_rb/data")
    ERA5_ROOT = str(ROOT / "data/era5_test")
    OUTPUT_ROOT = str(ROOT / "data/forecast_output")

    TIME_CHUNK = 8
    DEVICE = 'cuda:0'

    SKIP_EXISTING = True

def find_era5_00z_files(era5_root: str) -> List[Tuple[str, datetime]]:
    file_dates = [(os.fspath(path), dt) for path, dt in iter_era5_daily_files(era5_root)]
    file_dates.sort(key=lambda x: x[1])
    return file_dates

def load_era5_00z_initial_state(era5_file: str) -> Tuple[Optional[xr.Dataset], Optional[str], bool]:
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
            print(f"Warning: {era5_file} first time is not 00 UTC (hour={dt.hour}), skipped")
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

class SurfaceForecaster:

    def __init__(self, config: Config, ngcm=None):
        self.cfg = config
        self.device = torch.device(config.DEVICE)
        self.target_levels = TARGET_LEVELS

        self.static_manager = StaticDataManager(
            config.HRRR_STAT_PATH,
            device=self.device,
            target_levels=TARGET_LEVELS,
            static_data_path=config.STATIC_DATA_PATH
        )

        self.region_lat = np.load(os.path.join(config.HRRR_STAT_PATH, 'hrrr_west_lat.npy')).T
        self.region_lon = np.load(os.path.join(config.HRRR_STAT_PATH, 'hrrr_west_lon.npy')).T

        if ngcm is None:
            self.ngcm = AERO.NeuralGCMInference(
                config.NGCM_CHECKPOINT,
                inner_steps=1,
                outer_steps=config.TIME_STEPS
            )
        else:
            self.ngcm = ngcm

        self.diagnosis_model = DiagnosisModel(
            in_channels_dynamic=74,
            in_channels_static=14,
            out_channels=4,
            time_emb_dim=128,
            backbone='unet3plus'
        ).to(self.device)

        try:
            checkpoint = torch.load(
                config.DIAGNOSIS_CKPT, map_location=self.device, weights_only=True
            )
        except Exception:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                checkpoint = torch.load(config.DIAGNOSIS_CKPT, map_location=self.device)
        state_dict = checkpoint['model_state_dict']

        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k.replace('module.', '')] = v
        self.diagnosis_model.load_state_dict(new_state_dict)
        self.diagnosis_model.eval()

        self.input_mean = self.static_manager.input_mean
        self.input_std = self.static_manager.input_std

    def inference(self, input_list: List[xr.Dataset], time_list: List[str]) -> torch.Tensor:
        B = len(input_list)
        time_chunk = self.cfg.TIME_CHUNK

        with torch.inference_mode():

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

            B, T, C, W, H = t_phy.shape

            t_phy_norm = (t_phy - self.input_mean) / (self.input_std + 1e-9)
            corr_norm = (t_pred - t_phy) / (self.input_std + 1e-9)

            solar_feat = compute_solar_features(time_list, T, self.region_lat, self.region_lon, self.device)

            dyn_in = torch.cat([t_phy_norm, corr_norm, solar_feat], dim=2)

            static_in = self.static_manager.get_static_input(B)
            steps_all, hours_all, months_all = compute_time_indices(time_list, T, self.device)

            surface_chunks = []
            for t0 in range(0, T, time_chunk):
                t1 = min(t0 + time_chunk, T)
                tc = t1 - t0

                dyn = dyn_in[:, t0:t1].reshape(B * tc, 74, W, H)
                n_static = static_in.shape[1]
                sta = static_in.unsqueeze(1).expand(-1, tc, -1, -1, -1).reshape(B * tc, n_static, W, H)
                s = steps_all.view(B, T)[:, t0:t1].reshape(-1)
                h = hours_all.view(B, T)[:, t0:t1].reshape(-1)
                m = months_all.view(B, T)[:, t0:t1].reshape(-1)

                output_norm = self.diagnosis_model(dyn, sta, s, h, m)

                pred = output_norm * (self.static_manager.surface_std + 1e-9) + self.static_manager.surface_mean
                surface_chunks.append(pred.reshape(B, tc, 4, W, H))

            surface_pred = torch.cat(surface_chunks, dim=1)

            surface_pred = surface_pred.permute(0, 1, 2, 4, 3)

            return surface_pred

    def save_forecast(self, forecast_tensor: torch.Tensor, date: datetime, output_root: str) -> str:

        year = date.strftime('%Y')
        month = date.strftime('%m')
        day = date.strftime('%d')

        output_dir = os.path.join(output_root, year, month)
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, f'{day}_surface.h5')

        data = forecast_tensor[0].cpu().numpy()

        with h5py.File(output_path, 'w') as f:
            f.create_dataset(
                'fields',
                data=data,
                compression='gzip',
                compression_opts=4
            )

            f.attrs['init_time'] = date.strftime('%Y-%m-%d %H:%M:%S UTC')
            f.attrs['time_steps'] = int(data.shape[0])
            f.attrs['forecast_hours'] = 72
            f.attrs['variables'] = ['MSLP', 'U10', 'V10', 'T2M']
            f.attrs['units'] = ['Pa', 'm/s', 'm/s', 'K']
            f.attrs['shape'] = '(time_steps, variables, latitude, longitude)'
            f.attrs['note'] = '73 steps: t=0 analysis + hourly +1...+72 h'

        return output_path

def main():
    cfg = Config()

    print("=" * 70)
    print("AERO-ODE surface year-long forecast")
    print("=" * 70)
    print(f"inference steps: {cfg.TIME_STEPS} (00 UTC + 72 h)")
    print(f"saved steps: {cfg.TIME_STEPS} (includes t=0)")
    print(f"output shape: ({cfg.TIME_STEPS}, 4, 440, 408)")
    print(f"output variables: MSLP, U10, V10, T2M")
    print(f"input path: {cfg.ERA5_ROOT}")
    print(f"output path: {cfg.OUTPUT_ROOT}")
    print(f"filename pattern: DD_surface.h5")
    print(f"skip existing: {cfg.SKIP_EXISTING}")
    print("=" * 70)

    print("\n>>> Scanning ERA5 initial-condition files...")
    file_dates = find_era5_00z_files(cfg.ERA5_ROOT)
    print(f"found {len(file_dates)} ERA5 files")

    if len(file_dates) == 0:
        print("Error: no ERA5 files found.")
        print(f"check path: {cfg.ERA5_ROOT}")
        return

    first_date = file_dates[0][1]
    last_date = file_dates[-1][1]
    print(f"date range: {first_date.strftime('%Y-%m-%d')} ~ {last_date.strftime('%Y-%m-%d')}")

    print("\n>>> Initializing forecast model...")
    forecaster = SurfaceForecaster(cfg)

    success_count = 0
    skip_count = 0
    error_count = 0

    print("\n>>> Generating surface forecasts...")
    pbar = tqdm(file_dates, desc="Forecasting", unit="day")

    for era5_file, date in pbar:
        date_str = date.strftime('%Y-%m-%d')
        pbar.set_postfix_str(f"Processing: {date_str}")

        try:

            year = date.strftime('%Y')
            month = date.strftime('%m')
            day = date.strftime('%d')
            output_path = os.path.join(cfg.OUTPUT_ROOT, year, month, f'{day}_surface.h5')

            if cfg.SKIP_EXISTING and os.path.exists(output_path):
                skip_count += 1
                continue

            ds0, time_str, valid = load_era5_00z_initial_state(era5_file)

            if not valid:
                print(f"\nskip {date_str}: invalid initial condition")
                skip_count += 1
                continue

            input_list = [ds0]
            time_list = [time_str]

            forecast = forecaster.inference(input_list, time_list)

            if torch.isnan(forecast).any():
                print(f"\nwarning: {date_str} forecast contains NaN, skipped")
                ds0.close()
                error_count += 1
                continue

            expected_shape = (1, cfg.TIME_STEPS, 4, 440, 408)
            if forecast.shape != expected_shape:
                print(f"\nwarning: {date_str} forecast shape {forecast.shape} != expected {expected_shape}")

            saved_path = forecaster.save_forecast(forecast, date, cfg.OUTPUT_ROOT)
            success_count += 1

            ds0.close()

            if success_count % 10 == 0:
                torch.cuda.empty_cache()

            if success_count % 30 == 0:
                pbar.write(f"progress: ok {success_count}, skipped {skip_count}, errors {error_count}")

        except Exception as e:
            print(f"\nerror on {date_str}: {e}")
            import traceback
            traceback.print_exc()
            error_count += 1
            continue

    print("\n" + "=" * 70)
    print("Surface forecast generation finished.")
    print("=" * 70)
    print(f"ok: {success_count} days")
    print(f"skipped: {skip_count} days (exists or invalid)")
    print(f"errors: {error_count} days")
    print(f"total: {len(file_dates)} days")
    print("=" * 70)

    if success_count > 0:
        print(f"\nforecast files written to: {cfg.OUTPUT_ROOT}")
        print(f"file layout: YYYY/MM/DD_surface.h5")
        print(f"data shape: ({cfg.TIME_STEPS}, 4, 440, 408)  (t=0 + 72 h)")
        print(f"variable order: MSLP (Pa), U10 (m/s), V10 (m/s), T2M (K)")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nfailed: {e}")
        import traceback
        traceback.print_exc()
