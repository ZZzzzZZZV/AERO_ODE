"""
    python Generate_Yearly_Forecast_Surface.py
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

# Project imports
from film_inference_surface import DiagnosisModel, StaticDataManager, compute_time_indices, compute_solar_features, TARGET_LEVELS
import AERO_v3 as NGCM


# ============================================
# Config
# ============================================
class Config:
    """Forecast generation config"""
    
    # Forecast config
    TIME_STEPS = 73  # includes IC; saved output drops t=0
    
    # Model paths
    NGCM_CHECKPOINT = os.fspath(pc.ngcm_checkpoint(PROJECT_ROOT))
    # Surface variable diagnosis model checkpoint
    DIAGNOSIS_CKPT = os.fspath(pc.film_checkpoint_dir(PROJECT_ROOT, 'checkpoints_film_v2') / 'model_ep5_batch4499.pth')
    
    # Data paths
    HRRR_STAT_PATH = os.fspath(pc.hrrr_stat_root())
    STATIC_DATA_PATH = os.fspath(pc.surface_static_data(PROJECT_ROOT))  # Static feature data path
    ERA5_ROOT = os.fspath(pc.era5_root())
    
    # Output paths
    OUTPUT_ROOT = os.fspath(pc.predicted_output_root(PROJECT_ROOT))
    
    # Inference config
    TIME_CHUNK = 8  # chunk time steps to save VRAM
    DEVICE = 'cuda:0'
    
    # Skip existing output files
    SKIP_EXISTING = True


def find_era5_00z_files(era5_root: str) -> List[Tuple[str, datetime]]:
    """
Scan ERA5 tree for daily nc files
    
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
            
            # Parse date from YYYYMMDD.nc
            date_str = fname.replace('.nc', '')
            if len(date_str) != 8 or not date_str.isdigit():
                continue
            
            try:
                dt = datetime.strptime(date_str, "%Y%m%d")
                fpath = os.path.join(root, fname)
                file_dates.append((fpath, dt))
            except ValueError:
                continue
    
    # sorted by date
    file_dates.sort(key=lambda x: x[1])
    
    return file_dates


def load_era5_00z_initial_state(era5_file: str) -> Tuple[Optional[xr.Dataset], Optional[str], bool]:
    """
    Load ERA5 00Z initial condition
    
    Args:
        era5_file: ERA5 nc path
    
    Returns:
        ds0: xarray.Dataset at time=0 (time dim length=1)
        time_str: 'YYYY/MM/DD/HH'
        valid: bool, whether IC is valid
    """
    ds = None
    try:
        ds = xr.open_dataset(era5_file, decode_timedelta=False)
        
        # Check time dimension
        times = ds['time'].values
        if len(times) == 0:
            ds.close()
            return None, None, False
        
        # First timestamp (expect 00Z)
        t0 = np.datetime64(times[0], 's')
        dt = datetime.utcfromtimestamp(int(t0.astype('int64')))
        
        # Require 00Z
        if dt.hour != 0:
            print(f"Warning: {era5_file} first time is not 00Z (hour={dt.hour}), skip")
            ds.close()
            return None, None, False
        
        # time=0 only (time dim length=1)
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
    """
    Yearly surface forecast generator
    
    NeuralGCM + DiagnosisModel inference pipeline
    """
    
    def __init__(self, config: Config):
        self.cfg = config
        self.device = torch.device(config.DEVICE)
        
        # 1. Load static data manager
        print(">>> Loading Static Data...")
        self.static_manager = StaticDataManager(
            config.HRRR_STAT_PATH, 
            device=self.device,
            target_levels=TARGET_LEVELS,
            static_data_path=config.STATIC_DATA_PATH
        )
        
        # 2. Load regional grid
        self.region_lat = np.load(os.path.join(config.HRRR_STAT_PATH, 'lats.npy')).T
        self.region_lon = np.load(os.path.join(config.HRRR_STAT_PATH, 'lons.npy')).T
        
        # 3. Load NeuralGCM
        print(">>> Loading NeuralGCM...")
        self.ngcm = NGCM.NeuralGCMInference(
            config.NGCM_CHECKPOINT, 
            inner_steps=1, 
            outer_steps=config.TIME_STEPS
        )
        
        # 4. Load surface diagnosis model
        print(">>> Loading Surface Diagnosis Model...")
        self.diagnosis_model = DiagnosisModel(
            in_channels_dynamic=74,  # 35 physics + 35 correction + 4 solar
            in_channels_static=14,   # terrain + position + derived + land cover
            out_channels=4,          # mslp, u10, v10, t2m
            time_emb_dim=128,
            backbone='unet3plus'
        ).to(self.device)
        
        checkpoint = torch.load(config.DIAGNOSIS_CKPT, map_location=self.device)
        state_dict = checkpoint['model_state_dict']
        
        # Strip 'module.' prefix from DDP checkpoints
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k.replace('module.', '')] = v
        self.diagnosis_model.load_state_dict(new_state_dict)
        self.diagnosis_model.eval()
        
        # 5. Precompute normalization for NeuralGCM physics core
        self.input_mean = self.static_manager.input_mean
        self.input_std = self.static_manager.input_std
        
        print(">>> Model initialization complete!")
    
    def inference(self, input_list: List[xr.Dataset], time_list: List[str]) -> torch.Tensor:
        """
        Run NeuralGCM -> DiagnosisModel -> denormalize
        
        Args:
            input_list: [ds0] ERA5initial conditions
            time_list: ['YYYY/MM/DD/HH'] init time strings
        
        Returns:
            surface_pred: (1, T, 4, 440, 408) surface forecast
        """
        B = len(input_list)
        time_chunk = self.cfg.TIME_CHUNK
        
        with torch.inference_mode():
            # NeuralGCM inference (5 near-surface levels)
            t_pred, t_phy = self.ngcm.forward(
                input_list,
                target_levels=TARGET_LEVELS,  # [1000, 925, 850, 700, 500]
                include_era5_label=False,
                region_lon=self.region_lon,
                region_lat=self.region_lat,
            )
            
            # Geopotential height / 9.8 (first 5 channels are 5-level geopotential height)
            t_pred[:, :5] /= 9.8
            t_phy[:, :5] /= 9.8
            
            # Permute (B, C, W, H, T) -> (B, T, C, W, H)
            t_pred = t_pred.permute(0, 4, 1, 2, 3).to(self.device)
            t_phy = t_phy.permute(0, 4, 1, 2, 3).to(self.device)
            
            B, T, C, W, H = t_phy.shape  # C=35 (7 vars x 5 levels)
            
            # Input normalization
            t_phy_norm = (t_phy - self.input_mean) / (self.input_std + 1e-9)
            corr_norm = (t_pred - t_phy) / (self.input_std + 1e-9)
            
            # Solar features (4 channels)
            solar_feat = compute_solar_features(time_list, T, self.region_lat, self.region_lon, self.device)
            
            # Dynamic input: 35 physics + 35 correction + 4 solar = 74 channels
            dyn_in = torch.cat([t_phy_norm, corr_norm, solar_feat], dim=2)
            
            static_in = self.static_manager.get_static_input(B)
            steps_all, hours_all, months_all = compute_time_indices(time_list, T, self.device)
            
            # Chunked diagnosis (save VRAM)
            surface_chunks = []
            for t0 in range(0, T, time_chunk):
                t1 = min(t0 + time_chunk, T)
                tc = t1 - t0
                
                dyn = dyn_in[:, t0:t1].reshape(B * tc, 74, W, H)
                n_static = static_in.shape[1]  # 14
                sta = static_in.unsqueeze(1).expand(-1, tc, -1, -1, -1).reshape(B * tc, n_static, W, H)
                s = steps_all.view(B, T)[:, t0:t1].reshape(-1)
                h = hours_all.view(B, T)[:, t0:t1].reshape(-1)
                m = months_all.view(B, T)[:, t0:t1].reshape(-1)
                
                # Diagnosis output (normalized)
                output_norm = self.diagnosis_model(dyn, sta, s, h, m)
                
                # Denormalize
                pred = output_norm * (self.static_manager.surface_std + 1e-9) + self.static_manager.surface_mean
                surface_chunks.append(pred.reshape(B, tc, 4, W, H))
            
            surface_pred = torch.cat(surface_chunks, dim=1)
            
            # Permute (B, T, C, W, H) -> (B, T, C, H, W)
            # surface_pred shape (1, 49, 4, 440, 408) before dropping IC
            surface_pred = surface_pred.permute(0, 1, 2, 4, 3)
            
            return surface_pred
    
    def save_forecast(self, forecast_tensor: torch.Tensor, date: datetime, output_root: str) -> str:
        """
        Save h5 forecast (drop IC; keep t=1..48)
        
        Args:
            forecast_tensor: (1, T, 4, 440, 408), T includes IC
            date: datetime init date
            output_root: output root
        
        Returns:
            output_path: saved file path
        """
        # Output path YYYY/MM/DD_surface.h5
        year = date.strftime('%Y')
        month = date.strftime('%m')
        day = date.strftime('%d')
        
        output_dir = os.path.join(output_root, year, month)
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f'{day}_surface.h5')
        
        # To numpy, drop batch dim and IC (step 0)
        # forecast_tensor: (1, 49, 4, 440, 408) -> data: (48, 4, 440, 408)
        data = forecast_tensor[0, 1:, :, :, :].cpu().numpy()  # drop t=0 IC
        
        # Save h5
        with h5py.File(output_path, 'w') as f:
            f.create_dataset(
                'fields', 
                data=data, 
                compression='gzip', 
                compression_opts=4
            )
            # Add metadata
            f.attrs['init_time'] = date.strftime('%Y-%m-%d %H:%M:%S UTC')
            f.attrs['forecast_hours'] = 48  # saved leads excluding IC
            f.attrs['variables'] = ['MSLP', 'U10', 'V10', 'T2M']
            f.attrs['units'] = ['Pa', 'm/s', 'm/s', 'K']
            f.attrs['shape'] = '(time_steps, variables, latitude, longitude)'
            f.attrs['note'] = 'time_steps=1~48h forecast (initial state t=0 excluded)'
        
        return output_path


def main():
    """Main entry"""
    cfg = Config()
    
    print("=" * 70)
    print("AERO-Surface yearly surface forecast generator")
    print("=" * 70)
    print(f"Inference steps: {cfg.TIME_STEPS} (00Z + 72 h)")
    print(f"Saved steps: 72 (drop IC t=0)")
    print(f"Output shape: (72, 4, 440, 408)")
    print(f"Variables: MSLP, U10, V10, T2M")
    print(f"Input: {cfg.ERA5_ROOT}")
    print(f"Output: {cfg.OUTPUT_ROOT}")
    print(f"Filename pattern: DD_surface.h5")
    print(f"Skip existing: {cfg.SKIP_EXISTING}")
    print("=" * 70)
    
    # 1. Scan ERA5 files
    print("\n>>> Scanning ERA5 initial conditions...")
    file_dates = find_era5_00z_files(cfg.ERA5_ROOT)
    print(f"Found {len(file_dates)} ERA5 files")
    
    if len(file_dates) == 0:
        print("Error: no ERA5 files found")
        print(f"Check path: {cfg.ERA5_ROOT}")
        return
    
    # Date range
    first_date = file_dates[0][1]
    last_date = file_dates[-1][1]
    print(f"Date range: {first_date.strftime('%Y-%m-%d')} ~ {last_date.strftime('%Y-%m-%d')}")
    
    # 2. Initialize forecaster
    print("\n>>> Initialize forecast model...")
    forecaster = SurfaceForecaster(cfg)
    
    # 3. Statistics
    success_count = 0
    skip_count = 0
    error_count = 0
    
    # 4. Loop dates and forecast
    print("\n>>> Generating surface forecasts...")
    pbar = tqdm(file_dates, desc="Forecasting", unit="day")
    
    for era5_file, date in pbar:
        date_str = date.strftime('%Y-%m-%d')
        pbar.set_postfix_str(f"Processing: {date_str}")
        
        try:
            # Skip if output exists
            year = date.strftime('%Y')
            month = date.strftime('%m')
            day = date.strftime('%d')
            output_path = os.path.join(cfg.OUTPUT_ROOT, year, month, f'{day}_surface.h5')
            
            if cfg.SKIP_EXISTING and os.path.exists(output_path):
                skip_count += 1
                continue
            
            # Load 00Z IC
            ds0, time_str, valid = load_era5_00z_initial_state(era5_file)
            
            if not valid:
                print(f"\nSkip {date_str}: invalid IC")
                skip_count += 1
                continue
            
            # Run forecast
            input_list = [ds0]
            time_list = [time_str]
            
            forecast = forecaster.inference(input_list, time_list)
            
            # Check NaN
            if torch.isnan(forecast).any():
                print(f"\nWarning: {date_str} forecast has NaN, skip")
                ds0.close()
                error_count += 1
                continue
            
            # Validate output shape (includes IC)
            expected_shape = (1, cfg.TIME_STEPS, 4, 440, 408)
            if forecast.shape != expected_shape:
                print(f"\nWarning: {date_str} shape {forecast.shape} != expected {expected_shape}")
            
            # Save results
            saved_path = forecaster.save_forecast(forecast, date, cfg.OUTPUT_ROOT)
            success_count += 1
            
            # Close dataset
            ds0.close()
            
            # Periodic memory cleanup
            if success_count % 10 == 0:
                torch.cuda.empty_cache()
            
            # Periodic progress
            if success_count % 30 == 0:
                pbar.write(f"Progress: ok {success_count}, skip {skip_count}, err {error_count}")
                
        except Exception as e:
            print(f"\nError on {date_str}: {e}")
            import traceback
            traceback.print_exc()
            error_count += 1
            continue
    
    # 5. Final stats
    print("\n" + "=" * 70)
    print("surface variableForecast generation complete")
    print("=" * 70)
    print(f"Success: {success_count} days")
    print(f"Skipped: {skip_count} days (exists or invalid)")
    print(f"Errors: {error_count} days")
    print(f"Total: {len(file_dates)} days")
    print("=" * 70)
    
    if success_count > 0:
        print(f"\nForecasts saved to: {cfg.OUTPUT_ROOT}")
        print(f"Format: YYYY/MM/DD_surface.h5")
        print(f"Shape: (48, 4, 440, 408)  # t=1..48 h forecast (no IC)")
        print(f"Variable order: MSLP (Pa), U10 (m/s), V10 (m/s), T2M (K)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nProgram failed: {e}")
        import traceback
        traceback.print_exc()
