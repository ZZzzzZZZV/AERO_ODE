
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device_ids = [0]

import glob
import xarray
import pickle
import numpy as np
import neuralgcm
from dinosaur import horizontal_interpolation
from dinosaur import spherical_harmonic
from dinosaur import xarray_utils

# --- Configuration ---
# 请将此路径修改为您下载的NeuralGCM模型checkpoint (.pkl) 的实际路径
# 例如: 'neuralgcm_dynamic_forcing_deterministic_0_7_deg.pkl'
# 如果不确定模型路径，请确保该文件存在并修改此处
MODEL_PATH = '/home/zhangjing/Code/NeuralGCM/pkl/neuralgcm_04_30_2024_neural_gcm_dynamic_forcing_deterministic_1_4_deg.pkl' 

# 输入和输出目录
INPUT_ROOT = '/nfs/samba/数据聚变/气象数据/ERA5_Global_37Level_01Degree/nc_merge'
OUTPUT_ROOT = '/nfs/samba/数据聚变/气象数据/ERA5_Global_37Level_01Degree/Regridded_and_Filled'

# 指定处理年份 (例如 '2021')。如果设为 None，则处理 INPUT_ROOT 下所有年份。
TARGET_YEAR = '2021'

def load_model(model_path):
    """Load the NeuralGCM model to get the target grid configuration."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}. Please update MODEL_PATH in the script.")
    
    print(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        ckpt = pickle.load(f)
    
    # Instantiate model to get grid info. Using PressureLevelModel as generic entry point.
    # Note: The specific class might depend on the checkpoint, but from_checkpoint usually handles it.
    model = neuralgcm.PressureLevelModel.from_checkpoint(ckpt)
    return model

def create_regridder(sample_input_path, model_horizontal_coords):
    """
    Creates a regridder instance from a sample input file.
    This allows us to compute the regridding weights once and reuse them.
    """
    print(f"Initializing regridder using sample file: {sample_input_path}")
    ds = xarray.open_dataset(sample_input_path)
    
    # Ensure latitude and longitude exist for the source grid definition
    if 'latitude' not in ds.dims or 'longitude' not in ds.dims:
        if 'lat' in ds.dims: ds = ds.rename({'lat': 'latitude'})
        if 'lon' in ds.dims: ds = ds.rename({'lon': 'longitude'})
        
    source_grid = spherical_harmonic.Grid(
        latitude_nodes=ds.sizes['latitude'],
        longitude_nodes=ds.sizes['longitude'],
        latitude_spacing=xarray_utils.infer_latitude_spacing(ds.latitude),
        longitude_offset=xarray_utils.infer_longitude_offset(ds.longitude),
    )
    
    regridder = horizontal_interpolation.ConservativeRegridder(
        source_grid, model_horizontal_coords, skipna=True
    )
    ds.close()
    return regridder

def process_file(input_file, output_file, regridder):
    """
    Regrids and fills a single netCDF file using a pre-computed regridder.
    """
    print(f"Processing: {input_file}")
    
    # 1. Load dataset
    ds = xarray.open_dataset(input_file)
    
    # Standardize dimensions if needed (must match the regridder's source grid assumptions)
    if 'latitude' not in ds.dims or 'longitude' not in ds.dims:
        if 'lat' in ds.dims: ds = ds.rename({'lat': 'latitude'})
        if 'lon' in ds.dims: ds = ds.rename({'lon': 'longitude'})
    
    # 2. Regrid (using pre-computed regridder)
    # xarray_utils.regrid is an alias for regrid_horizontal
    try:
        regridded_ds = xarray_utils.regrid(ds, regridder)
    except ValueError as e:
        print(f"Warning: Grid mismatch potential in {input_file}. Error: {e}")
        # If grid is slightly different (e.g. numerical precision), one might need to rebuild regridder
        # But for split ERA5 files, they should be identical.
        raise e
    
    # 3. Fill NaNs
    filled_ds = xarray_utils.fill_nan_with_nearest(regridded_ds)
    
    # 4. Save output
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    encoding = {var: {'zlib': True, 'complevel': 5} for var in filled_ds.data_vars}
    filled_ds.to_netcdf(output_file, encoding=encoding)
    
    ds.close()
    regridded_ds.close()
    filled_ds.close()

def main():
    # 1. Load Model Metadata
    try:
        model = load_model(MODEL_PATH)
        target_horizontal_coords = model.data_coords.horizontal
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please verify the MODEL_PATH and your NeuralGCM installation.")
        return

    # 2. Walk through input directory
    search_path = INPUT_ROOT
    if TARGET_YEAR:
        search_path = os.path.join(INPUT_ROOT, str(TARGET_YEAR))
        if not os.path.exists(search_path):
            print(f"Error: Year directory not found: {search_path}")
            return
            
    print(f"Scanning directory: {search_path}")
    files_to_process = []
    
    for root, dirs, files in os.walk(search_path):
        for file in files:
            if file.endswith('.nc'):
                input_path = os.path.join(root, file)
                rel_path = os.path.relpath(input_path, INPUT_ROOT)
                output_path = os.path.join(OUTPUT_ROOT, rel_path)
                files_to_process.append((input_path, output_path))
    
    # Sort files by input path to ensure processing order
    files_to_process.sort(key=lambda x: x[0])
    
    if not files_to_process:
        print("No .nc files found.")
        return

    print(f"Found {len(files_to_process)} files to process.")
    
    # 3. Initialize Regridder (Once)
    # We assume all input files share the same spatial grid.
    first_input_file = files_to_process[0][0]
    try:
        regridder = create_regridder(first_input_file, target_horizontal_coords)
    except Exception as e:
        print(f"Failed to create regridder from sample file {first_input_file}: {e}")
        return
    
    # 4. Process files
    for input_path, output_path in files_to_process:
        if os.path.exists(output_path):
            print(f"Skipping existing: {output_path}")
            continue
            
        try:
            process_file(input_path, output_path, regridder)
        except Exception as e:
            print(f"Failed to process {input_path}: {e}")


if __name__ == "__main__":
    main()

