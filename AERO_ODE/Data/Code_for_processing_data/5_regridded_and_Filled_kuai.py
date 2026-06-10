
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
import multiprocessing
from functools import partial

# --- Configuration ---
# 请将此路径修改为您下载的NeuralGCM模型checkpoint (.pkl) 的实际路径
MODEL_PATH = '/home/zhangjing/Code/NeuralGCM/pkl/neuralgcm_04_30_2024_neural_gcm_dynamic_forcing_deterministic_1_4_deg.pkl' 

# 输入和输出目录
INPUT_ROOT = '/nfs/samba/数据聚变/气象数据/ERA5_Global_37Level_01Degree/nc_merge'
OUTPUT_ROOT = '/nfs/samba/数据聚变/气象数据/ERA5_Global_37Level_01Degree/Regridded_and_Filled'

# 指定处理年份 (例如 '2021')。如果设为 None，则处理 INPUT_ROOT 下所有年份。
TARGET_YEAR = '2024'

# 并行进程数。根据您的内存和CPU核数调整。
# 注意：0.1度数据内存占用较大，建议先设小一点尝试，避免内存溢出。
NUM_WORKERS = 2 

# 全局变量，用于在子进程中缓存 regridder
global_regridder = None

def load_model(model_path):
    """Load the NeuralGCM model to get the target grid configuration."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}. Please update MODEL_PATH in the script.")
    
    print(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        ckpt = pickle.load(f)
    
    model = neuralgcm.PressureLevelModel.from_checkpoint(ckpt)
    return model

def init_worker(sample_input_path, model_path):
    """
    Worker initializer: Creates the regridder once per process.
    Loads the model inside the worker to get target coords, avoiding pickling issues.
    """
    global global_regridder
    # print(f"Worker initializing regridder with {sample_input_path}...")
    
    try:
        # Load model locally in worker to get coords
        # This is slow (happens once per worker) but safe for multiprocessing
        with open(model_path, 'rb') as f:
            ckpt = pickle.load(f)
        # Assuming we just need the coords, we might not need to fully instantiate if we can extract from ckpt
        # But instantiating is safer to match logic
        model = neuralgcm.PressureLevelModel.from_checkpoint(ckpt)
        model_horizontal_coords = model.data_coords.horizontal
        
        ds = xarray.open_dataset(sample_input_path)
        
        # Ensure latitude and longitude exist
        if 'latitude' not in ds.dims or 'longitude' not in ds.dims:
            if 'lat' in ds.dims: ds = ds.rename({'lat': 'latitude'})
            if 'lon' in ds.dims: ds = ds.rename({'lon': 'longitude'})
            
        source_grid = spherical_harmonic.Grid(
            latitude_nodes=ds.sizes['latitude'],
            longitude_nodes=ds.sizes['longitude'],
            latitude_spacing=xarray_utils.infer_latitude_spacing(ds.latitude),
            longitude_offset=xarray_utils.infer_longitude_offset(ds.longitude),
        )
        
        global_regridder = horizontal_interpolation.ConservativeRegridder(
            source_grid, model_horizontal_coords, skipna=True
        )
        ds.close()
    except Exception as e:
        print(f"Error in worker initialization: {e}")
        raise e

def process_file_wrapper(file_info):
    """
    Wrapper function for multiprocessing.
    file_info is a tuple: (input_file, output_file)
    """
    input_file, output_file = file_info
    global global_regridder
    
    if global_regridder is None:
        raise RuntimeError("Regridder not initialized in worker process")
        
    if os.path.exists(output_file):
        print(f"Skipping existing: {output_file}")
        return

    print(f"Processing: {input_file}")
    
    try:
        # 1. Load dataset
        ds = xarray.open_dataset(input_file)
        
        # Standardize dimensions
        if 'latitude' not in ds.dims or 'longitude' not in ds.dims:
            if 'lat' in ds.dims: ds = ds.rename({'lat': 'latitude'})
            if 'lon' in ds.dims: ds = ds.rename({'lon': 'longitude'})
        
        # 2. Regrid (using global pre-computed regridder)
        try:
            regridded_ds = xarray_utils.regrid(ds, global_regridder)
        except ValueError as e:
            print(f"Warning: Grid mismatch potential in {input_file}. Error: {e}")
            raise e
        
        # 3. Fill NaNs
        filled_ds = xarray_utils.fill_nan_with_nearest(regridded_ds)
        
        # 4. Save output
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        # Using complevel=1 for faster writing (balance between size and speed)
        encoding = {var: {'zlib': True, 'complevel': 1} for var in filled_ds.data_vars}
        filled_ds.to_netcdf(output_file, encoding=encoding)
        
        ds.close()
        regridded_ds.close()
        filled_ds.close()
        
    except Exception as e:
        print(f"Failed to process {input_file}: {e}")

def main():
    # 1. Load Model Metadata (Main Process)
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
    print(f"Starting processing with {NUM_WORKERS} workers...")
    
    # 3. Process files in parallel
    # We use the first file to initialize the regridder structure in all workers
    first_input_file = files_to_process[0][0]
    
    # Use 'spawn' context if possible to avoid JAX/threading issues, 
    # but 'fork' (default on Linux) is usually faster for simple copy-on-write.
    # If you encounter JAX errors, uncomment the next line:
    # multiprocessing.set_start_method('spawn', force=True)
    
    # Extract only the necessary primitive parameters to reconstruct the grid/coords
    # This avoids pickling complex JAX/Dinosaur objects which fails with 'spawn'
    horizontal_coords_args = {
        'longitudes': target_horizontal_coords.longitudes,
        'latitudes': target_horizontal_coords.latitudes,
        # Add other necessary parameters if needed, or serialize differently
        # However, since pickling the Grid object itself is failing, we might need a different approach.
        # Strategy: Pass the MODEL_PATH to workers and let them load the model themselves.
    }
    
    with multiprocessing.Pool(
        processes=NUM_WORKERS, 
        initializer=init_worker, 
        initargs=(first_input_file, MODEL_PATH)  # Pass MODEL_PATH instead of coords object
    ) as pool:
        pool.map(process_file_wrapper, files_to_process)

if __name__ == "__main__":
    # Ensure spawn method is used
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()

# if __name__ == "__main__":
#     main()




