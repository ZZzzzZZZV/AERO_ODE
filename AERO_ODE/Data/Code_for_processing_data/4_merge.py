import os
import argparse
import xarray as xr
import numpy as np
from multiprocessing import Pool, cpu_count
import time
import warnings

# 忽略序列化警告
warnings.filterwarnings("ignore")

# ================= 配置区域 =================
YEAR = "2024"
INPUT_DIR = f"/nfs/samba/数据聚变/气象数据/ERA5_Global_37Level_01Degree/nc/{YEAR}"
OUTPUT_DIR = f"/nfs/samba/数据聚变/气象数据/ERA5_Global_37Level_01Degree/nc_processed_5/{YEAR}"
NUM_WORKERS = 2  
# ===========================================

# 变量重命名映射
UPPER_AIR_VAR_MAP = {
    'Z': 'geopotential', 'Q': 'specific_humidity', 'T': 'temperature',
    'U': 'u_component_of_wind', 'V': 'v_component_of_wind',
    'CIWC': 'specific_cloud_ice_water_content', 'CLWC': 'specific_cloud_liquid_water_content',
    'plev': 'isobaricInhPa', 'lon': 'longitude', 'lat': 'latitude'
}

SURFACE_VAR_MAP = {
    'CI': 'sea_ice_cover', 'SSTK': 'sea_surface_temperature',
    'U10M': '10m_u_component_of_wind', 'V10M': '10m_v_component_of_wind',
    'MSL': 'mean_sea_level_pressure', 'T2M': '2m_temperature',
    'lon': 'longitude', 'lat': 'latitude'
}

def standardize_and_sort_coords(ds):
    """
    标准化坐标:
    1. 经度转为 -180 ~ 180
    2. 经度排序 (从小到大)
    3. 纬度排序 (从小到大)
    """
    if 'longitude' in ds.coords:
        lon_name = 'longitude'
    elif 'lon' in ds.coords:
        lon_name = 'lon'
    else:
        return ds 
        
    ds.coords[lon_name] = (ds.coords[lon_name] + 180) % 360 - 180
    ds = ds.sortby(lon_name)
    
    lat_name = 'latitude' if 'latitude' in ds.coords else 'lat'
    if lat_name in ds.coords:
        ds = ds.sortby(lat_name)
        
    return ds

def process_single_task(args):
    dirpath, root_dir, output_dir = args
    sfc_path = os.path.join(dirpath, '_surface_data.nc')
    pl_path = os.path.join(dirpath, '_upper_air_data.nc')
    
    rel_path = os.path.relpath(dirpath, root_dir)
    target_dir = os.path.join(output_dir, rel_path)
    os.makedirs(target_dir, exist_ok=True)
    
    try:
        parts = rel_path.replace('\\', '/').split('/')
        if len(parts) >= 2:
            mm, dd = parts[0], parts[1]
            filename = f"{YEAR}{mm}{dd}.nc"
        else:
            filename = 'merged_data.nc'
    except:
        filename = 'merged_data.nc'

    out_path = os.path.join(target_dir, filename)

    try:
        with xr.open_dataset(sfc_path, chunks={}) as ds_sfc, \
             xr.open_dataset(pl_path, chunks={}) as ds_pl:
            
            # 1. 重命名
            rename_pl = {k: v for k, v in UPPER_AIR_VAR_MAP.items() if k in ds_pl.variables or k in ds_pl.dims}
            ds_pl = ds_pl.rename(rename_pl)
            
            rename_sfc = {k: v for k, v in SURFACE_VAR_MAP.items() if k in ds_sfc.variables or k in ds_sfc.dims}
            ds_sfc = ds_sfc.rename(rename_sfc)
            
            # 2. 坐标标准化
            ds_pl = standardize_and_sort_coords(ds_pl)
            ds_sfc = standardize_and_sort_coords(ds_sfc)
            
            # 3. 高空处理
            if 'isobaricInhPa' in ds_pl.coords:
                ds_pl['isobaricInhPa'] = ds_pl['isobaricInhPa'] / 100.0
                ds_pl['isobaricInhPa'].attrs['units'] = 'hPa'
                ds_pl['isobaricInhPa'].attrs['long_name'] = 'pressure'
                ds_pl = ds_pl.sortby('isobaricInhPa', ascending=False)
            
            # 4. 合并
            ds_merged = xr.merge([ds_pl, ds_sfc], compat='override')
            
            # 5. 维度重排序 (关键步骤)
            # 强制按照 (time, isobaricInhPa, latitude, longitude) 顺序排列
            # 对于地表变量 (没有 isobaricInhPa)，xarray 会自动忽略该维度，只排剩下的
            # ... (Ellipsis) 代表“其他未列出的维度”
            ds_merged = ds_merged.transpose('time', 'isobaricInhPa', 'latitude', 'longitude', ...)
            
            # 6. 保存
            ds_merged.to_netcdf(out_path)
            
            return f"完成: {rel_path} -> {filename}"

    except Exception as e:
        import traceback
        return f"ERROR ({dirpath}): {e}"

def get_tasks(root_dir, output_dir):
    tasks = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames.sort()
        if '_surface_data.nc' in filenames and '_upper_air_data.nc' in filenames:
            tasks.append((dirpath, root_dir, output_dir))
    return tasks

if __name__ == "__main__":
    start_time = time.time()
    
    print(f"扫描任务: {INPUT_DIR}")
    tasks = get_tasks(INPUT_DIR, OUTPUT_DIR)
    
    if not tasks:
        print("未找到任务。")
        exit()
        
    print(f"开始处理 {len(tasks)} 个任务 (Final-Transposed, {NUM_WORKERS} 进程)...")
    
    with Pool(processes=NUM_WORKERS) as pool:
        for result in pool.imap_unordered(process_single_task, tasks, chunksize=1):
            print(result)

    end_time = time.time()
    print(f"耗时: {end_time - start_time:.2f} 秒")