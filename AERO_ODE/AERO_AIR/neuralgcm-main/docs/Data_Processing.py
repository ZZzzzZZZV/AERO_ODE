import os
import glob
import jax
import numpy as np
import xarray as xr
import pandas as pd
from dinosaur import horizontal_interpolation, spherical_harmonic, xarray_utils
import neuralgcm
import pickle
from tqdm import tqdm

# ================= 配置区域 =================

# 1. 原始数据根目录
RAW_DATA_ROOT = '/path/to/your/raw_data'  # 请修改这里
# 假设结构是: RAW_DATA_ROOT/YYYY/MM/filename.nc

# 2. 文件名模式 (支持 glob 通配符)
# 假设每个月文件夹下文件名是固定的，或者包含特定关键字
PRESSURE_FILE_PATTERN = '*pressure*.nc'  # 匹配高空文件
SURFACE_FILE_PATTERN = '*surface*.nc'    # 匹配地表文件

# 3. 输出目录
OUTPUT_DIR = '/path/to/your/clean_data_nc' # 请修改这里
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 4. 模型路径 (用于获取网格信息)
MODEL_PATH = '/nfs/gpu_homes/gpu09/home/zhangjing/Code/NeuralGCM/pkl/neuralgcm_04_30_2024_neural_gcm_dynamic_forcing_deterministic_1_4_deg.pkl'

# 5. 变量映射表
VAR_DICT = {
    'siconc': 'sea_ice_cover', 
    'sst': 'sea_surface_temperature',
    'u': 'u_component_of_wind', 
    'v': 'v_component_of_wind', 
    'z': 'geopotential', 
    't': 'temperature', 
    'q': 'specific_humidity', 
    'ciwc': 'specific_cloud_ice_water_content', 
    'clwc': 'specific_cloud_liquid_water_content',
}

# ================= 核心函数 =================

def get_regridder(sample_ds, model_coords):
    """构建 Regridder (惰性单例模式，只构建一次)"""
    print(f"构建 Regridder: 原分辨率 {sample_ds.sizes['latitude']}x{sample_ds.sizes['longitude']} -> 目标 1.4度")
    input_grid = spherical_harmonic.Grid(
        latitude_nodes=sample_ds.sizes['latitude'],
        longitude_nodes=sample_ds.sizes['longitude'],
        latitude_spacing=xarray_utils.infer_latitude_spacing(sample_ds.latitude),
        longitude_offset=xarray_utils.infer_longitude_offset(sample_ds.longitude),
    )
    regridder = horizontal_interpolation.ConservativeRegridder(
        input_grid, model_coords.horizontal, skipna=True
    )
    return regridder

def process_and_save_month(year, month, p_path, s_path, model, regridder):
    """处理单月数据的原子操作"""
    
    out_filename = os.path.join(OUTPUT_DIR, f"era5_clean_{year}_{month}.nc")
    
    # 断点续传：如果已存在且完整，跳过
    if os.path.exists(out_filename):
        print(f"  [Skip] {out_filename} 已存在")
        return

    print(f"  [Load] 正在读取 {year}-{month} ...")
    # chunks='auto' 启用 Dask，避免爆内存
    try:
        ds_p = xr.open_dataset(p_path, chunks={'time': 24})
        ds_s = xr.open_dataset(s_path, chunks={'time': 24})
    except Exception as e:
        print(f"  [Error] 读取文件失败: {e}")
        return

    # 1. 时间对齐 (取交集)
    common_time = np.intersect1d(ds_p.time, ds_s.time)
    if len(common_time) == 0:
        print(f"  [Error] {year}-{month} 高空与地面数据时间无交集！")
        return
        
    ds_p = ds_p.sel(time=common_time)
    ds_s = ds_s.sel(time=common_time)
    
    # 2. 合并
    ds_merged = xr.merge([ds_p, ds_s])
    
    # 3. 重命名变量
    ds_merged = ds_merged.rename_vars({k: v for k, v in VAR_DICT.items() if k in ds_merged})
    
    # 4. 变量检测，确保有全部所需变量， 不再筛选，保留所有变量
    # 但我们需要确保模型需要的变量确实在里面，否则推理会报错
    needed_vars = model.input_variables + model.forcing_variables
    missing = [v for v in needed_vars if v not in ds_merged]
    if missing:
        print(f"  [Error] 严重警告：缺失模型关键变量: {missing}！")
        # 这里可以选择 return 跳过，或者继续保存但不完整
    
    # 5. Regrid 所有变量
    # 注意：regridder 会自动处理数据集里的所有 data_vars
    # 只要你的非模型变量也是 (time, level, lat, lon) 或者 (time, lat, lon) 格式
    # regridder 都能正确处理插值
    print(f"  [Process] Regridding ALL variables...")
    ds_in_memory = ds_merged.load()
    regridded = xarray_utils.regrid(ds_in_memory, regridder)
    filled = xarray_utils.fill_nan_with_nearest(regridded)
    
    # 6. Regrid & FillNa
    # 这是计算密集型步骤
    print(f"  [Process] Regridding & Filling NaNs...")
    regridded = xarray_utils.regrid(ds_in_memory, regridder)
    filled = xarray_utils.fill_nan_with_nearest(regridded)
    
    # 7. 优化并保存
    print(f"  [Save] 写入 {out_filename} ...")
    encoding = {}
    for var in filled.data_vars:
        encoding[var] = {
            'zlib': True,       # 压缩
            'complevel': 5,     # 压缩等级
            # 强制分块大小：时间维为1，空间维全量
            # 这样 DataLoader 读取任意时刻 t 的速度最快
            'chunksizes': (1,) + filled[var].shape[1:],
            # 确保不保存为 float64 浪费空间，通常 float32 足够
            'dtype': 'float32' 
        }
        
    filled.to_netcdf(out_filename, encoding=encoding)
    
    # 显式释放内存
    ds_p.close()
    ds_s.close()
    ds_merged.close()
    del ds_in_memory, filled
    
def main():
    # 0. 环境设置
    # 禁用 GPU 显存预分配，防止 Regrid 占满显存导致后续 OOM
    # 如果你的 horizontal_interpolation 是 CPU 版，这行没影响
    # 如果是 GPU 版，建议留着，或者设为 '' 强制用 CPU 跑预处理(通常够快了)
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    
    # 1. 加载模型配置
    print("正在加载模型配置...")
    with open(MODEL_PATH, 'rb') as f:
        ckpt = pickle.load(f)
    model = neuralgcm.PressureLevelModel.from_checkpoint(ckpt)
    
    # 2. 扫描目录结构
    print(f"扫描数据目录: {RAW_DATA_ROOT}")
    # 获取所有年份文件夹
    years = sorted([d for d in os.listdir(RAW_DATA_ROOT) 
                    if os.path.isdir(os.path.join(RAW_DATA_ROOT, d)) and d.isdigit()])
    
    regridder = None # 延迟初始化
    
    # 3. 双层循环遍历：年 -> 月
    for year in tqdm(years, desc="Years"):
        year_path = os.path.join(RAW_DATA_ROOT, year)
        months = sorted([d for d in os.listdir(year_path) 
                         if os.path.isdir(os.path.join(year_path, d))])
        
        for month in tqdm(months, desc=f"Months in {year}", leave=False):
            month_path = os.path.join(year_path, month)
            
            # 查找具体文件
            p_files = glob.glob(os.path.join(month_path, PRESSURE_FILE_PATTERN))
            s_files = glob.glob(os.path.join(month_path, SURFACE_FILE_PATTERN))
            
            if not p_files or not s_files:
                print(f"  [Skip] {year}-{month} 缺少 pressure 或 surface 文件")
                continue
                
            # 假设每个月文件夹下只有一个 pressure 和一个 surface 文件
            # 如果有多个，这里取第一个，或者你需要额外的合并逻辑
            p_path = p_files[0]
            s_path = s_files[0]
            
            # 首次运行时初始化 Regridder
            # 我们需要先读取一个文件来获取网格信息
            if regridder is None:
                sample_ds = xr.open_dataset(p_path, chunks={})
                regridder = get_regridder(sample_ds, model.data_coords)
                sample_ds.close()
                
            # 执行处理
            process_and_save_month(year, month, p_path, s_path, model, regridder)
            
    print("\n全部处理完成！")

if __name__ == "__main__":
    main()