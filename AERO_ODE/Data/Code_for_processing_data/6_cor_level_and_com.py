import netCDF4 as nc
import os
import time
from datetime import datetime
import argparse
from concurrent.futures import ProcessPoolExecutor
import shutil

# 配置部分
SRC_ROOT = r"/nfs/samba/数据聚变/气象数据/ERA5_Global_37Level_01Degree/Regridded_and_Filled"
DST_ROOT = r"/nfs/samba/数据聚变/气象数据/ERA5_Global_37Level_01Degree/Uncompressed_Regridded_and_Filled" # 区分目录

TARGET_YEARS = [2021] 
NUM_WORKERS = 2

def convert_single_file(args):
    """
    转换单个 NetCDF 文件的处理函数。
    主要功能：
    1. 复制源文件的维度和变量到目标文件。
    2. 重命名高度维度 'isobaricInhPa' 为 'level'。
    3. 优化存储分块 (Chunking) 策略，加速深度学习读取。
    4. 反转高度维度的数据顺序 (从高空到地面 -> 从地面到高空)。
    """
    src_path, dst_path = args
    
    # --- 检查跳过逻辑 ---
    # 如果目标文件已存在，且包含关键变量 'geopotential'（作为一个检查标记），则跳过
    if os.path.exists(dst_path):
        try:
            with nc.Dataset(dst_path, 'r') as ds:
                if 'geopotential' in ds.variables: 
                    return f"SKIP: {os.path.basename(src_path)}"
        except: pass

    # 确保目标目录存在
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    # 使用临时文件写入，避免处理中断导致产生损坏文件
    temp_path = dst_path + ".tmp"
    
    try:
        t0 = time.time()
        # 同时打开源文件(读)和临时文件(写)
        with nc.Dataset(src_path, 'r') as f_src, nc.Dataset(temp_path, 'w', format='NETCDF4') as f_dst:
            # 1. 复制全局属性
            f_dst.setncatts(f_src.__dict__)
            
            # 2. 复制维度 (Dimensions)
            for name, dimension in f_src.dimensions.items():
                # 将 'isobaricInhPa' 重命名为更通用的 'level'
                target_name = 'level' if name == 'isobaricInhPa' else name
                f_dst.createDimension(
                    target_name, (len(dimension) if not dimension.isunlimited() else None))

            # 3. 复制变量 (Variables)
            for name, variable in f_src.variables.items():
                new_dims = list(variable.dimensions)
                # 更新变量使用的维度名称
                if 'isobaricInhPa' in new_dims:
                    new_dims = ['level' if d == 'isobaricInhPa' else d for d in new_dims]
                
                # 变量名本身也需要重命名（如果它是坐标变量）
                target_var_name = 'level' if name == 'isobaricInhPa' else name
                
                # --- 深度学习优化的 Chunking 策略 ---
                # 目标：让每个时间步(time step)的数据在磁盘上物理连续，
                # 这样训练时一次读取一个样本（一个时间步）速度最快，无需跨磁盘块寻道。
                chunks = None
                if variable.ndim == 4: # 4D 变量: (time, level, lat, lon)
                    # 设置 chunk 为 (1, levels, lat, lon)，即每个时间步是一个独立的块
                    chunks = (1, variable.shape[1], variable.shape[2], variable.shape[3])
                elif variable.ndim == 3 and new_dims[0] == 'time': # 3D 变量: (time, lat, lon)
                    # 表面变量同理，每个时间步一个块
                    chunks = (1, variable.shape[1], variable.shape[2])
                
                # 创建变量，禁用压缩 (zlib=False) 以换取更快的读取速度（空间换时间）
                x = f_dst.createVariable(
                    target_var_name, variable.datatype, tuple(new_dims),
                    zlib=False, 
                    shuffle=False,
                    chunksizes=chunks # 应用优化后的 Chunking
                )
                
                # 复制变量属性
                f_dst[target_var_name].setncatts(variable.__dict__)

                # --- 4. 数据处理与写入 ---
                # 读取源数据到内存
                data = variable[:]

                # 反转高度维度 (level)
                # 原始数据通常是气压层：1000hPa, 975hPa, ... 1hPa (从地面到高空) 或反之
                # 某些模型需要特定的顺序，这里执行反转操作
                if 'level' in new_dims:
                    level_idx = new_dims.index('level')
                    
                    # 动态构建切片对象来进行反转
                    # 对应于 numpy/xarray 的 [:, ::-1, :, :] 操作
                    # slice(None) 等同于 :
                    # slice(None, None, -1) 等同于 ::-1
                    slices = [slice(None)] * len(new_dims)
                    slices[level_idx] = slice(None, None, -1) # 在 level 维度上步长为 -1
                    
                    data = data[tuple(slices)]

                # 将处理后的数据写入目标文件
                f_dst[target_var_name][:] = data
                
        # 只有在成功写入后，才将临时文件重命名为正式文件
        os.replace(temp_path, dst_path)
        return f"DONE: {os.path.basename(src_path)} ({time.time()-t0:.2f}s)"
        
    except Exception as e:
        # 发生错误时清理临时文件
        if os.path.exists(temp_path): os.remove(temp_path)
        return f"FAIL: {os.path.basename(src_path)} Error: {e}"

def main():
    # ... (与之前相同) ...
    tasks = []
    print(f"开始扫描源目录: {SRC_ROOT}")
    for year in TARGET_YEARS:
        year_str = str(year)
        year_dir = os.path.join(SRC_ROOT, year_str)
        if not os.path.exists(year_dir): continue
            
        for root, dirs, files in os.walk(year_dir):
            for file in files:
                if file.endswith(".nc"):
                    src_path = os.path.join(root, file)
                    rel_path = os.path.relpath(src_path, SRC_ROOT)
                    dst_path = os.path.join(DST_ROOT, rel_path)
                    tasks.append((src_path, dst_path))
    
    print(f"共发现 {len(tasks)} 个文件。使用 {NUM_WORKERS} 进程处理。")
    
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        for result in executor.map(convert_single_file, tasks):
            print(result)

if __name__ == "__main__":
    main()