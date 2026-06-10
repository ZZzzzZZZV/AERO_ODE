import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np
import jax
import glob
import os
from datetime import datetime, timedelta
from collections import OrderedDict
import re

# 该dataset假设数据已经过预处理，存储在 NetCDF 文件中
# 每个 nc 文件包含 24 小时数据，命名格式: YYYYMMDD.nc
# 目录结构: 年/月/日/YYYYMMDD.nc

class LRUCache:
    """简单的 LRU 缓存，用于缓存已加载的 xarray Dataset"""
    
    def __init__(self, max_size=10):
        self.cache = OrderedDict()
        self.max_size = max_size
    
    def get(self, key):
        if key in self.cache:
            # 移到最后（最近使用）
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                # 删除最久未使用的
                self.cache.popitem(last=False)
            self.cache[key] = value
    
    def clear(self):
        self.cache.clear()


class NeuralGCMDataset(Dataset):
    """
    带 LRU 缓存的懒加载 NeuralGCM 数据集
    
    特点:
    - 使用 LRU 缓存避免重复读取文件
    - 缓存最近使用的 N 个文件（默认10个）
    - 连续采样时命中率高，I/O 开销小
    """
    
    def __init__(self, 
                 data_dir, 
                 model_helper, 
                 prediction_steps=96,  # 默认4*24=96步
                 file_pattern="**/*.nc",
                 recursive=True,
                 hours_per_file=24,
                 cache_size=10):  # 缓存文件数
        """
        参数:
            data_dir: 数据根目录 (包含 年/月/日/YYYYMMDD.nc 的结构)
            model_helper: NeuralGCM 模型实例
            prediction_steps: 预测步数 (默认96, 即4天)
            file_pattern: 文件匹配模式
            recursive: 是否递归搜索
            hours_per_file: 每个文件包含的小时数 (默认24)
            cache_size: LRU 缓存大小 (默认10个文件)
        """
        self.model = model_helper
        self.steps = prediction_steps
        self.hours_per_file = hours_per_file
        self.data_dir = data_dir
        
        # LRU 缓存
        self.cache = LRUCache(max_size=cache_size)
        
        # 1. 索引所有文件路径（不打开文件）
        if recursive:
            files = sorted(glob.glob(os.path.join(data_dir, file_pattern), recursive=True))
        else:
            files = sorted(glob.glob(os.path.join(data_dir, file_pattern)))
        
        if not files:
            raise ValueError(f"在 {data_dir} 下未找到匹配 {file_pattern} 的文件")
        
        print(f"找到 {len(files)} 个 NetCDF 文件，正在建立索引...")
        
        # 2. 从文件名解析日期，建立文件索引
        self.file_list = []
        self.date_to_file = {}
        
        for f in files:
            basename = os.path.basename(f)
            match = re.match(r'(\d{8})\.nc', basename)
            if match:
                date_str = match.group(1)
                try:
                    date = datetime.strptime(date_str, '%Y%m%d')
                    self.file_list.append((date, f))
                    self.date_to_file[date] = f
                except ValueError:
                    continue
        
        if not self.file_list:
            raise ValueError("未能从文件名中解析出有效日期")
        
        self.file_list.sort(key=lambda x: x[0])
        
        # 3. 计算有效样本数
        self.files_needed = (self.steps + self.hours_per_file) // self.hours_per_file + 1
        self.total_hours = len(self.file_list) * self.hours_per_file
        self.valid_len = self.total_hours - self.steps
        
        if self.valid_len <= 0:
            raise ValueError(f"数据长度 ({self.total_hours} 小时) 不足以支持预测步长 ({self.steps})")
        
        self.start_date = self.file_list[0][0]
        self.end_date = self.file_list[-1][0]
        
        print(f"数据集索引完成。")
        print(f"  时间范围: {self.start_date.strftime('%Y-%m-%d')} ~ {self.end_date.strftime('%Y-%m-%d')}")
        print(f"  文件数: {len(self.file_list)}")
        print(f"  总时间步: {self.total_hours}")
        print(f"  有效样本: {self.valid_len}")
        print(f"  LRU 缓存大小: {cache_size} 文件")

    def __len__(self):
        return self.valid_len

    def _load_file(self, file_path):
        """
        从缓存加载文件，如果不在缓存中则读取并缓存
        """
        # 尝试从缓存获取
        cached = self.cache.get(file_path)
        if cached is not None:
            return cached
        
        # 缓存未命中，读取文件
        ds = xr.open_dataset(file_path, engine='netcdf4').load()
        
        # 处理坐标命名
        if "isobaricInhPa" in ds.coords:
            ds = ds.rename({"isobaricInhPa": "level"})
        
        # 确保 level 维度排序正确
        if ds.level[0] > ds.level[-1]:
            ds = ds.isel(level=slice(None, None, -1))
        
        # 放入缓存
        self.cache.put(file_path, ds)
        
        return ds

    def _get_files_for_window(self, start_hour_idx, num_hours):
        """获取覆盖指定时间窗口的文件信息"""
        files_info = []
        current_hour = start_hour_idx
        remaining_hours = num_hours
        
        while remaining_hours > 0:
            file_idx = current_hour // self.hours_per_file
            hour_in_file = current_hour % self.hours_per_file
            
            if file_idx >= len(self.file_list):
                raise IndexError(f"时间索引 {current_hour} 超出数据范围")
            
            hours_from_this_file = min(
                self.hours_per_file - hour_in_file,
                remaining_hours
            )
            
            _, file_path = self.file_list[file_idx]
            files_info.append((
                file_path,
                hour_in_file,
                hour_in_file + hours_from_this_file
            ))
            
            current_hour += hours_from_this_file
            remaining_hours -= hours_from_this_file
        
        return files_info

    def __getitem__(self, idx):
        """
        返回:
            Dict: {
                'inputs': ...,          # 模型输入 (t=0)
                'input_forcings': ...,  # 输入强迫 (t=0)
                'future_forcings': ..., # 未来强迫 (t=0 到 t=steps)
                'init_time_val': ...,   # 起报时间 (int64, nanoseconds)
                'targets': ...          # 真值数据
            }
        """
        # 1. 确定需要的文件和时间范围
        num_hours_needed = self.steps + 1
        files_info = self._get_files_for_window(idx, num_hours_needed)
        
        # 2. 从缓存加载并拼接数据
        slices = []
        for file_path, start_h, end_h in files_info:
            ds = self._load_file(file_path)
            slices.append(ds.isel(time=slice(start_h, end_h)))
        
        # 拼接
        if len(slices) == 1:
            ds_window = slices[0]
        else:
            ds_window = xr.concat(slices, dim='time')
        
        # 3. 筛选模型需要的变量
        ds_model = ds_window[self.model.input_variables + self.model.forcing_variables]
        
        # 4. 构造模型输入
        inputs = self.model.inputs_from_xarray(ds_model.isel(time=0))
        input_forcings = self.model.forcings_from_xarray(ds_model.isel(time=0))
        future_forcings = self.model.forcings_from_xarray(ds_model)
        
        # 5. 准备真值
        target_vars = ['temperature', 'u_component_of_wind', 'v_component_of_wind', 
                       'geopotential', 'specific_humidity']
        targets_dict = {}
        for var in target_vars:
            if var in ds_window:
                targets_dict[var] = ds_window[var].values
        
        # 6. 获取起报时间
        init_time = ds_window.time.values[0]
        
        return {
            'inputs': inputs,
            'input_forcings': input_forcings,
            'future_forcings': future_forcings,
            'targets': targets_dict,
            'init_time_val': np.datetime64(init_time, 'ns').astype(np.int64)
        }
    
    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()


# ==========================================
# 配套的 Collate Function
# ==========================================

def jax_collate_fn(batch_list):
    """
    将 list of dicts 转换为 dict of stacked arrays
    专门处理 JAX 的 PyTree 结构
    """
    result = {}
    keys = batch_list[0].keys()
    
    for key in keys:
        items = [d[key] for d in batch_list]
        
        if key == 'init_time_val':
            result[key] = np.stack(items, axis=0)
            
        elif key == 'targets':
            stacked_targets = {}
            target_vars = items[0].keys()
            for var in target_vars:
                var_arrays = [item[var] for item in items]
                stacked_targets[var] = np.stack(var_arrays, axis=0)
            result[key] = stacked_targets
            
        else:
            result[key] = jax.tree_util.tree_map(
                lambda *args: np.stack(args, axis=0), 
                *items
            )
            
    return result


# ==========================================
# 使用示例
# ==========================================


