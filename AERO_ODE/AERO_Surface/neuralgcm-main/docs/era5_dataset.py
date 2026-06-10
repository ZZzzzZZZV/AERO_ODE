import os
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
from datetime import datetime, timedelta
from typing import List, Tuple, Dict
from collections import OrderedDict

class ERA5GlobalDataset(Dataset):
    """
    NeuralGCM ERA5 数据集 (简化懒加载版)
    """
    def __init__(
        self,
        root_dir: str,
        input_vars: List[str] = [
            'geopotential', 'specific_humidity', 'temperature', 
            'u_component_of_wind', 'v_component_of_wind', 
            'specific_cloud_ice_water_content', 'specific_cloud_liquid_water_content',
            '10m_u_component_of_wind', '10m_v_component_of_wind', 
            '2m_temperature', 'mean_sea_level_pressure', 
            'sea_surface_temperature', 'sea_ice_cover'
        ],
        predict_lead_time: int = 100,
        history_len: int = 1,
        cache_size: int = 8
    ):
        super().__init__()
        self.root_dir = root_dir
        self.input_vars = input_vars
        self.predict_lead_time = predict_lead_time
        self.history_len = history_len
        self.cache_size = cache_size
        
        # LRU 文件缓存
        self._file_cache = OrderedDict()
        
        # 1. 扫描文件并建立时间索引
        self.files = self._scan_files()
        self.timeline = self._build_timeline()
        self.valid_indices = self._filter_valid_indices()
        
        if not self.valid_indices:
            raise ValueError(f"未找到足够连续的数据样本 (root: {root_dir})")

    def _scan_files(self) -> List[Tuple[str, datetime]]:
        """扫描所有NC文件并按时间排序"""
        files = []
        for root, _, filenames in os.walk(self.root_dir):
            for f in filenames:
                if f.endswith(".nc"):
                    try:
                        # 解析文件名 YYYYMMDD.nc
                        dt = datetime.strptime(f.split('.')[0], "%Y%m%d")
                        files.append((os.path.join(root, f), dt))
                    except: continue
        return sorted(files, key=lambda x: x[1])

    def _build_timeline(self):
        """将文件列表展开为小时级的时间线"""
        timeline = []
        for file_idx, (path, start_dt) in enumerate(self.files):
            for h in range(24): # 假设每个文件24小时
                timeline.append({
                    'path': path,
                    'time': start_dt + timedelta(hours=h),
                    'hour_idx': h
                })
        return timeline

    def _filter_valid_indices(self):
        """筛选出满足预测时长的起始点"""
        valid = []
        total = len(self.timeline)
        required = self.predict_lead_time
        
        for i in range(total - required):
            # 简单检查首尾时间差是否符合要求（保证连续）
            t_start = self.timeline[i]['time']
            t_end = self.timeline[i + required]['time']
            if (t_end - t_start).total_seconds() / 3600 == required:
                valid.append(i)
        return valid

    def _get_file_handle(self, path):
        """获取文件句柄（带缓存）"""
        if path in self._file_cache:
            self._file_cache.move_to_end(path)
            return self._file_cache[path]
        
        if len(self._file_cache) >= self.cache_size:
            self._file_cache.popitem(last=False)[1].close()
            
        # 懒加载打开
        ds = xr.open_dataset(path, engine='netcdf4')
        self._file_cache[path] = ds
        return ds

    def _load_data(self, start_idx, length):
        """核心读取逻辑：按文件分组读取，减少IO次数"""
        # 1. 截取需要的时间段信息
        slice_info = self.timeline[start_idx : start_idx + length]
        if not slice_info: return {}

        # 2. 按文件分组 (path -> [start_h, end_h])
        # 这是一个简化处理，假设切片内同一文件的索引是连续的
        groups = []
        if len(slice_info) > 0:
            curr_path = slice_info[0]['path']
            curr_start = slice_info[0]['hour_idx']
            curr_count = 0
            
            for item in slice_info:
                if item['path'] == curr_path:
                    curr_count += 1
                else:
                    groups.append((curr_path, curr_start, curr_start + curr_count))
                    curr_path = item['path']
                    curr_start = item['hour_idx']
                    curr_count = 1
            groups.append((curr_path, curr_start, curr_start + curr_count))

        # 3. 执行读取
        buffer = {v: [] for v in self.input_vars}
        
        for path, start, end in groups:
            ds = self._get_file_handle(path)
            time_slice = slice(start, end)
            
            for var in self.input_vars:
                # 这一步 .values 才会真正读取磁盘
                try:
                    data = ds[var].isel(time=time_slice).values
                    buffer[var].append(data)
                except KeyError:
                    raise KeyError(f"变量 {var} 缺失于文件 {path}")

        # 4. 拼接并转Tensor
        return {
            k: torch.from_numpy(np.concatenate(v, axis=0)).float() 
            for k, v in buffer.items()
        }

    def __getitem__(self, idx):
        global_idx = self.valid_indices[idx]
        
        # 读取初始场 (t=0)
        input_data = self._load_data(global_idx, self.history_len)
        # 去掉时间维度 (如果只有1帧)
        final_input = {k: v.squeeze(0) if self.history_len == 1 else v for k, v in input_data.items()}
        
        # 读取时间字符串
        time_str = self.timeline[global_idx]['time'].strftime("%Y/%m/%d/%H")
        
        # 读取未来场 (t=1 ~ t=100)
        target_data = self._load_data(global_idx + 1, self.predict_lead_time)
        
        return final_input, time_str, target_data

    def __len__(self):
        return len(self.valid_indices)

    def __del__(self):
        if hasattr(self, '_file_cache'):
            for ds in self._file_cache.values():
                ds.close()

if __name__ == "__main__":
    # 测试代码
    root = r"/nfs/samba/数据聚变/气象数据/ERA5_Global_37Level_01Degree/Regridded_and_Filled"
    if not os.path.exists(root):
        print("路径不存在，请检查挂载。")
    else:
        ds = ERA5GlobalDataset(root, cache_size=10)
        loader = DataLoader(ds, batch_size=3, num_workers=0)
        
        print("开始测试...")
        t0 = time.time()
        for i, batch in enumerate(loader):
            print(f"Batch {i} 耗时: {time.time()-t0:.2f}s")
            t0 = time.time()
            if i >= 2: break