import xarray as xr
import os
import argparse
import numpy as np

def check_coordinates_diff(dirpath):
    print(f"\n========================================")
    print(f"正在检查目录: {dirpath}")
    print(f"========================================")
    
    sfc_path = os.path.join(dirpath, '_surface_data.nc')
    pl_path = os.path.join(dirpath, '_upper_air_data.nc')
    
    if not os.path.exists(sfc_path) or not os.path.exists(pl_path):
        print("错误: 找不到输入文件")
        return

    try:
        # 打开文件 (不使用 chunks，直接读取坐标)
        ds_sfc = xr.open_dataset(sfc_path)
        ds_pl = xr.open_dataset(pl_path)
        
        print(f"\n[1] 维度大小对比:")
        print(f"  高空 (PL): {dict(ds_pl.dims)}")
        print(f"  地表 (SFC): {dict(ds_sfc.dims)}")
        
        # 检查经度 (lon)
        print(f"\n[2] 经度 (lon) 对比:")
        lon_pl = ds_pl['lon'].values
        lon_sfc = ds_sfc['lon'].values
        
        if len(lon_pl) != len(lon_sfc):
            print(f"  ! 长度不同: PL={len(lon_pl)}, SFC={len(lon_sfc)}")
        else:
            print(f"  长度一致: {len(lon_pl)}")
            # 检查数值差异
            diff = np.abs(lon_pl - lon_sfc)
            max_diff = np.max(diff)
            if max_diff == 0:
                print(f"  √ 数值完全一致")
            else:
                print(f"  ! 数值存在差异 (Max Diff: {max_diff:.10f})")
                print(f"    PL  Top 3: {lon_pl[:3]}")
                print(f"    SFC Top 3: {lon_sfc[:3]}")
                
        # 检查纬度 (lat)
        print(f"\n[3] 纬度 (lat) 对比:")
        lat_pl = ds_pl['lat'].values
        lat_sfc = ds_sfc['lat'].values
        
        if len(lat_pl) != len(lat_sfc):
            print(f"  ! 长度不同: PL={len(lat_pl)}, SFC={len(lat_sfc)}")
        else:
            diff_lat = np.abs(lat_pl - lat_sfc)
            max_diff_lat = np.max(diff_lat)
            if max_diff_lat == 0:
                print(f"  √ 数值完全一致")
            else:
                print(f"  ! 数值存在差异 (Max Diff: {max_diff_lat:.10f})")

    except Exception as e:
        print(f"读取出错: {e}")

if __name__ == "__main__":
    # ================= 配置区域 =================
    # 设置要检查的目录路径
    # 例如: TARGET_DIR = r"/nfs/samba/.../2021/01/01"
    TARGET_DIR = "/nfs/samba/数据聚变/气象数据/ERA5_Global_37Level_01Degree/nc/2021/01/01"
    # ===========================================

    check_coordinates_diff(TARGET_DIR)

