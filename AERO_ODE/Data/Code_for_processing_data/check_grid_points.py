import xarray as xr
import os
import argparse
import numpy as np

def check_grid_point_alignment(dirpath):
    print(f"\n========================================")
    print(f"正在深入检查网格点对齐: {dirpath}")
    print(f"========================================")
    
    sfc_path = os.path.join(dirpath, '_surface_data.nc')
    pl_path = os.path.join(dirpath, '_upper_air_data.nc')
    
    try:
        # 打开原始文件
        with xr.open_dataset(sfc_path) as ds_sfc, xr.open_dataset(pl_path) as ds_pl:
            
            # 获取经度数组
            lon_pl = ds_pl['lon'].values
            lon_sfc = ds_sfc['lon'].values
            
            # 标准化处理 (模拟 process_weather_data_final.py 中的逻辑)
            # 将 (0, 360) 转换为 (-180, 180) 并排序
            lon_pl_std = np.sort((lon_pl + 180) % 360 - 180)
            lon_sfc_std = np.sort((lon_sfc + 180) % 360 - 180)
            
            print(f"\n[1] 经度网格点检查:")
            print(f"  高空点数: {len(lon_pl_std)}")
            print(f"  地表点数: {len(lon_sfc_std)}")
            
            if len(lon_pl_std) != len(lon_sfc_std):
                print(f"  × 错误: 网格点数量不一致!")
                return

            # 计算逐点差异
            # 如果两个网格完全一样，这里的 diff 应该全是 0 (或者极小的浮点误差)
            diff = np.abs(lon_pl_std - lon_sfc_std)
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            print(f"  最大逐点偏差 (Max Diff): {max_diff:.10f}")
            print(f"  平均逐点偏差 (Mean Diff): {mean_diff:.10f}")
            
            if max_diff < 1e-4:
                print(f"  √ 完美对齐: 所有网格点在数学上重合。")
            else:
                print(f"  × 警告: 网格点存在显著偏差!")
                # 打印出偏差最大的前几个点
                bad_indices = np.where(diff > 1e-4)[0]
                print(f"    发现 {len(bad_indices)} 个不重合的点。")
                print(f"    示例 (前3个):")
                for idx in bad_indices[:3]:
                    print(f"      Idx {idx}: PL={lon_pl_std[idx]:.4f}, SFC={lon_sfc_std[idx]:.4f}, Diff={diff[idx]:.4f}")

            # 同样检查纬度
            print(f"\n[2] 纬度网格点检查:")
            lat_pl = np.sort(ds_pl['lat'].values)
            lat_sfc = np.sort(ds_sfc['lat'].values)
            
            diff_lat = np.abs(lat_pl - lat_sfc)
            max_diff_lat = np.max(diff_lat)
            
            print(f"  最大逐点偏差 (Max Diff): {max_diff_lat:.10f}")
            if max_diff_lat < 1e-4:
                print(f"  √ 完美对齐")
            else:
                print(f"  × 警告: 纬度存在偏差!")

    except Exception as e:
        print(f"检查出错: {e}")

if __name__ == "__main__":
    # ================= 配置区域 =================
    # 修改为您想检查的【原始输入目录】
    # 我们需要直接读取原始文件，手动模拟转换过程来验证数学上是否成立
    TARGET_DIR = "/nfs/samba/数据聚变/气象数据/ERA5_Global_37Level_01Degree/nc/2021/01/01"
    # ===========================================
    
    check_grid_point_alignment(TARGET_DIR)

