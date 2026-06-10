import os
import glob
import subprocess
import shutil
from pathlib import Path
from tqdm import tqdm

# ==================== 配置 ====================
INPUT_FILE = "/nfs/samba/数据聚变/气象数据/ERA5_Global_37Level_01Degree/surface/2021.nc"           # 你的地表数据文件名
TARGET_ROOT = "/nfs/samba/数据聚变/气象数据/ERA5_Global_37Level_01Degree/nc/2021" # 目标根目录 (与之前保持一致)
OVERWRITE = False                # 是否覆盖已存在文件
# =============================================

def split_surface_data():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 错误: 找不到输入文件 {INPUT_FILE}")
        return

    print("="*60)
    print("  地表数据逐天分割工具 (Yearly -> Daily)")
    print(f"  输入文件: {INPUT_FILE}")
    print(f"  目标目录: {TARGET_ROOT}")
    print("="*60 + "\n")

    # 创建临时工作目录
    temp_root = "temp_surface_processing"
    if os.path.exists(temp_root):
        shutil.rmtree(temp_root)
    os.makedirs(temp_root, exist_ok=True)

    try:
        # 第一步：按月分割 (splitmon)
        # 必须先按月分，否则直接splitday会把1月1日、2月1日...合并在一起
        print("Wait... 正在按月拆分大文件 (这可能需要一点时间)...")
        
        # cdo splitmon input prefix
        # 结果将是: temp_root/mon202401.nc, temp_root/mon202402.nc ...
        mon_prefix = os.path.join(temp_root, "mon")
        subprocess.run(['cdo', '-s', 'splitmon', INPUT_FILE, mon_prefix], check=True)
        
        mon_files = sorted(glob.glob(os.path.join(temp_root, "mon*.nc")))
        print(f"✓ 已拆分为 {len(mon_files)} 个月文件，准备逐日处理...\n")

        # 第二步：遍历每个月，按天分割 (splitday)
        total_days_processed = 0
        
        for mon_file in tqdm(mon_files, desc="处理月份"):
            # 解析年份和月份
            # 文件名类似: .../mon202401.nc
            fname = Path(mon_file).stem # mon202401
            date_str = fname.replace("mon", "") # 202401
            year = date_str[:4]
            month = date_str[4:6]
            
            # 为该月创建每一天的临时目录
            day_temp_dir = os.path.join(temp_root, f"days_{month}")
            os.makedirs(day_temp_dir, exist_ok=True)
            
            # cdo splitday mon_file prefix
            # 结果: day01.nc, day02.nc ...
            day_prefix = os.path.join(day_temp_dir, "day")
            subprocess.run(['cdo', '-s', 'splitday', mon_file, day_prefix], check=True)
            
            # 分发该月的所有天文件
            day_files = glob.glob(os.path.join(day_temp_dir, "day*.nc"))
            
            for day_file in day_files:
                # 解析日期
                # 文件名: day01.nc
                d_name = Path(day_file).stem # day01
                day_num = d_name.replace("day", "")
                
                # 确保是2位数字
                if len(day_num) == 1:
                    day_num = "0" + day_num
                    
                # 构建目标路径: TARGET/YYYY/MM/DD/_surface_data.nc
                target_dir = os.path.join(TARGET_ROOT, year, month, day_num)
                target_path = os.path.join(target_dir, "_surface_data.nc")
                
                # 如果之前的高空数据没有创建这个目录，这里补上
                os.makedirs(target_dir, exist_ok=True)
                
                if os.path.exists(target_path) and not OVERWRITE:
                    continue
                    
                shutil.move(day_file, target_path)
                total_days_processed += 1
            
            # 清理该月的临时文件
            shutil.rmtree(day_temp_dir)

        print("\n" + "="*60)
        print("处理完成!")
        print(f"  ✓ 总共分发天数: {total_days_processed}")
        print(f"  ✓ 输出文件名: _surface_data.nc")
        print("="*60)

    except subprocess.CalledProcessError as e:
        print(f"\n❌ CDO 执行错误: {e}")
    except Exception as e:
        print(f"\n❌ 发生异常: {e}")
    finally:
        # 清理所有临时文件
        if os.path.exists(temp_root):
            try:
                shutil.rmtree(temp_root)
            except:
                pass

if __name__ == "__main__":
    split_surface_data()