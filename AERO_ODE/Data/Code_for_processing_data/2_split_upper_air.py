import os
import glob
import subprocess
import shutil
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm
import time

# ==================== 配置 ====================
# 输入文件所在目录 (默认为当前目录)
SOURCE_DIR = "/nfs/samba/数据聚变/气象数据/ERA5_Global_37Level_01Degree/nc_Primitive_Data/2021" 

# 输出目录根路径 (会自动创建)
TARGET_DIR = "/nfs/samba/数据聚变/气象数据/ERA5_Global_37Level_01Degree/nc/2021"

# 并行进程数 (根据CPU核心数调整)
NUM_PROCESSES = 2

# 是否覆盖已存在的文件
OVERWRITE = False
# =============================================

def process_single_file(filepath):
    """处理单个NC文件的分割任务"""
    filename = os.path.basename(filepath)
    file_stem = Path(filepath).stem
    
    try:
        # 1. 从文件名解析年份和月份
        # 格式示例: 202401_01-08.nc
        # 取前6位作为 YYYYMM
        date_part = filename.split('_')[0]
        if len(date_part) != 6 or not date_part.isdigit():
            return {
                'status': 'skipped',
                'file': filename,
                'message': '文件名格式不匹配 (期望 YYYYMM_...)'
            }
            
        year = date_part[:4]
        month = date_part[4:6]
        
        # 2. 创建临时目录存放splitday的结果
        temp_dir = os.path.join(TARGET_DIR, "temp_processing", file_stem)
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        
        # 3. 使用 CDO 按天分割
        # 命令: cdo -s splitday infile outfile_prefix
        # 输出将是 prefix01.nc, prefix02.nc 等
        output_prefix = os.path.join(temp_dir, "day")
        
        cmd = ["cdo", "-s", "splitday", filepath, output_prefix]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            return {
                'status': 'failed',
                'file': filename,
                'message': f"CDO splitday 错误: {result.stderr}"
            }
            
        # 4. 将分割后的文件移动到目标结构中
        split_files = glob.glob(os.path.join(temp_dir, "day*.nc"))
        processed_days = []
        
        for split_file in split_files:
            # 获取日期 (DD)
            # 文件名类似 day01.nc
            split_filename = os.path.basename(split_file)
            day_str = split_filename.replace("day", "").replace(".nc", "")
            
            # 确保是2位数字 (cdo通常输出2位, 但以防万一)
            if len(day_str) == 1:
                day_str = f"0{day_str}"
            
            # 构建目标路径: TARGET/YYYY/MM/DD/_upper_air_data.nc
            day_dir = os.path.join(TARGET_DIR, year, month, day_str)
            target_file = os.path.join(day_dir, "_upper_air_data.nc")
            
            # 检查文件是否已存在
            if os.path.exists(target_file) and not OVERWRITE:
                continue
                
            # 创建目录并移动
            os.makedirs(day_dir, exist_ok=True)
            shutil.move(split_file, target_file)
            processed_days.append(day_str)
            
        # 5. 清理临时目录
        shutil.rmtree(temp_dir)
        
        return {
            'status': 'success',
            'file': filename,
            'days': len(processed_days)
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'file': filename,
            'message': str(e)
        }

def main():
    print("="*60)
    print("  NC文件按天分割工具")
    print(f"  源目录: {os.path.abspath(SOURCE_DIR)}")
    print(f"  目标目录: {os.path.abspath(TARGET_DIR)}")
    print("="*60 + "\n")
    
    # 检查CDO
    if shutil.which('cdo') is None:
        print("❌ 错误: 未找到 'cdo' 命令，请确保已安装 CDO。")
        return

    # 扫描文件
    search_pattern = os.path.join(SOURCE_DIR, "*.nc")
    files = sorted(glob.glob(search_pattern))
    
    if not files:
        print("❌ 未找到 .nc 文件")
        return
        
    print(f"找到 {len(files)} 个文件，准备处理...")
    
    # 创建基础临时目录
    os.makedirs(os.path.join(TARGET_DIR, "temp_processing"), exist_ok=True)
    
    # 并行处理
    results = []
    with Pool(NUM_PROCESSES) as p:
        # 使用tqdm显示进度条
        for result in tqdm(p.imap(process_single_file, files), total=len(files)):
            results.append(result)
            
            # 实时打印错误
            if result['status'] in ['failed', 'error']:
                tqdm.write(f"\n❌ {result['file']}: {result.get('message', '未知错误')}")
    
    # 清理空的临时总目录
    try:
        os.rmdir(os.path.join(TARGET_DIR, "temp_processing"))
    except:
        pass # 目录可能不为空或已被删除
        
    # 统计
    success = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] in ['failed', 'error'])
    skipped = sum(1 for r in results if r['status'] == 'skipped')
    
    print("\n" + "="*60)
    print(f"处理完成:")
    print(f"  ✓ 成功: {success}")
    print(f"  ✗ 失败: {failed}")
    print(f"  ⊘ 跳过: {skipped}")
    print(f"  输出目录: {TARGET_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()