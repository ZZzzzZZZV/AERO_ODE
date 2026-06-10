import os
import glob
import subprocess
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
import time
import warnings
import signal

warnings.filterwarnings('ignore')

# ==================== 配置 ====================
GRIB_DIR = r"/nfs/samba/数据聚变/气象数据/ERA5_Global_37Level_01Degree/grib/2021_b"
TARGET_DIR = r"/nfs/samba/数据聚变/气象数据/ERA5_Global_37Level_01Degree/nc_Primitive_Data/2021"

# GRIB_DIR = r"/nfs/samba/数据聚变/气象数据/ERA5_Global_37Level_01Degree/grib/surface"
# TARGET_DIR = r"/nfs/samba/数据聚变/气象数据/ERA5_Global_37Level_01Degree/surface"


NUM_PROCESSES = 1          # 串行处理（最稳定）
USE_COMPRESSION = False    # 不压缩（最快）
NO_TIMEOUT = True          # 关键：无超时限制
# =============================================


def get_file_size_gb(filepath):
    """获取文件大小（GB）"""
    return os.path.getsize(filepath) / (1024**3)


def convert_no_timeout(grib_file, target_dir):
    """
    无超时限制转换
    会一直等待直到完成或出错
    """
    file_name = Path(grib_file).stem
    output_path = os.path.join(target_dir, f"{file_name}.nc")
    
    file_size_gb = get_file_size_gb(grib_file)
    
    if os.path.exists(output_path):
        output_size = get_file_size_gb(output_path)
        return {
            'file': grib_file,
            'status': 'skipped',
            'message': f'已存在 ({output_size:.1f}GB)',
            'time': 0,
            'size_gb': file_size_gb
        }
    
    print(f"\n{'='*70}")
    print(f"开始: {file_name}")
    print(f"  大小: {file_size_gb:.2f} GB")
    print(f"  时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    
    start_time = time.time()
    
    try:
        # ================= 改动部分 =================
        # 基础命令参数
        # -t ecmwf: 强制使用ECMWF码表，确保变量名正确 (如 t, u, v 而不是 var130)
        # -f nc4:   输出 NetCDF4 格式
        base_cmd = ['cdo', '-t', 'ecmwf', '-s', '-O', '-f', 'nc4']
        
        if USE_COMPRESSION:
            # -z zip_1: 使用 Deflate 压缩级别 1
            cmd = base_cmd + ['-z', 'zip_1', 'copy', grib_file, output_path]
        else:
            cmd = base_cmd + ['copy', grib_file, output_path]
        # ===========================================
        
        # 关键：不设置timeout参数，会一直等待
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(grib_file) if os.path.dirname(grib_file) else '.'
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode != 0:
            print(f"\n✗ 失败: {file_name}")
            print(f"  错误: {result.stderr[:200]}")
            return {
                'file': grib_file,
                'status': 'failed',
                'message': f'CDO错误: {result.stderr[:80]}',
                'time': elapsed,
                'size_gb': file_size_gb
            }
        
        # 验证输出
        if not os.path.exists(output_path):
            print(f"\n✗ 失败: {file_name} - 输出文件未创建")
            return {
                'file': grib_file,
                'status': 'failed',
                'message': '输出文件未创建',
                'time': elapsed,
                'size_gb': file_size_gb
            }
        
        output_size_gb = get_file_size_gb(output_path)
        speed_mb_s = (file_size_gb * 1024) / elapsed if elapsed > 0 else 0
        
        print(f"\n✓ 完成: {file_name}")
        print(f"  耗时: {elapsed/60:.1f} 分钟 ({elapsed:.0f} 秒)")
        print(f"  输出: {output_size_gb:.2f} GB")
        print(f"  速度: {speed_mb_s:.1f} MB/s")
        
        return {
            'file': grib_file,
            'status': 'success',
            'message': f'✓ {elapsed/60:.1f}min',
            'time': elapsed,
            'size_gb': file_size_gb,
            'output_size_gb': output_size_gb,
            'speed_mb_s': speed_mb_s
        }
        
    except KeyboardInterrupt:
        # 用户按Ctrl+C
        print(f"\n⚠️  用户中断: {file_name}")
        if os.path.exists(output_path):
            os.remove(output_path)
        raise  # 向上传递，整个程序会停止
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n✗ 异常: {file_name}")
        print(f"  错误: {str(e)}")
        if os.path.exists(output_path):
            try:
                os.remove(output_path)
            except:
                pass
        return {
            'file': grib_file,
            'status': 'failed',
            'message': f'异常: {str(e)[:50]}',
            'time': elapsed,
            'size_gb': file_size_gb
        }


def format_time_remaining(seconds):
    """格式化剩余时间"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    return f"{hours}小时{minutes}分钟"


def main():
    print("\n" + "="*70)
    print("  ERA5 → NetCDF 转换工具 (已修正变量名)")
    print("  无超时限制 | 不会中断 | 强制ECMWF码表")
    print("="*70 + "\n")
    
    # 检查CDO
    try:
        subprocess.run(['cdo', '--version'], capture_output=True, timeout=5, check=True)
        print("✓ CDO已就绪\n")
    except:
        print("❌ CDO不可用，请确保已安装 cdo\n")
        return
    
    Path(TARGET_DIR).mkdir(parents=True, exist_ok=True)
    
    # 确保目录存在再切换
    if not os.path.exists(GRIB_DIR):
        print(f"❌ 输入目录不存在: {GRIB_DIR}")
        return

    os.chdir(GRIB_DIR)
    files = []
    for ext in ['*.grib', '*.grib2', '*.grb']:
        files.extend(glob.glob(ext))
    files = sorted(files)
    
    if not files:
        print(f"❌ 在 {GRIB_DIR} 未找到GRIB文件\n")
        return
    
    # 统计文件大小
    file_sizes = [(f, get_file_size_gb(f)) for f in files]
    total_size = sum(size for _, size in file_sizes)
    
    print(f"📊 文件统计:")
    print(f"  数量: {len(files)} 个")
    print(f"  总大小: {total_size:.1f} GB")
    print(f"  平均: {total_size/len(files):.2f} GB/文件")
    
    # 预估时间
    estimated_seconds = total_size * 30
    print(f"\n⏱️  预计时间: {format_time_remaining(estimated_seconds)}")
    print(f"     (保守估计，实际可能更快)")
    
    print(f"\n⚙️  处理模式:")
    print(f"  并行: {NUM_PROCESSES} 进程")
    print(f"  压缩: {'关闭' if not USE_COMPRESSION else '最低级别'}")
    print(f"  变量: 强制 ECMWF 命名 (如 t, u, v)")
    
    print(f"\n📁 输出目录: {TARGET_DIR}")
    print(f"\n💡 提示: 按 Ctrl+C 可随时安全中断")
    
    input(f"\n按 Enter 开始转换...")
    
    print(f"\n🚀 开始转换...\n")
    
    overall_start = time.time()
    results = []
    
    try:
        for i, (file, file_size) in enumerate(file_sizes, 1):
            print(f"\n[{i}/{len(files)}] 进度: {i/len(files)*100:.1f}%")
            
            # 如果已经处理过，快速跳过
            output_path = os.path.join(TARGET_DIR, f"{Path(file).stem}.nc")
            if os.path.exists(output_path):
                print(f"⊘ 跳过: {Path(file).name} (已存在)")
                results.append({
                    'file': file,
                    'status': 'skipped',
                    'time': 0,
                    'size_gb': file_size
                })
                continue
            
            # 转换
            result = convert_no_timeout(file, TARGET_DIR)
            results.append(result)
            
            # 显示进度和预估
            if len(results) > 0:
                completed = sum(1 for r in results if r['status'] in ['success', 'skipped'])
                total_time_so_far = time.time() - overall_start
                
                # 仅计算实际处理过的文件的时间（排除瞬间跳过的）
                processed_count = sum(1 for r in results if r['status'] == 'success')
                processed_time = sum(r['time'] for r in results if r['status'] == 'success')
                
                if processed_count > 0:
                     avg_time_per_file = processed_time / processed_count
                else:
                     avg_time_per_file = 0

                remaining_files = len(files) - completed
                
                # 如果没有处理过任何文件，使用保守估计，否则使用实际平均值
                est_base = avg_time_per_file if avg_time_per_file > 0 else 30 * (total_size/len(files))
                estimated_remaining = est_base * remaining_files
                
                print(f"\n📈 当前进度:")
                print(f"  已完成: {completed}/{len(files)}")
                print(f"  已用时: {total_time_so_far/60:.1f} 分钟")
                if estimated_remaining > 0:
                    print(f"  预计剩余: {format_time_remaining(estimated_remaining)}")
                    eta = time.time() + estimated_remaining
                    print(f"  预计完成: {time.strftime('%H:%M:%S', time.localtime(eta))}")
    
    except KeyboardInterrupt:
        print(f"\n\n⚠️  用户中断操作")
        print(f"  已完成的文件已保存，下次运行会自动跳过")
    
    total_time = time.time() - overall_start
    
    # 最终统计
    print("\n\n" + "="*70)
    print("  转换完成统计")
    print("="*70)
    
    success = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'failed')
    skipped = sum(1 for r in results if r['status'] == 'skipped')
    
    print(f"  ✓ 成功: {success}")
    print(f"  ✗ 失败: {failed}")
    print(f"  ⊘ 跳过: {skipped}")
    
    if total_time > 0:
        print(f"\n  ⏱️  时间:")
        print(f"  总耗时: {total_time/3600:.2f} 小时")
    
    print("="*70)
    
    # 失败文件
    failed_results = [r for r in results if r['status'] == 'failed']
    if failed_results:
        print(f"\n失败文件 ({len(failed_results)}个):")
        for r in failed_results:
            print(f"  ✗ {Path(r['file']).name}: {r['message']}")
            
    print(f"\n✅ 完成！输出目录: {TARGET_DIR}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序已终止")
    except Exception as e:
        print(f"\n\n❌ 程序异常: {e}")
        import traceback
        traceback.print_exc()