"""
诊断 Dataset 性能瓶颈的脚本

运行此脚本可以定位是哪个环节最慢：
1. 文件 I/O
2. 数据预处理
3. 模型编码
4. 模型推理
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import time
import numpy as np
import pickle
import xarray as xr
import jax
import neuralgcm
from Dataset import NeuralGCMDataset, jax_collate_fn
from torch.utils.data import DataLoader

# ==========================================
# 配置
# ==========================================
MODEL_PATH = '/nfs/gpu_homes/gpu09/home/zhangjing/Code/NeuralGCM/pkl/neuralgcm_04_30_2024_neural_gcm_dynamic_forcing_deterministic_1_4_deg.pkl'
DATA_DIR = '/nfs/samba/数据聚变/气象数据/ERA5_Global_37Level_01Degree/Regridded_and_Filled'
PREDICTION_STEPS = 96
NUM_SAMPLES = 5  # 测试样本数

print("=" * 60)
print("NeuralGCM 性能诊断")
print("=" * 60)

# ==========================================
# 1. 加载模型
# ==========================================
print("\n[1] 加载模型...")
t0 = time.time()
with open(MODEL_PATH, 'rb') as f:
    ckpt = pickle.load(f)
model = neuralgcm.PressureLevelModel.from_checkpoint(ckpt)
model_with_physics = model.with_physics_core_output(enable=True)
print(f"    模型加载耗时: {time.time() - t0:.2f}s")

# ==========================================
# 2. 创建数据集
# ==========================================
print("\n[2] 创建数据集...")
t0 = time.time()
dataset = NeuralGCMDataset(
    data_dir=DATA_DIR,
    model_helper=model,
    prediction_steps=PREDICTION_STEPS,
    cache_size=10
)
print(f"    数据集索引耗时: {time.time() - t0:.2f}s")

# ==========================================
# 3. 测试单个样本的各阶段耗时
# ==========================================
print("\n[3] 单样本各阶段耗时分析...")
print("-" * 60)

# 3.1 文件加载
t0 = time.time()
sample = dataset[0]  # 首次加载（无缓存）
file_load_cold = time.time() - t0
print(f"    文件加载（冷启动）: {file_load_cold:.3f}s")

t0 = time.time()
sample = dataset[1]  # 第二次（部分缓存命中）
file_load_warm = time.time() - t0
print(f"    文件加载（部分缓存）: {file_load_warm:.3f}s")

t0 = time.time()
sample = dataset[0]  # 完全缓存命中
file_load_cached = time.time() - t0
print(f"    文件加载（完全缓存）: {file_load_cached:.3f}s")

# 3.2 模型编码
print("-" * 60)
inputs = sample['inputs']
input_forcings = sample['input_forcings']
future_forcings = sample['future_forcings']
rng_key = jax.random.key(42)

t0 = time.time()
initial_state = model.encode(inputs, input_forcings, rng_key)
encode_time_first = time.time() - t0
print(f"    model.encode（首次，含JIT编译）: {encode_time_first:.3f}s")

t0 = time.time()
initial_state = model.encode(inputs, input_forcings, rng_key)
encode_time = time.time() - t0
print(f"    model.encode（JIT编译后）: {encode_time:.3f}s")

# 3.3 模型推理
print("-" * 60)
timedelta = np.timedelta64(1, 'h')

t0 = time.time()
final_state, predictions = model_with_physics.unroll(
    initial_state,
    future_forcings,
    steps=PREDICTION_STEPS,
    timedelta=timedelta,
    start_with_input=True,
)
unroll_time_first = time.time() - t0
print(f"    model.unroll（首次，含JIT编译）: {unroll_time_first:.3f}s")

t0 = time.time()
final_state, predictions = model_with_physics.unroll(
    initial_state,
    future_forcings,
    steps=PREDICTION_STEPS,
    timedelta=timedelta,
    start_with_input=True,
)
unroll_time = time.time() - t0
print(f"    model.unroll（JIT编译后）: {unroll_time:.3f}s")

# 3.4 结果转换
print("-" * 60)
times = np.arange(PREDICTION_STEPS)

t0 = time.time()
predictions_ds, predictions_phy_ds = model_with_physics.data_to_xarray_with_physics(
    predictions, times=times
)
convert_time = time.time() - t0
print(f"    data_to_xarray_with_physics: {convert_time:.3f}s")

# ==========================================
# 4. 批量测试
# ==========================================
print("\n[4] 批量测试（5个样本）...")
print("-" * 60)

total_times = []
for i in range(NUM_SAMPLES):
    t0 = time.time()
    
    # 加载数据
    sample = dataset[i * 24]  # 每隔24小时取一个样本
    
    # 编码
    initial_state = model.encode(
        sample['inputs'], 
        sample['input_forcings'], 
        rng_key
    )
    
    # 推理
    final_state, predictions = model_with_physics.unroll(
        initial_state,
        sample['future_forcings'],
        steps=PREDICTION_STEPS,
        timedelta=timedelta,
        start_with_input=True,
    )
    
    # 转换
    predictions_ds, _ = model_with_physics.data_to_xarray_with_physics(
        predictions, times=times
    )
    
    elapsed = time.time() - t0
    total_times.append(elapsed)
    print(f"    样本 {i+1}: {elapsed:.2f}s")

print("-" * 60)
print(f"    平均耗时: {np.mean(total_times):.2f}s")
print(f"    最快: {np.min(total_times):.2f}s")
print(f"    最慢: {np.max(total_times):.2f}s")

# ==========================================
# 5. 性能建议
# ==========================================
print("\n" + "=" * 60)
print("性能瓶颈分析")
print("=" * 60)

print("\n各阶段耗时占比（JIT编译后）:")
total = file_load_cached + encode_time + unroll_time + convert_time
print(f"  - 文件加载: {file_load_cached:.3f}s ({file_load_cached/total*100:.1f}%)")
print(f"  - 模型编码: {encode_time:.3f}s ({encode_time/total*100:.1f}%)")
print(f"  - 模型推理: {unroll_time:.3f}s ({unroll_time/total*100:.1f}%)")
print(f"  - 结果转换: {convert_time:.3f}s ({convert_time/total*100:.1f}%)")

print("\n诊断结论:")
if file_load_cold > 5:
    print("  ⚠️ 文件加载较慢（可能是网络存储/HDD）")
    print("     建议: 使用 Kerchunk 或转换为 Zarr 格式")
    
if encode_time > 1:
    print("  ⚠️ 模型编码较慢")
    print("     建议: 检查 GPU 是否正常使用")

if unroll_time > 10:
    print("  ⚠️ 模型推理较慢")
    print("     建议: 检查 GPU 显存是否充足")

if convert_time > 2:
    print("  ⚠️ 结果转换较慢")
    print("     建议: 可以跳过此步骤，直接使用 predictions 计算 RMSE")

print("\n" + "=" * 60)

