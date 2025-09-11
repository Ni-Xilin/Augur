import sys
import os
import torch
import time
import psutil
import numpy as np
from types import SimpleNamespace # 使用标准库替代 dotdict
from generator.generator import TSTGenerator 

def measure_cpu_overhead(args):
    """
    在CPU上，以batch_size=1的模式，测量Augur生成器的
    1. 单次扰动生成延迟
    2. 实时计算足迹
    """
    
    device = torch.device("cpu")
    process = psutil.Process(os.getpid())

    # --- 1. 加载模型到CPU ---
    print("--- 正在加载预训练的生成器模型到 CPU ---")
    generator = TSTGenerator(seq_len=args.seq_len, 
                             patch_len=args.patch_len,
                             pred_len=args.pred_len,
                             feat_dim=args.enc_in, 
                             depth=args.depth, 
                             scale_factor=args.scale_factor, 
                             n_layers=args.n_layers, 
                             d_model=args.d_model, 
                             n_heads=args.n_heads,
                             individual=args.individual, 
                             d_k=None, d_v=None, 
                             d_ff=args.d_ff, 
                             norm='BatchNorm', 
                             attn_dropout=args.att_dropout, 
                             head_dropout=args.head_dropout, 
                             act=args.activation,pe='zeros', 
                             learn_pe=True,pre_norm=False, 
                             res_attention=False, 
                             store_attn=False).to(device)
    

    checkpoint = torch.load(args.checkpoints, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    print("模型加载完成。")

    # --- 2. 测量单次生成延迟 ---
    print(f"\n--- 正在测量单次扰动生成 ({args.seq_len} -> {args.pred_len}) 的延迟 ---")
    num_iterations = 500  # 进行多次测量以获得稳定的平均值
    latencies = []
    
    # "热身"运行，确保所有库和模型都已完全加载
    with torch.no_grad():
        dummy_input = torch.randn(1, args.enc_in, args.seq_len).to(device)
        _ = generator(dummy_input)

    # 开始精确计时
    for _ in range(num_iterations):
        z = torch.randn(1, args.enc_in, args.seq_len).to(device)
        start_time = time.perf_counter()
        _ = generator(z)
        end_time = time.perf_counter()
        latencies.append((end_time - start_time) * 1000) # 转换为毫秒

    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)

    # --- 3. 测量实时计算足迹 (CPU & RAM) ---
    print(f"\n--- 正在测量 {args.footprint_duration_sec} 秒内的计算足迹 (CPU & RAM) ---")
    
    cpu_usage_list = []
    
    start_time_footprint = time.time()
    initial_ram_mb = process.memory_info().rss / (1024 * 1024)
    
    with torch.no_grad():
        while time.time() - start_time_footprint < args.footprint_duration_sec:
            # 记录CPU使用率
            cpu_usage_list.append(process.cpu_percent(interval=0.1))
            # 持续生成数据以模拟负载
            z = torch.randn(1, args.enc_in, args.seq_len).to(device)
            _ = generator(z)

    # 获取最终的资源占用数据
    avg_cpu_percent = np.mean(cpu_usage_list)
    peak_ram_mb = process.memory_info().rss / (1024 * 1024)
    
    # --- 4. 打印最终报告 ---
    print("\n\n--- Augur(deepcorr) CPU 计算开销评估报告 ---")
    print("==================================================")
    print(f"### 1. 单次扰动生成延迟 (Perturbation Generation Latency, {args.seq_len} -> {args.pred_len})")
    print("--------------------------------------------------")
    print(f"平均生成延迟 (Mean):        {avg_latency:.4f} ms")
    print(f"95百分位生成延迟 (95th Pctl): {p95_latency:.4f} ms")
    print("--------------------------------------------------")

    print("\n### 2. 实时计算足迹 (Real-time Computational Footprint)")
    print("--------------------------------------------------")
    print(f"平均CPU使用率 (Average CPU Usage): {avg_cpu_percent:.2f} % (per core)")
    print(f"峰值内存占用 (Peak RAM Usage):   {peak_ram_mb:.2f} MB")
    print("--------------------------------------------------")
    print("==================================================")

if __name__ == '__main__':
    # !!! --- 请根据您要测试的模型进行修改 --- !!!
    args = SimpleNamespace()

    # --- 测试控制参数 ---
    args.footprint_duration_sec = 10 # 测量资源足迹的持续时间（秒）

    # --- 模型架构参数 (必须与您加载的模型完全一致) ---
    # 示例配置 for DeepCorr
    args.seq_len = 96
    args.pred_len = 48
    args.patch_len = 4
    args.stride = 48
    args.enc_in = 4 # time_in, time_out, size_in, size_out
    args.d_model = 512
    args.n_heads = 8
    args.depth = 3
    args.scale_factor = 2
    args.n_layers = 2
    args.d_ff = 1024
    args.individual = True
    args.att_dropout = 0.0
    args.head_dropout = 0.0
    args.activation = 'gelu'

    # --- 检查点路径 (最重要的参数) ---
    # 请确保这里的路径指向您希望测试的 .pth 文件
    args.checkpoints = 'checkpoints/deepcorr300/deepcorr300_PatchTST_Deepcorr300_sl96_pl48_dm512_nh8/generator_checkpoint_9_acc_0.878_Gacc0.0423_advipd0.137_advsize0.009.pth'

    measure_cpu_overhead(args)