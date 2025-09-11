import torch
import numpy as np
import os
import pickle

def calculate_efficiency_metrics(result_filename):
    """
    加载预生成的 DeepCorr 结果文件，计算并分析
    1. 应用延迟开销 (Applied Latency Overhead)
    2. 带宽开销 (Bandwidth Overhead)
    """


    # 检查文件是否存在
    if not os.path.exists(result_filename):
        print(f"错误: 结果文件未找到: '{result_filename}'")
        print("请确保 'result_filename' 变量设置正确，并且文件确实存在于该路径。")
        return

    print(f"正在从 '{result_filename}' 加载结果文件...")

    # --- 2. 加载数据 ---
    try:
        with open(result_filename, "rb") as fp:
            results = pickle.load(fp)
        
        adv_samples = results["adv_samples"]
        original_samples = results["original_samples"]
    except Exception as e:
        print(f"加载或解析 pickle 文件时出错: {e}")
        return
    
    print(f"加载完成。样本总数: {adv_samples.shape[0]}")

    # === 延迟开销计算 ===
    print("\n--- 正在计算应用延迟开销 (Applied Latency Overhead) ---")
    # 根据您的代码，时间特征位于第0和第3行 (IPD)
    time_feature_rows = [0, 3] 
    original_time_features = original_samples[:, 0, time_feature_rows, :]
    adv_time_features = adv_samples[:, 0, time_feature_rows, :]
    applied_delays_tensor = torch.clamp(adv_time_features - original_time_features, min=0)
    
    all_delays = applied_delays_tensor.flatten().cpu().numpy()
    non_zero_delays = all_delays[all_delays > 1e-6]

    # === 带宽开销计算 ===
    print("--- 正在计算带宽开销 (Bandwidth Overhead) ---")
    # 根据您的代码，大小特征位于第4和第7行
    size_feature_rows = [4, 7]
    original_size_features = original_samples[:, 0, size_feature_rows, :]
    adv_size_features = adv_samples[:, 0, size_feature_rows, :]
    applied_padding_tensor = torch.clamp(adv_size_features - original_size_features, min=0)

    # 计算总原始大小和总填充大小 (L1范数，代表实际字节数)
    # 注意：这里的单位是经过 /1000.0 处理后的，但不影响百分比计算
    total_original_size = torch.sum(torch.abs(original_size_features)).item()
    total_padding_added = torch.sum(applied_padding_tensor).item()
    
    # 计算百分比
    bandwidth_overhead_percentage = (total_padding_added / total_original_size) * 100 if total_original_size > 0 else 0

    # --- 结果输出 ---
    
    # 输出延迟开销
    print("\n\n--- Augur 效率指标评估报告(deepcorr) ---")
    print("==================================================")
    print("### 1. 应用延迟开销 (Applied Latency Overhead)")
    print("该指标衡量了为实现防御而主动添加到数据包上的延迟统计。")
    print("--------------------------------------------------")
    if len(non_zero_delays) > 0:
        # 单位已经是毫秒(ms)，因为数据集中乘以了1000
        mean_delay = np.mean(non_zero_delays)
        median_delay = np.median(non_zero_delays)
        std_dev_delay = np.std(non_zero_delays)
        percentile_95_delay = np.percentile(non_zero_delays, 95)
        max_delay = np.max(non_zero_delays)
        
        print(f"分析的非零延迟点数量: {len(non_zero_delays):,}")
        print(f"平均延迟 (Mean):        {mean_delay:.4f} ms")
        print(f"延迟中位数 (Median):    {median_delay:.4f} ms")
        print(f"延迟标准差 (Std Dev):   {std_dev_delay:.4f} ms")
        print(f"95百分位延迟 (95th Pctl): {percentile_95_delay:.4f} ms")
        print(f"最大延迟 (Max):         {max_delay:.4f} ms")
    else:
        print("未发现任何有效的时间延迟扰动。")
    print("--------------------------------------------------")

    # 输出带宽开销
    print("\n### 2. 带宽开销 (Bandwidth Overhead)")
    print("该指标衡量了为实现防御而添加的数据包填充（Padding）所占原始流量大小的百分比。")
    print("--------------------------------------------------")
    print(f"总原始流量大小 (Sum of L1 norms): {total_original_size:,.2f} (scaled units)")
    print(f"总添加填充大小 (Sum of L1 norms): {total_padding_added:,.2f} (scaled units)")
    print(f"带宽开销百分比:                 {bandwidth_overhead_percentage:.4f} %")
    print("--------------------------------------------------")
    print("==================================================")


if __name__ == '__main__':
    # !!! --- 请修改这里 --- !!!
    # 将下面的变量设置为您通过 test() 函数生成的 .p 文件的确切名称
    result_filename = 'target_model/deepcorr/deepcorr300/Gbase/Gdeepcorr_advsamples_time0.1343_size0.0089.p'
    
    calculate_efficiency_metrics(result_filename)