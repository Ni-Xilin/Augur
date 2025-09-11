import torch
import numpy as np
import os
import pickle

def calculate_efficiency_metrics_deepcoffea(sessions_filepath):
    """
    加载为 DeepCOFFEA 保存的会话列表文件，并计算
    1. 应用延迟开销 (Applied Latency Overhead)
    2. 带宽开销 (Bandwidth Overhead)
    """


    if not os.path.exists(sessions_filepath):
        print(f"错误: 结果文件未找到: '{sessions_filepath}'")
        print("请确保 'setting' 变量设置正确，并且您已经运行了修改后的 test() 函数来生成该文件。")
        return

    print(f"正在从 '{sessions_filepath}' 加载会话数据...")

    # --- 2. 加载数据 ---
    try:
        with open(sessions_filepath, "rb") as fp:
            results = pickle.load(fp)
        
        adv_sessions_list = results["adv_sessions"]
        original_sessions_list = results["original_sessions"]
    except Exception as e:
        print(f"加载或解析 pickle 文件时出错: {e}")
        return
    
    print(f"加载完成。会话总数: {len(adv_sessions_list)}")

    # --- 3. 遍历列表，累加延迟和大小 ---
    all_non_zero_delays = []
    total_original_size = 0.0
    total_padding_added = 0.0

    for original_session, adv_session in zip(original_sessions_list, adv_sessions_list):
        # 提取时间特征 (IPD)
        original_time = original_session[0, :]
        adv_time = adv_session[0, :]
        
        # 计算并累加延迟
        delays = torch.clamp(adv_time - original_time, min=0)
        non_zero_delays_in_session = delays[delays > 1e-6].numpy()
        if len(non_zero_delays_in_session) > 0:
            all_non_zero_delays.extend(non_zero_delays_in_session)

        # 提取大小特征
        original_size = original_session[1, :]
        adv_size = adv_session[1, :]

        # 计算并累加大小和填充
        padding = torch.clamp(adv_size - original_size, min=0)
        total_original_size += torch.sum(torch.abs(original_size)).item()
        total_padding_added += torch.sum(padding).item()

    # --- 4. 统计分析和输出 ---
    all_non_zero_delays = np.array(all_non_zero_delays)
    
    # 计算带宽开销百分比
    bandwidth_overhead_percentage = (total_padding_added / total_original_size) * 100 if total_original_size > 0 else 0

    print("\n\n--- Augur 效率指标评估报告 (for DeepCOFFEA) ---")
    print("==================================================")
    print("### 1. 应用延迟开销 (Applied Latency Overhead)")
    print("该指标衡量了为实现防御而主动添加到数据包上的延迟统计。")
    print("--------------------------------------------------")
    if len(all_non_zero_delays) > 0:
        # DeepCOFFEA数据集中IPD单位是毫秒
        mean_delay = np.mean(all_non_zero_delays) 
        median_delay = np.median(all_non_zero_delays) 
        std_dev_delay = np.std(all_non_zero_delays)
        percentile_95_delay = np.percentile(all_non_zero_delays, 95)
        max_delay = np.max(all_non_zero_delays)
        
        print(f"分析的非零延迟点数量: {len(all_non_zero_delays):,}")
        print(f"平均延迟 (Mean):        {mean_delay:.4f} ms")
        print(f"延迟中位数 (Median):    {median_delay:.4f} ms")
        print(f"延迟标准差 (Std Dev):   {std_dev_delay:.4f} ms")
        print(f"95百分位延迟 (95th Pctl): {percentile_95_delay:.4f} ms")
        print(f"最大延迟 (Max):         {max_delay:.4f} ms")
    else:
        print("未发现任何有效的时间延迟扰动。")
    print("--------------------------------------------------")

    print("\n### 2. 带宽开销 (Bandwidth Overhead)")
    print("该指标衡量了为实现防御而添加的数据包填充（Padding）所占原始流量大小的百分比。")
    print("--------------------------------------------------")
    # DeepCOFFEA数据集中size单位是字节，所以结果单位是Bytes
    print(f"总原始流量大小 (Bytes): {total_original_size:,.2f}KB")
    print(f"总添加填充大小 (Bytes): {total_padding_added:,.2f}KB")
    print(f"带宽开销百分比:       {bandwidth_overhead_percentage:.4f} %")
    print("--------------------------------------------------")
    print("==================================================")

if __name__ == '__main__':
    # !!! --- 请修改这里 --- !!!
    # 将 'setting' 变量设置为您保存了 deepcoffea 扰动文件的目录名
    sessions_filepath = 'EfficiencyEvaluation/deepcoffea_sessions_for_overhead_calc.p'
    
    calculate_efficiency_metrics_deepcoffea(sessions_filepath)