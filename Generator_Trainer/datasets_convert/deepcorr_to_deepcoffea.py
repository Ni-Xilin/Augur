# -*- coding: utf-8 -*-
# 注意，因为drop_last=True，因此转换后的数据集长度会比原始数据集略短
import numpy as np
import pickle
import pathlib
from tqdm import tqdm
import torch
import argparse

def convert_deepcorr_to_deepcoffea(perturbed_pickle_path, metadata_path, output_npz_path, flow_size):
    """
    使用元数据，将扰动后的DeepCorr格式数据转换回DeepCoffea的_session.npz格式。
    此版本已修正，可以正确处理因flow_size限制导致的截断和补零情况，并计算扰动率。

    参数:
    - perturbed_pickle_path (str): 扰动后的 .p 文件路径，包含 "adv_samples" 键。
    - metadata_path (str): 包含原始顺序和时间戳的元数据 .pkl 文件路径。
    - output_npz_path (str): 输出的DeepCoffea格式 .npz 文件路径。
    - flow_size (int): 在扰动阶段使用的流量长度。
    """
    try:
        print(f"正在从 {perturbed_pickle_path} 加载扰动后的DeepCorr数据...")
        with open(perturbed_pickle_path, 'rb') as f:
            results_dict = pickle.load(f)
            perturbed_samples_tensor = results_dict["adv_samples"]
        
        print(f"正在从 {metadata_path} 加载元数据...")
        with open(metadata_path, 'rb') as f:
            metadata_list = pickle.load(f)
            
        # 确保元数据列表与扰动样本数量一致
        if len(metadata_list) > len(perturbed_samples_tensor):
            print(f"注意: 由于扰动器设置了drop_last=True, 元数据样本数({len(metadata_list)})多于扰动样本数({len(perturbed_samples_tensor)})。")
            print("将截断元数据以进行匹配。")
            metadata_list = metadata_list[:len(perturbed_samples_tensor)]

        print(f"加载了 {len(perturbed_samples_tensor)} 条样本及其元数据。")

    except FileNotFoundError as e:
        print(f"错误：找不到文件: {e.filename}")
        return
    except KeyError:
        print(f"错误: 输入的pickle文件 {perturbed_pickle_path} 中不包含 'adv_samples' 键。")
        return
    except Exception as e:
        print(f"加载文件时发生错误: {e}")
        return

    final_tor_ipds, final_tor_sizes = [], []
    final_exit_ipds, final_exit_sizes = [], []
    
    # 新增：用于存储每条样本的扰动率
    all_time_rates = []
    all_size_rates = []

    print("开始反向转换并计算扰动率...")
    for i in tqdm(range(len(perturbed_samples_tensor)), desc="反向转换进度"):
        p_sample_tensor = perturbed_samples_tensor[i].squeeze(0) # Shape: (8, flow_size)
        m_sample = metadata_list[i]
        
        def reconstruct_flow(perturbed_channels, original_metadata_sizes, is_here_flow):
            if is_here_flow:
                inc_ipd_ch, out_ipd_ch = 0, 3
                inc_size_ch, out_size_ch = 4, 7
            else:
                inc_ipd_ch, out_ipd_ch = 2, 1
                inc_size_ch, out_size_ch = 6, 5

            inc_ipd = (perturbed_channels[inc_ipd_ch, :] / 1000.0).tolist()
            out_ipd = (perturbed_channels[out_ipd_ch, :] / 1000.0).tolist()
            inc_size = (perturbed_channels[inc_size_ch, :] * 1000.0).tolist()
            out_size = (perturbed_channels[out_size_ch, :] * 1000.0).tolist()

            packet_info = []
            num_original_packets = len(original_metadata_sizes)
            effective_num_packets = min(num_original_packets, flow_size)
            original_sizes_to_process = original_metadata_sizes[:effective_num_packets]

            inc_ipd_idx, out_ipd_idx = 0, 0
            inc_size_idx, out_size_idx = 0, 0

            for original_size_val in original_sizes_to_process:
                if original_size_val < 0:
                    if inc_ipd_idx < len(inc_ipd) and inc_size_idx < len(inc_size):
                        packet_info.append({'ipd': inc_ipd[inc_ipd_idx], 'size': -inc_size[inc_size_idx]})
                        inc_ipd_idx += 1
                        inc_size_idx += 1
                else:
                    if out_ipd_idx < len(out_ipd) and out_size_idx < len(out_size):
                        packet_info.append({'ipd': out_ipd[out_ipd_idx], 'size': out_size[out_size_idx]})
                        out_ipd_idx += 1
                        out_size_idx += 1
            
            ipds_final = np.array([p['ipd'] for p in packet_info]) * 1000.0
            sizes_final = np.array([p['size'] for p in packet_info]) / 1000.0
            ipds_final_signed = np.sign(sizes_final) * np.abs(ipds_final)
            return ipds_final_signed, sizes_final

        # 重建 tor ('here') 流量
        tor_ipds_signed, tor_sizes = reconstruct_flow(p_sample_tensor, m_sample['tor_sizes'], is_here_flow=True)
        final_tor_ipds.append(tor_ipds_signed)
        final_tor_sizes.append(tor_sizes)

        # 重建 exit ('there') 流量
        exit_ipds_signed, exit_sizes = reconstruct_flow(p_sample_tensor, m_sample['exit_sizes'], is_here_flow=False)
        final_exit_ipds.append(exit_ipds_signed)
        final_exit_sizes.append(exit_sizes)
        
        # --- 新增：计算扰动率 ---
        # 只计算 tor (here) 流量的扰动率，因为这是被攻击的部分
        effective_len = len(tor_ipds_signed)
        if effective_len > 0:
            # 获取有效长度的原始数据
            original_ipd = m_sample['tor_ipds'][:effective_len]
            # *** 核心修正 ***
            # 原始元数据中的size单位是bytes，需要转换为KB以匹配还原后的adv_size_tensor
            original_size_in_bytes = m_sample['tor_sizes'][:effective_len]
            original_size_in_kb = original_size_in_bytes / 1000.0

            # 转换为Tensor进行计算
            adv_ipd_tensor = torch.from_numpy(tor_ipds_signed)       # 单位: ms
            adv_size_tensor = torch.from_numpy(tor_sizes)            # 单位: KB
            original_ipd_tensor = torch.from_numpy(original_ipd)     # 单位: ms
            original_size_tensor = torch.from_numpy(original_size_in_kb) # 单位: KB (已修正)

            # 计算L2范数和比率
            ipd_l2_distance = torch.linalg.norm(adv_ipd_tensor.float() - original_ipd_tensor.float(), ord=2)
            size_l2_distance = torch.linalg.norm(adv_size_tensor.float() - original_size_tensor.float(), ord=2)
            
            original_ipd_l2 = torch.linalg.norm(original_ipd_tensor.float(), ord=2)
            original_size_l2 = torch.linalg.norm(original_size_tensor.float(), ord=2)
            
            time_rate = (ipd_l2_distance / (original_ipd_l2 + 1e-9)).item()
            size_rate = (size_l2_distance / (original_size_l2 + 1e-9)).item()
            
            all_time_rates.append(time_rate)
            all_size_rates.append(size_rate)

    # 计算平均扰动率
    mean_time_rate = np.mean(all_time_rates) if all_time_rates else 0
    mean_size_rate = np.mean(all_size_rates) if all_size_rates else 0

    # 4. 保存为npz文件
    output_path = pathlib.Path(f'{output_npz_path}_time{mean_time_rate:.4f}_size{mean_size_rate:.4f}.npz')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 新增：将扰动率一并保存
    np.savez_compressed(
        output_path,
        tor_ipds=np.array(final_tor_ipds, dtype=object),
        tor_sizes=np.array(final_tor_sizes, dtype=object),
        exit_ipds=np.array(final_exit_ipds, dtype=object),
        exit_sizes=np.array(final_exit_sizes, dtype=object),
        mean_time_rate=mean_time_rate,
        mean_size_rate=mean_size_rate
    )
    print("-" * 30)
    print(f"反向转换完成！")
    print(f"平均时间扰动率 (L2 Ratio): {mean_time_rate:.4f}")
    print(f"平均大小扰动率 (L2 Ratio): {mean_size_rate:.4f}")
    print(f"已将 {len(final_tor_ipds)} 条样本及扰动率保存为DeepCoffea格式: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="将扰动后的DeepCorr数据转换回DeepCoffea格式，并计算扰动率")
    
    parser.add_argument('--perturbed_path', type=str, 
                        default='datasets_convert/Gconverted_flow5000_advsamples.p', 
                        help='扰动后的DeepCorr数据文件路径 (.p)')
    parser.add_argument('--metadata_path', type=str, 
                        default='datasets_convert/conversion_metadata.pkl', 
                        help='转换时生成的元数据文件路径 (.pkl)')
    parser.add_argument('--output_path', type=str, 
                        default='datasets_convert/reverted_from_deepcorr_test_session', 
                        help='最终输出的DeepCoffea格式文件路径 (.npz)')
    parser.add_argument('--flow_size', type=int, default=5000, 
                        help='扰动时使用的flow_size，用于处理截断')

    args = parser.parse_args()
    
    convert_deepcorr_to_deepcoffea(args.perturbed_path, args.metadata_path, args.output_path, args.flow_size)
