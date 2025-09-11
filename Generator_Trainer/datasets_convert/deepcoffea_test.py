# -*- coding: utf-8 -*-
# 导入所需的库
import torch
import torch.nn as nn
import numpy as np
import pickle
import pathlib
from tqdm import tqdm
import argparse
import os
from sklearn.metrics.pairwise import cosine_similarity

# 确保可以从您的项目中正确导入模型
# from target_model import Deepcoffea
# 为了代码的独立性，这里我们假设Deepcoffea.Model已经定义好了
# 在您的实际环境中，请确保下面的导入是有效的
try:
    from target_model import Deepcoffea
except ImportError:
    print("警告: 无法从 'target_model' 导入 'Deepcoffea'。将使用一个占位模型。")
    # 定义一个占位模型以保证代码可以运行
    class Deepcoffea:
        class Model(nn.Module):
            def __init__(self, emb_size, input_size):
                super().__init__()
                self.fc = nn.Linear(input_size, emb_size)
            def forward(self, x):
                return self.fc(x)


def partition_sessions_into_windows(sessions, delta, win_size, n_wins, tor_len, device):
    """
    将完整的会话流量数据分割成DeepCoffea模型所需的窗口格式。
    
    参数:
    - sessions (list of torch.Tensor): 包含多个会话的列表，每个会话形状为 (2, session_length)。
    - ... (其他参数与DeepCoffea设置相同)
    
    返回:
    - torch.Tensor: 形状为 [num_sessions, n_wins, tor_len * 2] 的窗口化数据。
    """
    win_size_ms = win_size * 1000
    delta_ms = delta * 1000
    offset = win_size_ms - delta_ms

    all_sessions_windows = []

    for session in sessions:
        session_ipd = session[0, :]
        session_size = session[1, :]
        cumulative_time = session_ipd.abs().cumsum(dim=0)
        
        single_session_windows = []
        for wi in range(int(n_wins)):
            start_time = wi * offset
            end_time = start_time + win_size_ms

            start_idx = torch.searchsorted(cumulative_time, start_time).item()
            end_idx = torch.searchsorted(cumulative_time, end_time).item()

            window_ipd = session_ipd[start_idx:end_idx]
            window_size = session_size[start_idx:end_idx]

            if len(window_ipd) > 0:
                window_ipd = torch.cat([torch.tensor([0.0]).to(device), window_ipd[1:]])

            window = torch.cat([window_ipd, window_size])

            if window.shape[0] < tor_len * 2:
                padding = torch.zeros(tor_len * 2 - window.shape[0], device=device)
                window = torch.cat([window, padding])

            window = window[:tor_len * 2]
            single_session_windows.append(window)
        
        all_sessions_windows.append(torch.stack(single_session_windows))

    return torch.stack(all_sessions_windows)


def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 1. 加载预训练的DeepCoffea模型
    print("正在加载预训练的DeepCoffea模型...")
    state_dict = torch.load(args.model_path, map_location=device)
    anchor_model = Deepcoffea.Model(emb_size=args.emb_size, input_size=args.tor_len * 2).to(device)
    pandn_model = Deepcoffea.Model(emb_size=args.emb_size, input_size=args.exit_len * 2).to(device)
    anchor_model.load_state_dict(state_dict['anchor_state_dict'])
    pandn_model.load_state_dict(state_dict['pandn_state_dict'])
    anchor_model.eval()
    pandn_model.eval()
    print("模型加载成功。")

    # 2. 加载数据
    print(f"正在加载扰动后的Tor流量数据从: {args.reverted_data_path}")
    reverted_data = np.load(args.reverted_data_path, allow_pickle=True)
    tor_ipds_list = reverted_data['tor_ipds']
    tor_sizes_list = reverted_data['tor_sizes']

    print(f"正在加载原始Exit窗口数据从: {args.original_exit_data_path}")
    original_exit_data = np.load(args.original_exit_data_path, allow_pickle=True)
    # 形状转换: (n_wins, n_samples, win_len) -> (n_samples, n_wins, win_len)
    original_exit_windows = np.transpose(original_exit_data['test_exit'], (1, 0, 2))
    
    # --- 核心修改：将样本数量限制为前 N 条 ---
    num_samples_to_test = min(len(tor_ipds_list), args.num_test_samples)
    print(f"将测试样本数量限制为前 {num_samples_to_test} 条。")
    tor_ipds_list = tor_ipds_list[:num_samples_to_test]
    tor_sizes_list = tor_sizes_list[:num_samples_to_test]
    original_exit_windows = original_exit_windows[:num_samples_to_test]
    
    num_samples = len(tor_ipds_list)
    if num_samples != original_exit_windows.shape[0]:
        print(f"警告：截取后，扰动Tor样本数 ({num_samples}) 与原始Exit样本数 ({original_exit_windows.shape[0]}) 仍不匹配。")
        num_samples = min(num_samples, original_exit_windows.shape[0])
        print(f"将使用 {num_samples} 个样本进行评估。")

    # 3. 处理并扰动Tor流量
    perturbed_tor_sessions = []
    for i in range(num_samples):
        ipds = torch.tensor(tor_ipds_list[i], dtype=torch.float32, device=device)
        sizes = torch.tensor(tor_sizes_list[i], dtype=torch.float32, device=device)
        perturbed_tor_sessions.append(torch.stack([ipds, sizes]))
        
    print("正在将扰动后的Tor会话分割成窗口...")
    perturbed_tor_windows = partition_sessions_into_windows(
        perturbed_tor_sessions,
        args.delta, args.win_size, args.n_wins, args.tor_len, device
    ) # 形状: (n_samples, n_wins, win_len)

    # 4. 准备模型输入并进行推理
    Gtor_embs = []
    exit_embs = []
    
    # 调整数据形状以匹配模型输入
    # (n_samples, n_wins, win_len) -> (n_wins, n_samples, win_len) -> (n_wins*n_samples, win_len)
    Gxa_batch = torch.permute(perturbed_tor_windows, (1, 0, 2)).reshape(-1, args.tor_len * 2).float()
    xp_batch = torch.from_numpy(original_exit_windows[:num_samples]).permute(1, 0, 2).reshape(-1, args.exit_len * 2).float().to(device)

    print("正在通过模型进行推理...")
    with torch.no_grad():
        for i in tqdm(range(0, Gxa_batch.shape[0], args.batch_size), desc="推理进度"):
            gxa_sub_batch = Gxa_batch[i:i+args.batch_size]
            xp_sub_batch = xp_batch[i:i+args.batch_size]
            
            Ga_out = anchor_model(gxa_sub_batch)
            p_out = pandn_model(xp_sub_batch)
            
            Gtor_embs.append(Ga_out.cpu().numpy())
            exit_embs.append(p_out.cpu().numpy())

    Gtor_embs = np.concatenate(Gtor_embs)
    exit_embs = np.concatenate(exit_embs)
    
    # 5. 计算相关性矩阵并保存
    print("正在计算相关性矩阵...")
    Gcorr_matrix = cosine_similarity(Gtor_embs, exit_embs)
    Gsim = np.mean(np.diag(Gcorr_matrix))
    
    output_filename = f"Gcorr_matrix_reverted_sim_{Gsim:.4f}_samples{num_samples}.npz"
    output_path = pathlib.Path(args.output_dir) / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(output_path, corr_matrix=Gcorr_matrix, sim=Gsim)
    
    print("-" * 30)
    print("评估完成！")
    print(f"在扰动后、转换回的前 {num_samples} 条数据集上的平均对角线相似度为: {Gsim:.4f}")
    print(f"相关性矩阵已保存至: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='评估从DeepCorr扰动并转换回的DeepCoffea数据的性能')

    # --- 路径参数 ---
    parser.add_argument('--model_path', type=str, 
                        default='target_model/deepcoffea/deepcoffea_d3_ws5_nw11_thr20_tl500_el800_nt1000_ap1e-01_es64_lr1e-03_mep100000_bs256/best_loss.pth',
                        help='预训练的DeepCoffea模型权重路径')
    parser.add_argument('--reverted_data_path', type=str, 
                        default='datasets_convert/reverted_from_deepcorr_test_session_time0.1085_size0.0097.npz',
                        help='被扰动并转换回DeepCoffea格式的数据文件路径 (.npz)')
    parser.add_argument('--original_exit_data_path', type=str, 
                        default='target_model/deepcoffea/dataset/CrawlE_Proc/filtered_and_partitioned/d3_ws5_nw11_thr20_tl500_el800_nt1000_test.npz',
                        help='原始未扰动的DeepCoffea测试集窗口文件路径 (.npz)，用于获取exit流量')
    parser.add_argument('--output_dir', type=str,
                        default='datasets_convert/deepcoffea_d3_ws5_nw11_thr20_tl500_el800_nt1000_ap1e-01_es64_lr1e-03_mep100000_bs256/deepcoffea_PatchTST_Deepcoffea_sl150_pl70_dm512_nh8_pal10_s70',
                        help='保存输出的相关性矩阵的目录')
    
    # --- 模型和数据参数 (必须与训练和数据生成时使用的参数一致) ---
    parser.add_argument('--delta', type=int, default=3, help='窗口重叠时间（秒）')
    parser.add_argument('--win_size', type=int, default=5, help='每个窗口的大小（秒）')
    parser.add_argument('--n_wins', type=int, default=11, help='每个会话的窗口数量')
    parser.add_argument('--tor_len', type=int, default=500, help='Tor窗口的目标长度')
    parser.add_argument('--exit_len', type=int, default=800, help='Exit窗口的目标长度')
    parser.add_argument('--emb_size', type=int, default=64, help='嵌入向量大小')
    
    # --- 运行参数 ---
    parser.add_argument('--gpu', type=int, default=0, help='使用的GPU ID')
    parser.add_argument('--batch_size', type=int, default=256, help='推理时的批处理大小')
    parser.add_argument('--num_test_samples', type=int, default=500, help='用于评估的样本数量（默认为500）')
    
    args = parser.parse_args()
    main(args)
