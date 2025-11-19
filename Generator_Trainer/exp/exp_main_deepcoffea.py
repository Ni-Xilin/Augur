from data_provider.data_factory import data_provider
from exp.exp_basic_deepcoffea import Exp_Basic
from target_model import Deepcoffea
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from generator.generator import TSTGenerator
from utils.metrics import metric
import numpy as np
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import pathlib
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler 
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from torchviz import make_dot
import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')


class Loss(nn.Module):
    def __init__(self,beta=4,alpha = 1.5, gamma=0.1):
        super(Loss, self).__init__()
        self.beta = beta 
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self,time_rate, size_rate, Ga_out, p_out):

        criterion = torch.nn.CosineEmbeddingLoss(margin=-0.5, reduction='mean')
        cosine_loss = criterion(Ga_out, p_out, torch.full((Ga_out.size(0),), -1.0).to(Ga_out.device))
        
        loss=self.beta*cosine_loss+self.alpha*time_rate+self.gamma*size_rate
        # loss = self.gamma*cosine_loss
        return loss, cosine_loss,self.beta,self.alpha,self.gamma

class DeepCoffeaDataset(Dataset):
    '''
        返回batch_size个会话tor端的正常（待扰动）流量,exit端窗口流量：self.win_tor[:,idx,:],self.win_exit[:,idx,:],包括ipd和size
    '''

    def __init__(self, win_data,train=True):
        
        
        if train:
            mode = "train"
        else:
            mode = "test"
        '''
            v_tor:[n_wins, n_train_samples, tor_len * 2]
            v_exit:[n_wins, n_train_samples, exit_len * 2]  
            v_label:[n_wins, n_train_samples, 1]
            
        '''
        self.win_tor =  win_data[f"{mode}_tor"]
        self.win_exit=  win_data[f"{mode}_exit"]
        
        # session_tor_ipd = session_data['tor_ipd']
        # session_tor_size = session_data['tor_size']
        # session_tor_len = session_data['tor_lengths']#[n_train_samples,n_wins]
        # session_label = session_data['labels']
        
        print(f"Number of (session)examples to {mode}: {self.win_exit.shape[1]}, number of windows: {self.win_exit.shape[0]}")

    def __len__(self):
        return self.win_exit.shape[1]

    def __getitem__(self, idx):
        #返回正常流量
        return idx,self.win_tor[:,idx,:],self.win_exit[:,idx,:]
    
    
@torch.no_grad()
def inference(anchor, pandn, loader, dev):
    loader.sampler.train = False
    anchor.eval()
    pandn.eval()

    tor_embs = []
    exit_embs = []
    for _, (xa_batch, xp_batch) in enumerate(loader):
        xa_batch, xp_batch = xa_batch.to(dev), xp_batch.to(dev)

        a_out = anchor(xa_batch)
        p_out = pandn(xp_batch)

        tor_embs.append(a_out.cpu().numpy())
        exit_embs.append(p_out.cpu().numpy())

    tor_embs = np.concatenate(tor_embs)     # (N, emb_size)
    exit_embs = np.concatenate(exit_embs)   # (N, emb_size)
    print(f"Inference {len(loader.dataset)} pairs done.")
    return tor_embs, exit_embs

def deeocoffea_eval(corr_matrix, n_wins, vote_thr,threshold):
    n_test = corr_matrix.shape[0] // n_wins
    votes = np.zeros((n_test, n_test), dtype=np.int64)
    for wi in range(0, n_wins):
            corr_matrix_win = corr_matrix[n_test*wi:n_test*(wi + 1), n_test*wi:n_test*(wi + 1)]

            for i in range(n_test):
                for j in range(n_test):
                    if corr_matrix_win[i,j] >= threshold:
                        votes[i,j] += 1
    tp, fp, tn, fn = 0, 0, 0, 0
    for i in range(n_test):
        for j in range(n_test):
            if votes[i,j] >= vote_thr and i == j:
                tp += 1
            elif votes[i,j] >= vote_thr and i != j:
                fp += 1
            elif votes[i,j] < vote_thr and i == j:
                fn += 1
            else:   # votes[i,j] < vote_thr and i != j
                tn += 1

    if tp + fn == 0:
        tprs = 0.0
    else:
        tprs = tp / (tp + fn)
    
    if fp + tn == 0:
        fprs = 0.0
    else:
        fprs= fp / (fp + tn)

    if tp + fp == 0:
        prs = 0.0
    else:
        prs = tp / (tp + fp)

    return tprs, fprs, prs

# 放置在脚本的开头，模型定义之前
class DifferentiableSearchsorted(torch.autograd.Function):
    """
    一个 torch.searchsorted 的可微版本。
    它在反向传播时使用线性插值作为代理梯度，将梯度传递给分箱的边界。
    """
    @staticmethod
    def forward(ctx, sorted_sequence, values):
        """
        前向传播：执行 searchsorted 并保存必要信息。
        """
        # 执行标准的searchsorted
        indices = torch.searchsorted(sorted_sequence, values, right=True)
        
        # 将张量鉗制在有效索引范围内，防止越界
        indices.clamp_(0, sorted_sequence.size(-1) - 1)

        # 为反向传播保存输入和输出
        ctx.save_for_backward(sorted_sequence, values, indices)
        return indices

    @staticmethod
    def backward(ctx, grad_output):
        """
        反向传播：使用线性插值计算代理梯度。
        """
        sorted_sequence, values, indices = ctx.saved_tensors
        
        # 我们只关心 sorted_sequence 的梯度，因为 values (时间点) 是固定的
        grad_sorted_sequence = torch.zeros_like(sorted_sequence)
        
        # 获取每个值左右两边的边界点
        # 左边界索引
        idx_left = (indices - 1).clamp(0, sorted_sequence.size(-1) - 1)
        # 右边界索引
        idx_right = indices.clamp(0, sorted_sequence.size(-1) - 1)

        # 获取边界值
        val_left = sorted_sequence.gather(-1, idx_left)
        val_right = sorted_sequence.gather(-1, idx_right)
        
        # 计算插值权重 (alpha)
        # 加上一个极小值避免除以零
        weight_right = (values - val_left) / (val_right - val_left + 1e-9)
        weight_left = 1 - weight_right
        
        # 将下游梯度按权重分配给左右两个边界点
        # 使用.scatter_add_()可以处理多个值映射到同一个索引的情况
        grad_sorted_sequence.scatter_add_(-1, idx_left, grad_output * weight_left)
        grad_sorted_sequence.scatter_add_(-1, idx_right, grad_output * weight_right)
        
        # values (start_time, end_time) 是固定的，没有梯度
        grad_values = None
        
        return grad_sorted_sequence, grad_values

def split_time_series(data, seq_len, pred_len, stride,device):
    """
    将时间序列数据分划为序列批次和预测批次，并返回 torch.Tensor 类型的数据。

    参数:
        data (torch.Tensor): 输入的时间序列数据，形状为 (feature, tor_len)。
        seq_len (int): 序列批次的长度。
        pred_len (int): 预测批次的长度。
        stride (int): 每次移动的步长。

    返回:
        seq_batches (torch.Tensor): 形状为 (num_batches, feature, seq_len) 的序列批次。
        pred_batches (torch.Tensor): 形状为 (num_batches, feature, pred_len) 的预测批次。
    """
    num_features, total_len = data.shape
    num_batches = (total_len - seq_len - pred_len ) // stride + 1 
    seq_batches = []
    pred_batches = []
    
    for i in range(num_batches+1):
        start_idx = i * stride
        end_idx = start_idx + seq_len
        seq_batch = data[:, start_idx:end_idx]
        
        if end_idx + pred_len > total_len:
            pred_batch = data[:, end_idx:total_len]
        else:
            pred_batch = data[:, end_idx:end_idx + pred_len]
        
        # 如果预测批次的长度不够，用0填充
        if pred_batch.shape[1] < pred_len:
            pred_batch = torch.cat([pred_batch, torch.zeros(num_features, pred_len - pred_batch.shape[1]).to(device)], dim=1)
        
        seq_batches.append(seq_batch)
        pred_batches.append(pred_batch)
    
    seq_batches = torch.stack(seq_batches)
    pred_batches = torch.stack(pred_batches)
    
    return seq_batches, pred_batches

def reconstruct_adv_preds(adv_preds, original_sessions, seq_len, pred_len, stride, session_lengths):
    """
    将 adv_preds 还原为原始的时间序列数据，考虑到每段会话长度不一致，并处理填充的0。

    参数:
        adv_preds (torch.Tensor): 形状为 (num_batches, feature, pred_len) 的扰动后的预测批次。
        original_sessions (list of torch.Tensor): 原始的每段会话数据，每个会话的形状为 (feature, session_length)。
        seq_len (int): 序列批次的长度。
        pred_len (int): 预测批次的长度。
        stride (int): 每次移动的步长。
        session_lengths (list): 每段会话的实际长度列表。

    返回:
        reconstructed_sessions (list of torch.Tensor): 还原后的每段会话数据，每个会话的形状为 (feature, session_length)。
    """
    num_batches, num_features, _ = adv_preds.shape
    reconstructed_sessions = [torch.clone(session) for session in original_sessions]

    # 遍历每个批次，将 adv_preds 替换到原始会话的对应位置
    session_idx = 0
    j = 0
    for i in range(num_batches):
        pred_start_idx = seq_len+j * stride
        pred_end_idx = pred_start_idx + pred_len
        j += 1
        
        # 确保预测批次的长度不超过会话的实际长度
        if pred_end_idx > session_lengths[session_idx] and pred_start_idx <= session_lengths[session_idx]:
            pred_end_idx = session_lengths[session_idx]
            pred_len_tmp = pred_end_idx - pred_start_idx
            reconstructed_sessions[session_idx][:, pred_start_idx:pred_end_idx] = adv_preds[i, :, :pred_len_tmp]
            #更新会话索引
            session_idx = session_idx+1
            j = 0
        else:
            # 替换原始会话中的预测部分
            reconstructed_sessions[session_idx][:, pred_start_idx:pred_end_idx] = adv_preds[i, :, :pred_len]
            
    return reconstructed_sessions

# def reconstruct_adv_preds(adv_preds, original_sessions, seq_len, pred_len, stride, session_lengths):
#     """
#     反向
#     将 adv_preds 还原为原始的时间序列数据，考虑到每段会话长度不一致，并处理填充的0。

#     参数:
#         adv_preds (torch.Tensor): 形状为 (num_batches, feature, pred_len) 的扰动后的预测批次。
#         original_sessions (list of torch.Tensor): 原始的每段会话数据，每个会话的形状为 (feature, session_length)。
#         seq_len (int): 序列批次的长度。
#         pred_len (int): 预测批次的长度。
#         stride (int): 每次移动的步长。
#         session_lengths (list): 每段会话的实际长度列表。

#     返回:
#         reconstructed_sessions (list of torch.Tensor): 还原后的每段会话数据，每个会话的形状为 (feature, session_length)。
#     """
#     num_batches, num_features, _ = adv_preds.shape
#     reconstructed_sessions = [torch.clone(session) for session in original_sessions]

#     # 遍历每个批次，将 adv_preds 替换到原始会话的对应位置
#     session_idx = len(session_lengths)-1 # 当前处理的会话索引
#     j = (session_lengths[session_idx] -seq_len - pred_len)// stride+1
#     #从最后一个batch开始替换
#     for i in reversed(range(num_batches)):
#         pred_start_idx = seq_len+j * stride
#         pred_end_idx = pred_start_idx + pred_len
        
#         if  pred_end_idx > session_lengths[session_idx] and pred_start_idx <= session_lengths[session_idx]:
#             pred_end_idx = session_lengths[session_idx]
#             pred_len_tmp = pred_end_idx - pred_start_idx
#             reconstructed_sessions[session_idx][:, pred_start_idx:pred_end_idx] = adv_preds[i, :, :pred_len_tmp]
#         elif j == 0 :
#             #更新会话索引
#             reconstructed_sessions[session_idx][:, pred_start_idx:pred_end_idx] = adv_preds[i, :, :pred_len]
#             session_idx = session_idx-1
#             j = (session_lengths[session_idx] -seq_len- pred_len)// stride+1
#             continue
#         else:
#             # 替换原始会话中的预测部分
#             reconstructed_sessions[session_idx][:, pred_start_idx:pred_end_idx] = adv_preds[i, :, :pred_len]
            
#         j -= 1
#     return reconstructed_sessions

def get_window_indices(original_sessions, delta, win_size, n_wins):
    """
    只用于计算并返回固定窗口索引，不参与梯度计算。
    """
    total_indices_list = []
    for original_session in original_sessions:
        win_size_ms = win_size * 1000
        delta_ms = delta * 1000
        offset = win_size_ms - delta_ms

        session_ipd = original_session[0, :]
        cumulative_time = session_ipd.abs().cumsum(dim=0)
        
        indices_list = []
        for wi in range(int(n_wins)):
            start_time = wi * offset
            end_time = start_time + win_size_ms
            
            # 核心：计算索引并用 .item() 取出，变成普通整数
            start_idx = torch.searchsorted(cumulative_time, start_time).item()
            end_idx = torch.searchsorted(cumulative_time, end_time).item()
            indices_list.append((start_idx, end_idx))
        total_indices_list.append(indices_list)
    return total_indices_list

def partition_adv_sessions_by_original_ipd(adv_sessions, delta, win_size,n_wins, tor_len, device,total_indices_list=None):
    """
    按照 IPD 划分 adv_sessions 的窗口，并将每个窗口的 IPD 第一个元素改为 0。

    参数:
        adv_sessions (list of torch.Tensor): 每段会话的扰动后数据，每个会话的形状为 (2, session_length)。
        delta (int): 窗口重叠时间（秒）。
        win_size (int): 每个窗口的大小（秒）。
        n_wins (int): 每段会话的窗口数量。
        tor_len (int): 每个窗口的目标长度前部分ipd，后部分size,随后填充0至tor_len*2。

    返回:
        partitioned_data (list of torch.Tensor): 划分后的窗口数据，每个窗口的形状为 (tor_len * 2)。形状为 [batch_size,n_wins, tor_len * 2]
    """
    win_size_ms = win_size * 1000  # 将 win_size 转换为毫秒
    delta_ms = delta * 1000  # 将 delta 转换为毫秒
    offset = win_size_ms - delta_ms  # 计算窗口的偏移量

    partitioned_data = [] 
    for i,session in enumerate(adv_sessions):
        
        partitioned_data_sigle = []  #单独一个会话的
        
        session_ipd = session[0, :]  # IPD 数据（毫秒）
        session_size = session[1, :]  # size 数据
        indices_list = total_indices_list[i] 
        # 累积 IPD 得到绝对时间（毫秒）
        # cumulative_time = session_ipd.abs().cumsum(dim=0)
        
        for wi in range(int(n_wins)):
            # start_time = wi * offset
            # end_time = start_time + win_size_ms

            # 找到窗口内的数据索引
            # start_idx = torch.searchsorted(cumulative_time, start_time).item()
            # end_idx = torch.searchsorted(cumulative_time, end_time).item()

            # 提取窗口内的 IPD 和 size 数据
            #原论文数据处理也是ipd数据不够才补充size（疑问）
            # if end_idx > 500:
            #     end_idx = 500
            start_idx, end_idx = indices_list[wi] 
            window_ipd = session_ipd[start_idx:end_idx]
            window_size = session_size[start_idx:end_idx]

            # 将每个窗口的 IPD 第一个元素改为 0
            if len(window_ipd) > 0:
                window_ipd = torch.cat([torch.tensor([0.0]).to(device), window_ipd[1:]])

            # 将 IPD 和 size 数据连接在一起
            window = torch.cat([window_ipd, window_size])

            # 如果窗口内的数据不足，填充零
            if window.shape[0] < tor_len * 2:
                padding = torch.zeros(tor_len * 2 - window.shape[0]).to(device)
                window = torch.cat([window, padding])

            # 确保窗口数据长度为 tor_len * 2
            window = window[:tor_len * 2]

            partitioned_data_sigle.append(window)
            
        partitioned_data_sigle =torch.stack(partitioned_data_sigle,dim = 0)  #将一个会话的所有窗口堆叠成一个张量
        partitioned_data.append(partitioned_data_sigle)
    # 将所有会话堆叠成一个张量
    partitioned_data = torch.stack(partitioned_data,dim = 0)    #[batch_size,n_wins, tor_len * 2]

    return partitioned_data

# 放置在 partition_adv_sessions_by_ipd 函数之前
def differentiable_sample_1d(data, query_indices):
    """
    从 data 中采样 query_indices 指定的位置 (支持浮点数索引)。
    
    参数:
        data: [length] 原始数据 (session_ipd 或 session_size)
        query_indices: [output_len] 我们想要采样的位置索引 (可以是浮点数)
        
    返回:
        sampled_data: [output_len] 采样结果，保留了对 query_indices 的梯度
    """
    length = data.size(0)
    
    # 1. 找到左右整数索引
    # floor() 不可导，但 data[idx] 的内容梯度可以传；
    # 关键在于后面的 alpha 权重计算，它保留了位置梯度。
    idx_floor = query_indices.floor().long()
    idx_ceil = idx_floor + 1
    
    # 2. 处理越界情况 (Clamp防止报错)
    idx_floor_clamped = idx_floor.clamp(0, length - 1)
    idx_ceil_clamped = idx_ceil.clamp(0, length - 1)
    
    # 3. 获取左右两边的实际数值
    val_floor = data[idx_floor_clamped]
    val_ceil = data[idx_ceil_clamped]
    
    # 4. 计算插值权重 alpha (这是梯度回传的关键!)
    # 如果 query_index 从 5.4 变成 5.5，alpha 变大，result 就会更偏向 val_ceil
    alpha = query_indices - idx_floor.float() # alpha range: [0, 1]
    
    # 5. 线性插值
    sampled_data = val_floor * (1 - alpha) + val_ceil * alpha
    
    # 6. 掩码处理：对于原本就在数组范围之外的索引，强制置为 0
    # 使用 detach() 避免对 mask 求导 (mask 只是为了去掉无效数据)
    mask = (query_indices >= 0) & (query_indices < length - 1)
    mask = mask.float().detach()
    
    return sampled_data * mask

def partition_adv_sessions_differentiable(adv_sessions, delta, win_size, n_wins, tor_len, device):
    """
    完全可微的窗口划分函数。
    """
    win_size_ms = win_size * 1000
    delta_ms = delta * 1000
    offset = win_size_ms - delta_ms

    partitioned_data = []
    
    # 实例化您的可微搜索类
    differentiable_search = DifferentiableSearchsorted.apply
    
    # 预生成一个 0 到 tor_len-1 的网格，用于相对索引
    # 这代表我们在每个窗口中想要采样的点数
    base_grid = torch.arange(tor_len, dtype=torch.float32, device=device)
    
    for session in adv_sessions:
        partitioned_data_single = []
        
        session_ipd = session[0, :]
        session_size = session[1, :]
        
        # 累积时间 (梯度源头之一)
        cumulative_time = session_ipd.abs().cumsum(dim=0)

        for wi in range(int(n_wins)):
            # 1. 计算目标时间范围
            start_time = torch.tensor(float(wi * offset), device=device)
            end_time = torch.tensor(float(start_time + win_size_ms), device=device)

            # 2. 获取【浮点数】索引 (梯度源头之二)
            # 注意：这里千万不要加 .long()，保留 float 类型！
            start_idx_float = differentiable_search(cumulative_time, start_time)
            end_idx_float = differentiable_search(cumulative_time, end_time)
            
            # 3. 构建采样坐标系 (Sampling Grid)
            # 我们从 start_idx_float 开始，往后采 tor_len 个点
            # 比如 start 是 5.4，我们采 5.4, 6.4, 7.4 ...
            sample_indices = start_idx_float + base_grid
            
            # 4. 执行可微采样 (替代硬切片)
            window_ipd = differentiable_sample_1d(session_ipd, sample_indices)
            window_size = differentiable_sample_1d(session_size, sample_indices)
            
            # 5. 处理 end_time 的约束 (关键步骤)
            # 硬切片时，数据只取到 end_idx。现在我们采了 tor_len 个点，可能超出了 end_idx。
            # 我们用 Soft Mask 把超出 end_idx_float 的部分“压”为 0。
            # 这样 end_idx_float 的梯度也能发生作用！
            valid_mask = torch.sigmoid((end_idx_float - sample_indices) * 10.0) # 10.0 是 steepness
            
            window_ipd = window_ipd * valid_mask
            window_size = window_size * valid_mask

            # 6. 修正第一个元素 (保持原逻辑：将窗口内第一个 IPD 置 0)
            # 为了保持可微，我们不能直接赋值 window_ipd[0] = 0
            # 而是生成一个 mask：[0, 1, 1, 1...]
            zero_mask = torch.ones_like(window_ipd)
            zero_mask[0] = 0.0
            window_ipd = window_ipd * zero_mask
            
            # 7. 拼接
            window = torch.cat([window_ipd, window_size])
            
            # 8. 填充 (Padding)
            # 在采样法中，我们实际上已经固定采了 tor_len 个点。
            # 如果 sample_indices 超出了 session 长度，sample_1d 函数会自动置 0。
            # 所以我们得到的结果长度天然就是 tor_len，不需要额外的 padding 逻辑！
            # 但为了保险起见，或者如果您原本逻辑是 padding 到 tor_len*2：
            if window.shape[0] < tor_len * 2:
                 padding = torch.zeros(tor_len * 2 - window.shape[0], device=device)
                 window = torch.cat([window, padding])
            else:
                 window = window[:tor_len * 2] # 确保长度一致

            partitioned_data_single.append(window)
            
        partitioned_data_single = torch.stack(partitioned_data_single, dim=0)
        partitioned_data.append(partitioned_data_single)

    partitioned_data = torch.stack(partitioned_data, dim=0)
    return partitioned_data

def partition_adv_sessions_by_ipd(adv_sessions, delta, win_size,n_wins, tor_len, device):
    """
    按照 IPD 划分 adv_sessions 的窗口，并将每个窗口的 IPD 第一个元素改为 0。

    参数:
        adv_sessions (list of torch.Tensor): 每段会话的扰动后数据，每个会话的形状为 (2, session_length)。
        delta (int): 窗口重叠时间（秒）。
        win_size (int): 每个窗口的大小（秒）。
        n_wins (int): 每段会话的窗口数量。
        tor_len (int): 每个窗口的目标长度前部分ipd，后部分size,随后填充0至tor_len*2。

    返回:
        partitioned_data (list of torch.Tensor): 划分后的窗口数据，每个窗口的形状为 (tor_len * 2)。形状为 [batch_size,n_wins, tor_len * 2]
    """
    win_size_ms = win_size * 1000  # 将 win_size 转换为毫秒
    delta_ms = delta * 1000  # 将 delta 转换为毫秒
    offset = win_size_ms - delta_ms  # 计算窗口的偏移量

    partitioned_data = []
    for session in adv_sessions:
        
        partitioned_data_sigle = []  #单独一个会话的
        
        session_ipd = session[0, :]  # IPD 数据（毫秒）
        session_size = session[1, :]  # size 数据
        session_length = session_ipd.shape[0]

        # 累积 IPD 得到绝对时间（毫秒）
        cumulative_time = session_ipd.abs().cumsum(dim=0)

        for wi in range(int(n_wins)):
            start_time = wi * offset
            end_time = start_time + win_size_ms

            # 找到窗口内的数据索引
            start_idx = torch.searchsorted(cumulative_time, start_time).item()
            end_idx = torch.searchsorted(cumulative_time, end_time).item()

            # 提取窗口内的 IPD 和 size 数据
            #原论文数据处理也是ipd数据不够才补充size（疑问）
            # if end_idx > 500:
            #     end_idx = 500
            window_ipd = session_ipd[start_idx:end_idx]
            window_size = session_size[start_idx:end_idx]

            # 将每个窗口的 IPD 第一个元素改为 0
            if len(window_ipd) > 0:
                window_ipd = torch.cat([torch.tensor([0.0]).to(device), window_ipd[1:]])

            # 将 IPD 和 size 数据连接在一起
            window = torch.cat([window_ipd, window_size])
                
            # 如果窗口内的数据不足，填充零
            if window.shape[0] < tor_len * 2:
                padding = torch.zeros(tor_len * 2 - window.shape[0]).to(device)
                window = torch.cat([window, padding])

            # 确保窗口数据长度为 tor_len * 2
            window = window[:tor_len * 2]

            partitioned_data_sigle.append(window)
            
        partitioned_data_sigle =torch.stack(partitioned_data_sigle,dim = 0)  #将一个会话的所有窗口堆叠成一个张量
        partitioned_data.append(partitioned_data_sigle)
    # 将所有会话堆叠成一个张量
    partitioned_data = torch.stack(partitioned_data,dim = 0)    #[batch_size,n_wins, tor_len * 2]

    return partitioned_data


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        self.args = args
        self.ckpt = pathlib.Path(self.args.target_model_path).resolve()
        self.fields = self.ckpt.name.split("_")
        self.delta = int(self.fields[-12].split("d")[-1])
        self.win_size = int(self.fields[-11].split("ws")[-1])
        self.n_wins = int(self.fields[-10].split("nw")[-1])
        self.threshold = int(self.fields[-9].split("thr")[-1])
        self.tor_len = int(self.fields[-8].split("tl")[-1])
        self.exit_len = int(self.fields[-7].split("el")[-1])

        self.n_test = int(self.fields[-6].split("nt")[-1])
        self.emb_size = int(self.fields[-4].split("es")[-1])
        
        self.batch_size = self.args.batch_size
        
        super(Exp_Main, self).__init__(args)
        
    def _build_model(self):
        target_model_dict = {
            'Deepcoffea': Deepcoffea,
        }
        
        Target_model_anchor = target_model_dict[self.args.target_model].Model(emb_size=self.emb_size, input_size=self.tor_len*2).float()
        Target_model_pandn = target_model_dict[self.args.target_model].Model(emb_size=self.emb_size, input_size=self.exit_len*2).float()
        
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
            Target_model_anchor = nn.DataParallel(Target_model_anchor, device_ids=self.args.device_ids)
            Target_model_pandn = nn.DataParallel(Target_model_pandn, device_ids=self.args.device_ids)
            
        Generator = TSTGenerator(seq_len=self.args.seq_len, 
                                patch_len=self.args.patch_len,
                                pred_len=self.args.pred_len,
                                feat_dim=self.args.enc_in, 
                                depth=self.args.depth, 
                                scale_factor=self.args.scale_factor, 
                                n_layers=self.args.n_layers, 
                                d_model=self.args.d_model, 
                                n_heads=self.args.n_heads,
                                individual=self.args.individual, 
                                d_k=None, d_v=None, 
                                d_ff=self.args.d_ff, 
                                norm='BatchNorm', 
                                attn_dropout=self.args.att_dropout, 
                                head_dropout=self.args.head_dropout, 
                                act=self.args.activation,pe='zeros', 
                                learn_pe=True,pre_norm=False, 
                                res_attention=False, 
                                store_attn=False)
        
        return Target_model_anchor,Target_model_pandn,Generator

    def _get_data(self, flag):
        
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.generator.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        
        criterion = Loss()
            
        return criterion

    
    def train(self, setting):
        
        pred_len = self.args.pred_len
        seq_len = self.args.seq_len
        stride = self.args.stride
        
        delta = self.delta
        win_size = self.win_size
        n_wins = self.n_wins
        threshold = self.threshold
        tor_len = self.tor_len
        exit_len = self.exit_len

        n_test = self.n_test
        emb_size = self.emb_size
        batch_size = self.batch_size
        #n_test表示在初始电路中的测试集划分数量，当然，在处理过程中，一些不符合要求的电路会被筛选掉，因此剩余的电路数量会小或等于n_test
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        
        state_dict = torch.load(os.path.join(self.args.target_model_path,'best_loss.pth'), map_location=self.device)
        
        anchor = self.target_model_anchor
        pandn = self.target_model_pandn
        generator = self.generator
        
        anchor.load_state_dict(state_dict['anchor_state_dict'])
        pandn.load_state_dict(state_dict['pandn_state_dict'])
        
        anchor.eval()
        pandn.eval()
        generator.train()
        
        # Freeze Target Model parameters
        for param in anchor.parameters():
            param.requires_grad = False
        for param in pandn.parameters():
            param.requires_grad = False
            
        data_path=pathlib.Path(self.args.data_path)
        train_path =data_path / "filtered_and_partitioned" / f"d{delta}_ws{win_size}_nw{n_wins}_thr{threshold}_tl{tor_len}_el{exit_len}_nt{n_test}_train.npz"
        test_path = data_path / "filtered_and_partitioned" / f"d{delta}_ws{win_size}_nw{n_wins}_thr{threshold}_tl{tor_len}_el{exit_len}_nt{n_test}_test.npz"
        train_session = data_path / "filtered_and_partitioned" / f"d{delta}_ws{win_size}_nw{n_wins}_thr{threshold}_tl{tor_len}_el{exit_len}_nt{n_test}_train_session.npz"
        test_session = data_path / "filtered_and_partitioned" / f"d{delta}_ws{win_size}_nw{n_wins}_thr{threshold}_tl{tor_len}_el{exit_len}_nt{n_test}_test_session.npz"
        train_data_win = np.load(train_path)
        train_data_session = np.load(train_session,allow_pickle=True)
        
        train_steps  = (np.reshape(train_data_win['train_tor'], [-1, tor_len*2]).astype('float32').shape[0])//batch_size+1
        
        # early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        scheduler_oclr = lr_scheduler.OneCycleLR(optimizer = model_optim,
                                            steps_per_epoch = 20,
                                            pct_start = self.args.pct_start,
                                            epochs = self.args.train_epochs,
                                            max_lr = self.args.learning_rate,
                                            )
        
        scheduler_steplr = lr_scheduler.StepLR(model_optim, step_size=1, gamma=0.5)
            
        for epoch in range(self.args.train_epochs):  
            
            #根据读取数据的规则，size被缩小了1000倍，time被放大了1000倍
            #总损失，分类损失，扰动的L2距离相似度损失（时间，大小）(平均值)
            total_loss = []
            totol_cosine_loss = []

            
            #L2范式的比率
            total_time_rate = []
            total_size_rate = []
                
            #DeepcoffeaDataset返回的是一次会话的11个窗口的流量
            train_set = DeepCoffeaDataset(train_data_win,train=True)
            train_loader = DataLoader(train_set, batch_size=batch_size,shuffle=True)

            # test_set = DeepCoffeaDataset(test_data,train=False)
            # test_loader = DataLoader(test_set, batch_size=batch_size,shuffle=False)
            
            epoch_time = time.time()

            for i, (idx,_,xp_batch) in tqdm(enumerate(train_loader),ncols=120):
                
                model_optim.zero_grad()
                
                xp_batch = xp_batch.to(self.device)   #xp_batch:[batch_size, n_wins, exit_len*2]
                
                idx = [int(x) for x in idx]
                #bacthsession:[batch_size,],每个会话的窗口的流量长度不一致
                batch_session_ipd = train_data_session['tor_ipds'][idx]
                batch_session_size = train_data_session['tor_sizes'][idx]
                
                batch_session_ipd = [torch.tensor(tensor).to(self.device) for tensor in batch_session_ipd]
                batch_session_size = [torch.tensor(tensor).to(self.device) for tensor in batch_session_size]
                
                
                seq_batches =[]     #存储batches的所有seq_batches
                pred_batches = []   #存储对应batches的所有pred_batches
                len_sessions = []    #存储每个会话的长度
                batch_sessions = []   #存储每个会话的流量
                
                #对每一条会话进行处理[batch_size, ] -> batch_seq:[, FEATURE_DIM, seq_len],batch_pred:[, FEATURE_DIM, pred_len]
                for j in range(batch_size):
                                        
                    len_session = len(batch_session_ipd[j])
                    
                    batch_session = torch.stack([batch_session_ipd[j],batch_session_size[j]],dim=0)   #batch_session:[feature , len]
                    
                    seq_batch, pred_batch= split_time_series(batch_session, seq_len=seq_len, pred_len=pred_len, stride=stride, device = self.device)   #处理每条会话
                    
                    seq_batches.append(seq_batch)
                    pred_batches.append(pred_batch)
                    len_sessions.append(len_session)
                    batch_sessions.append(batch_session)
                    
                #将每条会话的所有seq_batches和pred_batches拼接在一起
                seq_batches = torch.cat(seq_batches,dim=0)     #seq_batches:[, feature_dim, seq_len]
                pred_batches = torch.cat(pred_batches,dim=0)    #pred_batches:[, feature_dim, pred_len]      
                
                z = seq_batches.float()  #z:[,, feature_dim, seq_len]
                
                #生成扰动
                perturbation = generator(z) # z -> perturbation：[,feature_dim,seq_len] -> [,feat_dim,pred_len]
                
                abs_x = torch.abs(pred_batches)
                #避免原地修改
                constraint_perturbation = perturbation.clone()
                # perturbation[:,0,:]=torch.abs(perturbation[:,0,:]) 
                # perturbation[:,1,:]=torch.abs(perturbation[:,1,:])
                constraint_ipd = torch.abs(perturbation[:,0,:])     #约束ipd
                constraint_size = torch.abs(perturbation[:,1,:])    #约束size
                
                constraint_perturbation[:,0,:] = constraint_ipd
                constraint_perturbation[:,1,:] = constraint_size
                
                adv_preds = torch.sign(pred_batches) * (torch.abs(pred_batches) + constraint_perturbation)

                #还原所有会话流量，batch_sessions：所有原始的每段会话数据，每个会话的形状为 [feature, session_length]
                adv_sessions = reconstruct_adv_preds(adv_preds, batch_sessions, seq_len, pred_len, stride, len_sessions) #将扰动添加到batch_sessions中
                
                #生成扰动后的样本
                # Gxa_batch = partition_adv_sessions_by_ipd(adv_sessions, delta, win_size,n_wins, tor_len, device = self.device)  
                Gxa_batch = partition_adv_sessions_differentiable(adv_sessions, delta, win_size,n_wins, tor_len, device = self.device)   

                #Ga_out: [batch_size*n_wins, emb_size], p_out: [batch_size*n_wins, emb_size]

                
                
                Gxa_batch = Gxa_batch.reshape(-1, tor_len*2).float()    #Gxa_batch:[n_wins,batch_size, exit_len*2] -> [batch_size*n_wins, exit_len*2]
                xp_batch = xp_batch.reshape(-1, exit_len*2).float()   #xp_batch:[n_wins, batch_size ,exit_len*2] -> [batch_size*n_wins, exit_len*2]
                
                Ga_out = anchor(Gxa_batch)   
                p_out = pandn(xp_batch)
                

                #计算扰动L2距离
                ipdl2_distances = torch.stack([torch.linalg.norm(p - o,ord=2,dim =-1) for p, o in zip([row[0] for row in adv_sessions] , batch_session_ipd)])
                sizel2_distances = torch.stack([torch.linalg.norm(p - o,ord=2,dim =-1) for p, o in zip([row[1] for row in adv_sessions], batch_session_size)])
                
                #会话L2距离
                time0L2 = torch.stack([torch.linalg.norm(x,ord=2,dim =-1) for x in  batch_session_ipd])
                size0L2 = torch.stack([torch.linalg.norm(x,ord=2,dim =-1) for x in  batch_session_size])

                time_rate = (ipdl2_distances/time0L2).mean()
                size_rate = (sizel2_distances/size0L2).mean()

                loss,cosine_loss,beta,alpha,gamma = criterion (time_rate, size_rate,Ga_out, p_out)

                total_loss.append(loss.item())  
                totol_cosine_loss.append(cosine_loss.item())

                total_time_rate.append(time_rate.item())
                total_size_rate.append(size_rate.item())

                    
                loss.backward()
                model_optim.step()
                # scheduler_oclr.step()
                # if(epoch >3 and cosine_loss.item() < 0.8 and time_rate <0.13 and size_rate <0.15):
                #     torch.save({
                #         'epoch': epoch + 1,
                #         'meancos_loss': meancosine_loss-0.5,
                #         'Time L2 rate':time_rate,
                #         'Size L2 rate':size_rate,
                #         'generator_state_dict': self.generator.state_dict(),
                #         'optimizer_state_dict': model_optim.state_dict(),
                #         'loss': meanloss,
                #     }, os.path.join(path, f'generator_checkpoint_{epoch + 1}_cos{cosine_loss-0.5:.3f}_advipd{time_rate:.3f}_advsize{size_rate:.3f}.pth'))
                if(i>19):
                    break
                
            adjust_learning_rate(model_optim, scheduler_oclr, epoch + 1, self.args, printout=False)
            print("\nEpoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            print('Updating learning rate to {}'.format(model_optim.param_groups[0]['lr']))
            
            meanloss = np.mean(total_loss)
            meancosine_loss = np.mean(totol_cosine_loss)
            meantimeL2_rate = np.mean(total_time_rate)
            meansizeL2_rate = np.mean(total_size_rate)
            
            print("Epoch: {0},Loss: {1}, Cosine Loss: {2}, Time L2 rate: {3}, Size L2 rate: {4}\n".format(epoch + 1, meanloss, meancosine_loss, meantimeL2_rate, meansizeL2_rate))
            # if(meancosine_loss < 0.80 and meantimeL2_rate <0.15 and meansizeL2_rate <0.15):
            #     torch.save({
            #         'epoch': epoch + 1,
            #         'meancos_loss': meancosine_loss-0.5,
            #         'Time L2 rate':meantimeL2_rate,
            #         'Size L2 rate':meansizeL2_rate,
            #         'generator_state_dict': self.generator.state_dict(),
            #         'optimizer_state_dict': model_optim.state_dict(),
            #         'loss': meanloss,
            #         'beta': beta,
            #         'alpha': alpha, 
            #         'gamma': gamma 
            #     }, os.path.join(path, f'generator_checkpoint_{epoch + 1}_cos{meancosine_loss-0.5:.3f}_advipd{meantimeL2_rate:.3f}_advsize{meansizeL2_rate:.3f}.pth'))

                # 打印保存信息
                # print(f"Checkpoint saved at epoch {epoch + 1}")
            
            # self.test()
            

    def test(self, setting):
        
        pred_len = self.args.pred_len
        seq_len = self.args.seq_len
        stride = self.args.stride
        
        delta = self.delta
        win_size = self.win_size
        n_wins = self.n_wins
        threshold = self.threshold
        tor_len = self.tor_len
        exit_len = self.exit_len

        n_test = self.n_test
        emb_size = self.emb_size
        batch_size = self.batch_size
        #n_test表示在初始电路中的测试集划分数量，当然，在处理过程中，一些不符合要求的电路会被筛选掉，因此剩余的电路数量会小或等于n_test
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            print('No checkpoint found')
            
        state_dict = torch.load(os.path.join(self.args.target_model_path,'best_loss.pth'), map_location=self.device)
        
        generator_path = 'checkpoints/deepcoffea/deepcoffea_d3_ws5_nw11_thr20_tl500_el800_nt1000_ap1e-01_es64_lr1e-03_mep100000_bs256/deepcoffea_PatchTST_Deepcoffea_sl150_pl70_dm512_nh8_pal10_s70/generator_checkpoint_16_cos0.163_advipd0.149_advsize0.005.pth'
        generator_state_dict = torch.load(generator_path, map_location=self.device)
        
        anchor = self.target_model_anchor
        pandn = self.target_model_pandn
        generator = self.generator
        
        anchor.load_state_dict(state_dict['anchor_state_dict'])
        pandn.load_state_dict(state_dict['pandn_state_dict'])
        generator.load_state_dict(generator_state_dict['generator_state_dict'])
        
        
        anchor.eval()
        pandn.eval()
        generator.eval()
                    
        data_path=pathlib.Path(self.args.data_path)
        train_path =data_path / "filtered_and_partitioned" / f"d{delta}_ws{win_size}_nw{n_wins}_thr{threshold}_tl{tor_len}_el{exit_len}_nt{n_test}_train.npz"
        test_path = data_path / "filtered_and_partitioned" / f"d{delta}_ws{win_size}_nw{n_wins}_thr{threshold}_tl{tor_len}_el{exit_len}_nt{n_test}_test.npz"
        train_session = data_path / "filtered_and_partitioned" / f"d{delta}_ws{win_size}_nw{n_wins}_thr{threshold}_tl{tor_len}_el{exit_len}_nt{n_test}_train_session.npz"
        test_session = data_path / "filtered_and_partitioned" / f"d{delta}_ws{win_size}_nw{n_wins}_thr{threshold}_tl{tor_len}_el{exit_len}_nt{n_test}_test_session.npz"
        
        test_data_win = np.load(test_path)
        test_data_session = np.load(test_session,allow_pickle=True)
            
        test_set = DeepCoffeaDataset(test_data_win,train=False)
        test_loader = DataLoader(test_set, batch_size=batch_size,shuffle=True,drop_last=True)

        Gtor_embs = []
        tor_embs = []
        exit_embs = []
        
        #根据读取数据的规则，size被缩小了1000倍，time被放大了1000倍
        total_sizeL2_distances = []
        total_timeL2_distances = []
        
        #向量总距离（时间，大小）
        total_size0L2 = []
        total_time0L2 = []
        
        #L2范式的比率
        total_time_rate = []
        total_size_rate = [] 
               
        with torch.no_grad():
            num_iter = 500
            batch_iter = num_iter/ batch_size 
            total_indices_list = []
            for i, (idx,xa_batch,xp_batch) in tqdm(enumerate(test_loader),ncols=120):  
                          
                xa_batch = xa_batch.to(self.device)   #xa_batch:[batch_size, n_wins, exit_len*2]
                xp_batch = xp_batch.to(self.device)   #xp_batch:[batch_size, n_wins, exit_len*2]
                
                idx = [int(x) for x in idx]
                #bacthsession:[batch_size,],每个会话的窗口的流量长度不一致
                batch_session_ipd = test_data_session['tor_ipds'][idx]
                batch_session_size = test_data_session['tor_sizes'][idx]
                
                batch_session_ipd = [torch.tensor(tensor).to(self.device) for tensor in batch_session_ipd]
                batch_session_size = [torch.tensor(tensor).to(self.device) for tensor in batch_session_size]
                
                
                seq_batches =[]     #存储batches的所有seq_batches
                pred_batches = []   #存储对应batches的所有pred_batches
                len_sessions = []    #存储每个会话的长度
                batch_sessions = []   #存储每个会话的流量
                
                #对每一条会话进行处理[batch_size, ] -> batch_seq:[, FEATURE_DIM, seq_len],batch_pred:[, FEATURE_DIM, pred_len]
                for j in range(batch_size):
                                        
                    len_session = len(batch_session_ipd[j])
                    
                    batch_session = torch.stack([batch_session_ipd[j],batch_session_size[j]],dim=0)   #batch_session:[feature , len]
                    
                    seq_batch, pred_batch= split_time_series(batch_session, seq_len=seq_len, pred_len=pred_len, stride=stride, device = self.device)   #处理每条会话
                    
                    seq_batches.append(seq_batch)
                    pred_batches.append(pred_batch)
                    len_sessions.append(len_session)
                    batch_sessions.append(batch_session)
                    
                #将每条会话的所有seq_batches和pred_batches拼接在一起
                seq_batches = torch.cat(seq_batches,dim=0)     #seq_batches:[, feature_dim, seq_len]
                pred_batches = torch.cat(pred_batches,dim=0)    #pred_batches:[, feature_dim, pred_len]      
                
                z = seq_batches.float()  #z:[,, feature_dim, seq_len]
                
                #生成扰动
                perturbation = generator(z) # z -> perturbation：[,feature_dim,seq_len] -> [,feat_dim,pred_len]
                
                abs_x = torch.abs(pred_batches)
                #避免原地修改
                constraint_perturbation = perturbation.clone()
                # perturbation[:,0,:]=torch.abs(perturbation[:,0,:])
                # perturbation[:,1,:]=torch.abs(perturbation[:,1,:])
                constraint_ipd = torch.abs(perturbation[:,0,:])    #约束ipd
                constraint_size = torch.abs(perturbation[:,1,:])    #约束size
                
                constraint_perturbation[:,0,:] = constraint_ipd
                constraint_perturbation[:,1,:] = constraint_size
                
                adv_preds = torch.sign(pred_batches) * (torch.abs(pred_batches) + constraint_perturbation)
                
                adv_sessions = reconstruct_adv_preds(adv_preds, batch_sessions, seq_len, pred_len, stride, len_sessions) #将扰动添加到batch_sessions中
                # total_indices_list = get_window_indices(batch_sessions, delta, win_size, n_wins)  #获取每个会话的窗口索引列表
                #生成扰动后的样本
                Gxa_batch = partition_adv_sessions_by_ipd(adv_sessions, delta, win_size,n_wins, tor_len, device = self.device)
                # Gxa_batch = partition_adv_sessions_by_original_ipd(adv_sessions, delta, win_size,n_wins, tor_len, self.device,total_indices_list)    
                

                #Ga_out: [batch_size*n_wins, emb_size], p_out: [batch_size*n_wins, emb_size]            
                Gxa_batch = Gxa_batch.reshape(-1, tor_len*2).float()    #Gxa_batch:[n_wins,batch_size, exit_len*2] -> [batch_size*n_wins, exit_len*2]
                xa_batch =  xa_batch.reshape(-1, tor_len*2).float()    #xa_batch:[n_wins,batch_size, exit_len*2] -> [batch_size*n_wins, exit_len*2]
                xp_batch = xp_batch.reshape(-1, exit_len*2).float()   #xp_batch:[n_wins, batch_size ,exit_len*2] -> [batch_size*n_wins, exit_len*2]
                
                Ga_out = anchor(Gxa_batch)
                a_out = anchor(xa_batch)
                p_out = pandn(xp_batch)
                
                Gtor_embs.append(Ga_out.cpu().numpy())
                tor_embs.append(a_out.cpu().numpy())
                exit_embs.append(p_out.cpu().numpy())
                
                
                #计算扰动L2距离
                ipdl2_distances = torch.stack([torch.linalg.norm(p - o,ord=2,dim=-1) for p, o in zip([row[0] for row in adv_sessions] , batch_session_ipd)])
                sizel2_distances = torch.stack([torch.linalg.norm(p - o,ord=2,dim=-1) for p, o in zip([row[1] for row in adv_sessions], batch_session_size)])
                
                #会话L2距离
                time0L2 = torch.stack([torch.linalg.norm(x,ord=2,dim=-1) for x in  batch_session_ipd])
                size0L2 = torch.stack([torch.linalg.norm(x,ord=2,dim=-1) for x in  batch_session_size])

                time_rate = (ipdl2_distances/time0L2).mean()
                size_rate = (sizel2_distances/size0L2).mean()
                
                print('Batch: {0},session_idx:{1} \nTime L2 distance rate: {2}, Size L2 Distance rate: {3}'.format(i,idx, time_rate.item(), size_rate.item()))
                
                print(f"{batch_size*(i+1)}session's pairs done.")
                

                total_time_rate.append(time_rate.item())
                total_size_rate.append(size_rate.item())
                if(i >= batch_iter-1):
                    break
            
            Gtor_embs = np.concatenate(Gtor_embs) # (N, emb_size)
            tor_embs = np.concatenate(tor_embs)     # (N, emb_size)
            exit_embs = np.concatenate(exit_embs)   # (N, emb_size)
            
            corr_matrix = cosine_similarity(tor_embs, exit_embs)
            Gcorr_matrix = cosine_similarity(Gtor_embs, exit_embs)
            
            meantimeL2_rate = np.mean(total_time_rate)
            meansizeL2_rate = np.mean(total_size_rate)
            
            sim = np.mean(np.diag(corr_matrix))
            Gsim = np.mean(np.diag(Gcorr_matrix))
            np.savez_compressed(os.path.join(path,f"corrmatrix_sim{sim:.4f}"), corr_matrix=corr_matrix,sim = sim)
            np.savez_compressed(os.path.join(path,f"Gcorrmatrix_time{meantimeL2_rate:.4f}_size{meansizeL2_rate:.4f}_sim{Gsim:.4f}"), corr_matrix=Gcorr_matrix,meantimeL2_rate = meantimeL2_rate,meansizeL2_rate= meansizeL2_rate,sim= Gsim)
                    
            print('Time L2 distance rate: {0}, Size L2 Distance rate: {1}'.format(meantimeL2_rate.item(), meansizeL2_rate.item()))
            print('Cosine Similarity: {0}, G Cosine Similarity: {1}'.format(sim, Gsim))