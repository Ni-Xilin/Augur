import torch
import numpy as np
import pickle
import pathlib
from target_model.Deepcorr300 import Model as Net

def format_deepcorr_input_from_samples(here_sample, there_sample, flow_size=300):
    """
    根据您最新的脚本逻辑，从两个预处理好的样本中提取'here'和'there'部分，
    并将它们组合成DeepCorr的输入张量。

    Args:
        here_sample (torch.Tensor): 形状为 (1, 8, flow_size)，提供 "here" 部分的流。
        there_sample (torch.Tensor): 形状为 (1, 8, flow_size)，提供 "there" 部分的流。
        flow_size (int): 序列的目标长度。

    Returns:
        torch.Tensor: 形状为 (1, 1, 8, flow_size) 的张量，可直接输入模型。
    """
    input_tensor = torch.zeros((1, 1, 8, flow_size)).to(here_sample.device)

    # 从 here_sample 中提取 here 部分的特征 (rows 0, 3, 4, 7)
    here_indices = [0, 3, 4, 7]
    input_tensor[0, 0, here_indices, :] = here_sample[0, here_indices, :]

    # 从 there_sample 中提取 there 部分的特征 (rows 1, 2, 5, 6)
    there_indices = [1, 2, 5, 6]
    input_tensor[0, 0, there_indices, :] = there_sample[0, there_indices, :]
    
    return input_tensor

# --- 1. 参数和模型设置 ---
N_SAMPLES = 11
FLOW_SIZE = 300

# 设置设备
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if use_cuda else "cpu")
print(f"正在使用设备: {device}")

# 加载模型
print("正在加载预训练的DeepCorr模型...")
model = Net().to(device)
try:
    model_path = 'target_model/deepcorr/deepcorr300/tor_199_epoch23_acc0.82dict.pth'
    state_dict = torch.load(model_path, map_location=device) # 添加 weights_only=False 以兼容新版PyTorch
    model.load_state_dict(state_dict)
    model.eval()
    print("模型加载成功。")
except FileNotFoundError:
    print(f"错误: 未找到模型文件 {model_path}。请检查路径。")
    exit()
except Exception as e:
    print(f"加载模型时出错: {e}")
    exit()

# --- 2. 加载预处理好的扰动后样本 ---
print("正在加载扰动后的样本文件...")
try:
    adv_samples_fpath = pathlib.Path("target_model/deepcorr/deepcorr300/Gbase/Gdeepcorr_advsamples_time0.1343_size0.0089.p")
    with open(adv_samples_fpath, "rb") as fp:
        data = pickle.load(fp)
        total_adv_samples = data["adv_samples"] # total_adv_samples 是一个Tensor
    print(f"样本加载成功，总共 {len(total_adv_samples)} 个样本。")
except FileNotFoundError:
    print(f"错误: 未找到样本文件 {adv_samples_fpath}。")
    exit()

# 选取前16个样本用于测试
adv_samples_subset = total_adv_samples[:N_SAMPLES]
print(f"已选取前 {N_SAMPLES} 个样本进行配对评估。")

# --- 3. 计算分数矩阵 ---
score_matrix = np.zeros((N_SAMPLES, N_SAMPLES))

print(f"\n开始计算 {N_SAMPLES}x{N_SAMPLES} 的分数矩阵...")
with torch.no_grad():
    for i in range(N_SAMPLES):
        for j in range(N_SAMPLES):
            # 获取第 i 个样本作为 "here" 流的来源
            here_sample = adv_samples_subset[i].to(device)
            # 获取第 j 个样本作为 "there" 流的来源
            there_sample = adv_samples_subset[j].to(device)
            
            # 格式化为模型输入
            input_tensor = format_deepcorr_input_from_samples(here_sample, there_sample, FLOW_SIZE)
            input_tensor = input_tensor.to(device)
            
            # 模型推理
            output_raw = model(input_tensor, dropout=0.0)
            
            # 应用Sigmoid函数得到概率
            # output_prob = torch.sigmoid(output_raw)
            output_prob = output_raw  # 取平均值作为该对的分数
            
            # 存入矩阵
            score_matrix[i, j] = output_prob.item()
        
        print(f"  完成第 {i+1}/{N_SAMPLES} 行的计算...")

# --- 4. 打印并分析结果 ---
print("\n" + "="*50)
print(f"--- DeepCorr  分数矩阵 ({N_SAMPLES}x{N_SAMPLES}) ---")
print("行: 'here' 部分来源, 列: 'there' 部分来源")
print("="*50)

# 设置打印选项，使其更易读
np.set_printoptions(precision=4, suppress=True, linewidth=120)
print(score_matrix)

with open("plotwork/Gdeepcorr_score_matrix.p", "wb") as fp:
    pickle.dump(score_matrix, fp)
