import torch
import numpy as np
import pickle
import pathlib
from tqdm import tqdm

# 导入您的两个DeepCorr模型定义
from target_model.Deepcorr100 import Model as Deepcorr100Model
from target_model.Deepcorr700 import Model as Deepcorr700Model

def format_deepcorr_input_from_samples(here_sample, there_sample, flow_size):
    """
    从两个预处理好的样本中提取'here'和'there'部分，
    并将它们组合成DeepCorr的输入张量。
    """
    input_tensor = torch.zeros((1, 1, 8, flow_size)).to(here_sample.device)
    here_indices = [0, 3, 4, 7]
    input_tensor[0, 0, here_indices, :flow_size] = here_sample[0, here_indices, :flow_size]
    there_indices = [1, 2, 5, 6]
    input_tensor[0, 0, there_indices, :flow_size] = there_sample[0, there_indices, :flow_size]
    return input_tensor

# --- 1. 参数和模型设置 ---
N_SAMPLES = 11
FLOW_SIZE_DC100 = 100
FLOW_SIZE_DC700 = 700
THRESHOLD_DC100 = 0.01 # m-DeepCorr第一阶段的筛选阈值

# 设置设备
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
print(f"正在使用设备: {device}")

# --- 加载 DC100 和 DC700 模型 ---
print("正在加载预训练的 m-DeepCorr 模型 (DC100 和 DC700)...")
model_dc100 = Deepcorr100Model().to(device)
model_dc700 = Deepcorr700Model().to(device)

try:
    # 加载 DC100 权重
    dc100_path = 'target_model/deepcorr/deepcorr100/tor_199_epoch10_acc0.66.pth'
    model_dc100.load_state_dict(torch.load(dc100_path, map_location=device))
    model_dc100.eval()
    print("DC100 模型加载成功。")
    
    # 加载 DC700 权重
    dc700_path = 'target_model/deepcorr/deepcorr700/tor700_199_epoch11_acc0.88.pth'
    model_dc700.load_state_dict(torch.load(dc700_path, map_location=device,))
    model_dc700.eval()
    print("DC700 模型加载成功。")
    
except Exception as e:
    print(f"加载模型时出错: {e}")
    exit()

# --- 2. 加载预处理好的样本 ---
print("正在加载样本文件...")
try:
    adv_samples_fpath = pathlib.Path("target_model/mdeepcorr/Gbase/Gmdeepcorr_advsamples_time0.1463_size0.0367.p")
    with open(adv_samples_fpath, "rb") as fp:
        data = pickle.load(fp)
        # 假设我们使用扰动后的样本进行评估
        total_adv_samples = data["original_samples"]
    print(f"样本加载成功，总共 {len(total_adv_samples)} 个样本。")
except Exception as e:
    print(f"加载样本时出错: {e}")
    exit()

adv_samples_subset = total_adv_samples[:N_SAMPLES]
print(f"已选取前 {N_SAMPLES} 个样本进行配对评估。")

# --- 3. 计算 m-DeepCorr 分数矩阵 ---
score_matrix = np.zeros((N_SAMPLES, N_SAMPLES))
print(f"\n开始计算 {N_SAMPLES}x{N_SAMPLES} 的 m-DeepCorr 分数矩阵...")
print(f"第一阶段筛选阈值 (Threshold for DC100) = {THRESHOLD_DC100}")

with torch.no_grad():
    for i in tqdm(range(N_SAMPLES), desc="Processing 'here' streams"):
        for j in range(N_SAMPLES):
            here_sample = adv_samples_subset[i].to(device)
            there_sample = adv_samples_subset[j].to(device)
            
            # --- 阶段 1: 使用 DC100 进行初步筛选 ---
            input_tensor_100 = format_deepcorr_input_from_samples(here_sample, there_sample, FLOW_SIZE_DC100)
            output_raw_100 = model_dc100(input_tensor_100, dropout=0.0)
            score_100 = torch.sigmoid(output_raw_100).item()
            # --- 阶段 2: 根据阈值决定是否使用 DC700 ---
            if score_100 >= THRESHOLD_DC100:
                # 如果分数高于阈值，则使用更精确的DC700模型评估
                input_tensor_700 = format_deepcorr_input_from_samples(here_sample, there_sample, FLOW_SIZE_DC700)
                output_raw_700 = model_dc700(input_tensor_700, dropout=0.0)
                score_700 = torch.sigmoid(output_raw_700).item()
                final_score = output_raw_700
            else:
                # 如果分数低于阈值，则直接使用DC100的低分作为最终结果
                final_score = output_raw_100
                
            score_matrix[i, j] = final_score

# --- 4. 打印并保存结果 ---
print("\n" + "="*50)
# print(f"--- m-DeepCorr Sigmoid 分数矩阵 ({N_SAMPLES}x{N_SAMPLES}) ---")
print("行: 'here' 部分来源, 列: 'there' 部分来源")
print("="*50)

np.set_printoptions(precision=4, suppress=True, linewidth=120)
print(score_matrix)

# 确保保存结果的目录存在
output_dir = pathlib.Path("plotwork")
output_dir.mkdir(exist_ok=True)

# 保存矩阵数据
matrix_path = output_dir / "mdeepcorr_score_matrix.p"
with open(matrix_path, "wb") as fp:
    pickle.dump(score_matrix, fp)
print(f"\n分数矩阵已保存到 '{matrix_path}'")

