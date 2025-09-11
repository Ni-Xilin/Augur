import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import pathlib
import os
import random
from target_model.Deepcorr700 import Model as Deepcorr700Model
from target_model.Deepcorr100 import Model as Deepcorr100Model
from tqdm import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def pad_or_truncate(data_list, target_length):
    """
    将列表填充零或截断到目标长度。
    """
    current_length = len(data_list)
    if current_length >= target_length:
        return data_list[:target_length]
    else:
        padding_needed = target_length - current_length
        return data_list + [0.0] * padding_needed

def evaluate_model(model, data_loader, device):
    """
    评估模型并返回所有预测概率和真实标签。
    """
    model.eval()
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)
            outputs_raw = model(data, dropout=0.0)
            outputs_prob = torch.sigmoid(outputs_raw).cpu().numpy().flatten()
            
            all_outputs.extend(outputs_prob)
            all_labels.extend(labels.cpu().numpy().flatten())
    
    all_outputs = np.array(all_outputs)
    all_labels = np.array(all_labels)
    
    # 返回所有输出和标签，而不是在这里计算TP/FP，因为后续筛选需要原始输出
    return all_outputs, all_labels

def main():
    # 数据集加载
    dataset_raw = []
    all_runs = {'8872':'192.168.122.117','8802':'192.168.122.117','8873':'192.168.122.67','8803':'192.168.122.67',
                '8874':'192.168.122.113','8804':'192.168.122.113','8875':'192.168.122.120',
                '8876':'192.168.122.30','8877':'192.168.122.208','8878':'192.168.122.58'}

    for name in all_runs:
        dataset_path = f'target_model/deepcorr/dataset/{name}_tordata300.pickle'
        # 仅当文件存在时加载，避免警告信息反复打印
        if os.path.exists(dataset_path):
            dataset_raw += pickle.load(open(dataset_path, 'rb'))
        else:
            print(f"警告: 数据集文件 {dataset_path} 未找到，跳过加载。")
    print(f"Loaded {len(dataset_raw)} raw samples.") 

    negetive_samples = 199 
    batch_size = 64

    # 选择 GPU 设备
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu") # 可以选择其他cuda设备，如 cuda:1，cuda:2
    print(f"Using device: {device}")

    # --- 加载 DC100 模型 ---
    print("Loading DC100 model...")
    model_dc100 = Deepcorr100Model().to(device)
    try:
        model_dc100.load_state_dict(torch.load('target_model/deepcorr/deepcorr100/tor_199_epoch10_acc0.66.pth', map_location=device))
    except FileNotFoundError:
        print("错误：未找到DC100模型文件。请检查路径 'target_model/deepcorr/deepcorr100/tor100_199_epochXX_accYY.pth' 并替换为实际文件名。")
        return
    print("DC100 model loaded.")

    # --- 加载 DC700 模型 ---
    print("Loading DC700 model...")
    model_dc700 = Deepcorr700Model().to(device)
    try:
        model_dc700.load_state_dict(torch.load('target_model/deepcorr/deepcorr700/tor700_199_epoch11_acc0.88.pth', map_location=device)) 
    except FileNotFoundError:
        print("错误：未找到DC700模型文件。请检查路径 'target_model/deepcorr/deepcorr700/tor700_199_epochXX_accYY.pth' 并替换为实际文件名。")
        return
    print("DC700 model loaded.")

    # 加载测试索引
    test_index_path = 'target_model/deepcorr/test_index.pickle'
    try:
        with open(test_index_path, 'rb') as f:
            full_test_index = pickle.load(f)[:1000] # 使用与 evalDeepcorr.py 相同的1000个测试样本 [cite: 1]
    except FileNotFoundError:
        print(f"错误: 测试索引文件 {test_index_path} 未找到。请检查路径。")
        return
    print(f"Loaded {len(full_test_index)} test indices.") 

    # --- 阶段 1: 使用 DC100 进行初步筛选 ---
    print("\nPhase 1: Evaluating with DC100 (flow_size=100)...")

    # 用于存储第一阶段筛选出的，且包含原始流量索引信息的列表
    filtered_l2s_for_stage2_list = []
    filtered_labels_for_stage2_list = [] # 这些是第二阶段的真实标签
    
    # 重新构建 generate_data_with_indices 函数以获取原始流索引
    def generate_data_with_indices(dataset_raw, test_index_list, flow_size, negetive_samples_per_positive):
        l2s_all = []
        labels_all = []
        original_pair_indices = [] # 存储 (original_entry_idx, original_exit_idx)
        
        # 处理数据集以适应当前的 flow_size
        dataset_processed_for_gen = []
        for sample in dataset_raw:
            processed_sample = {
                'here': [
                    {'<-': pad_or_truncate(sample['here'][0]['<-'], flow_size), '->': pad_or_truncate(sample['here'][0]['->'], flow_size)},
                    {'<-': pad_or_truncate(sample['here'][1]['<-'], flow_size), '->': pad_or_truncate(sample['here'][1]['->'], flow_size)}
                ],
                'there': [
                    {'<-': pad_or_truncate(sample['there'][0]['<-'], flow_size), '->': pad_or_truncate(sample['there'][0]['->'], flow_size)},
                    {'<-': pad_or_truncate(sample['there'][1]['<-'], flow_size), '->': pad_or_truncate(sample['there'][1]['->'], flow_size)}
                ]
            }
            dataset_processed_for_gen.append(processed_sample)


        random_test_pool = list(test_index_list) # 用于负样本选择的池
        
        for i_entry_idx in tqdm(test_index_list):
            # 正样本
            current_l2s = torch.zeros((1, 8, flow_size))
            current_l2s[0, 0,:] = torch.tensor(dataset_processed_for_gen[i_entry_idx]['here'][0]['<-']) * 1000.0  
            current_l2s[0, 1,:] = torch.tensor(dataset_processed_for_gen[i_entry_idx]['there'][0]['->']) * 1000.0  
            current_l2s[0, 2,:] = torch.tensor(dataset_processed_for_gen[i_entry_idx]['there'][0]['<-']) * 1000.0  
            current_l2s[0, 3,:] = torch.tensor(dataset_processed_for_gen[i_entry_idx]['here'][0]['->']) * 1000.0  
            current_l2s[0, 4,:] = torch.tensor(dataset_processed_for_gen[i_entry_idx]['here'][1]['<-']) / 1000.0  
            current_l2s[0, 5,:] = torch.tensor(dataset_processed_for_gen[i_entry_idx]['there'][1]['->']) / 1000.0  
            current_l2s[0, 6,:] = torch.tensor(dataset_processed_for_gen[i_entry_idx]['there'][1]['<-']) / 1000.0  
            current_l2s[0, 7,:] = torch.tensor(dataset_processed_for_gen[i_entry_idx]['here'][1]['->']) / 1000.0  
            
            l2s_all.append(current_l2s)
            labels_all.append(1)
            original_pair_indices.append((i_entry_idx, i_entry_idx)) # 正样本，入口和出口是同一个原始流

            # 负样本
            m = 0
            # 每次 shuffle 不同的随机种子，确保负样本选择的多样性
            np.random.seed(i_entry_idx) # 使用入口流索引作为随机种子，使得每个入口流的负样本选择固定
            np.random.shuffle(random_test_pool)
            
            for i_exit_idx in random_test_pool:
                if i_exit_idx == i_entry_idx or m >= negetive_samples_per_positive:
                    continue

                current_l2s = torch.zeros((1, 8, flow_size))
                current_l2s[0, 0,:] = torch.tensor(dataset_processed_for_gen[i_exit_idx]['here'][0]['<-']) * 1000.0  
                current_l2s[0, 1,:] = torch.tensor(dataset_processed_for_gen[i_entry_idx]['there'][0]['->']) * 1000.0  
                current_l2s[0, 2,:] = torch.tensor(dataset_processed_for_gen[i_entry_idx]['there'][0]['<-']) * 1000.0  
                current_l2s[0, 3,:] = torch.tensor(dataset_processed_for_gen[i_exit_idx]['here'][0]['->']) * 1000.0  
                current_l2s[0, 4,:] = torch.tensor(dataset_processed_for_gen[i_exit_idx]['here'][1]['<-']) / 1000.0  
                current_l2s[0, 5,:] = torch.tensor(dataset_processed_for_gen[i_entry_idx]['there'][1]['->']) / 1000.0  
                current_l2s[0, 6,:] = torch.tensor(dataset_processed_for_gen[i_entry_idx]['there'][1]['<-']) / 1000.0  
                current_l2s[0, 7,:] = torch.tensor(dataset_processed_for_gen[i_exit_idx]['here'][1]['->']) / 1000.0  
                
                l2s_all.append(current_l2s)
                labels_all.append(0)
                original_pair_indices.append((i_entry_idx, i_exit_idx)) # 负样本

                m += 1
        
        return torch.stack(l2s_all).float(), torch.tensor(labels_all).float(), original_pair_indices
    
    
    threshold_dc100 = 0.01 # 示例阈值，实际应用中需要根据模型性能曲线调整
    # 阶段 1 数据生成（带索引）
    test_l2s_100_indexed, test_labels_100_indexed, original_pairs_100 = generate_data_with_indices(dataset_raw, full_test_index, 100, negetive_samples)
    
    # 创建DataLoader for Phase 1
    test_dataset_100_indexed = TensorDataset(test_l2s_100_indexed, test_labels_100_indexed)
    test_loader_100_indexed = DataLoader(test_dataset_100_indexed, batch_size=batch_size, shuffle=False, drop_last=False) 

    # 执行阶段 1 评估
    outputs_dc100, labels_dc100 = evaluate_model(model_dc100, test_loader_100_indexed, device)

    # 筛选出 DC100 认为相关的流量对的全局索引
    stage1_correlated_global_indices = np.where(outputs_dc100 > threshold_dc100)[0]
    
    # 从这些全局索引中提取原始的 (entry_idx, exit_idx) 对及其真实标签，用于第二阶段
    filtered_original_pairs_for_stage2_data = [] # 存储 (original_entry_idx, original_exit_idx, true_label)
    for global_idx in stage1_correlated_global_indices:
        entry_idx, exit_idx = original_pairs_100[global_idx]
        true_label = labels_dc100[global_idx]
        filtered_original_pairs_for_stage2_data.append((entry_idx, exit_idx, true_label))

    if not filtered_original_pairs_for_stage2_data:
        print("No correlated pairs found in Phase 1. M-DeepCorr analysis complete.")
        return

    # 构建第二阶段的数据集
    filtered_l2s_for_stage2_list = []
    filtered_labels_for_stage2_list = []

    for entry_idx, exit_idx, true_label in filtered_original_pairs_for_stage2_data:
        current_l2s_700 = torch.zeros((1, 8, 700))
        entry_flow_data = dataset_raw[entry_idx] 
        exit_flow_data = dataset_raw[exit_idx] 

        current_l2s_700[0, 0,:] = torch.tensor(pad_or_truncate(entry_flow_data['here'][0]['<-'], 700)) * 1000.0  
        current_l2s_700[0, 1,:] = torch.tensor(pad_or_truncate(exit_flow_data['there'][0]['->'], 700)) * 1000.0  
        current_l2s_700[0, 2,:] = torch.tensor(pad_or_truncate(exit_flow_data['there'][0]['<-'], 700)) * 1000.0  
        current_l2s_700[0, 3,:] = torch.tensor(pad_or_truncate(entry_flow_data['here'][0]['->'], 700)) * 1000.0  
        current_l2s_700[0, 4,:] = torch.tensor(pad_or_truncate(entry_flow_data['here'][1]['<-'], 700)) / 1000.0  
        current_l2s_700[0, 5,:] = torch.tensor(pad_or_truncate(exit_flow_data['there'][1]['->'], 700)) / 1000.0  
        current_l2s_700[0, 6,:] = torch.tensor(pad_or_truncate(exit_flow_data['there'][1]['<-'], 700)) / 1000.0  
        current_l2s_700[0, 7,:] = torch.tensor(pad_or_truncate(entry_flow_data['here'][1]['->'], 700)) / 1000.0  

        filtered_l2s_for_stage2_list.append(current_l2s_700.squeeze(0)) # 移除 batch 维度
        filtered_labels_for_stage2_list.append(true_label)
    
    filtered_l2s_for_stage2 = torch.stack(filtered_l2s_for_stage2_list).unsqueeze(1) # 添加 channel 维度
    filtered_labels_for_stage2 = torch.tensor(filtered_labels_for_stage2_list)

    # 创建 DataLoader for Phase 2
    test_dataset_700 = TensorDataset(filtered_l2s_for_stage2.float(), filtered_labels_for_stage2.float())
    test_loader_700 = DataLoader(test_dataset_700, batch_size=batch_size, shuffle=False, drop_last=False)
    
    # 这里的阈值需要根据你在DC700上的TPR/FPR曲线来选择，以达到最终的性能目标
    # threshold_dc700 = 0.01 # 示例阈值，实际应用中需要根据模型性能曲线调整

    print(f"\nPhase 2: Evaluating with DC700 (flow_size=700) on {len(filtered_l2s_for_stage2_list)} filtered pairs...")
    outputs_dc700, labels_dc700 = evaluate_model(model_dc700, test_loader_700, device)
    
    final_outputs = outputs_dc100
    final_labels = labels_dc100
    for i, global_idx in enumerate(stage1_correlated_global_indices):
        final_outputs[global_idx] = outputs_dc700[i]
    # --- 最终结果统计 ---
    # `outputs_dc700` 和 `labels_dc700_true` 已经包含了所有经过阶段1筛选后的样本的预测和真实标签
    
    # final_tp_count = np.sum((outputs_dc700 > threshold_dc700) & (labels_dc700_true == 1))
    # final_fp_count = np.sum((outputs_dc700 > threshold_dc700) & (labels_dc700_true == 0))
    # final_tn_count = np.sum((outputs_dc700 <= threshold_dc700) & (labels_dc700_true == 0))
    # final_fn_count = np.sum((outputs_dc700 <= threshold_dc700) & (labels_dc700_true == 1))

    # 计算原始完整测试集中的总真实正样本数和总真实负样本数
    # total_true_positives_in_full_test_set = np.sum(test_labels_100_indexed.numpy() == 1)
    # total_true_negatives_in_full_test_set = np.sum(test_labels_100_indexed.numpy() == 0)

    # 最终的 TP Rate 和 FP Rate
    # final_tpr = final_tp_count / total_true_positives_in_full_test_set if total_true_positives_in_full_test_set > 0 else 0
    # final_fpr = final_fp_count / total_true_negatives_in_full_test_set if total_true_negatives_in_full_test_set > 0 else 0

    # print(f"\n--- M-DeepCorr Final Results (Thresholds: DC100={threshold_dc100}, DC700={threshold_dc700}) ---")
    # print(f"Total True Positives (in full test set): {total_true_positives_in_full_test_set}")
    # print(f"Total True Negatives (in full test set): {total_true_negatives_in_full_test_set}")
    # print(f"Number of pairs passed Phase 1: {len(filtered_original_pairs_for_stage2_data)}")
    # print(f"Final True Positive Count: {final_tp_count}")
    # print(f"Final False Positive Count: {final_fp_count}")
    # print(f"Final True Negative Count: {final_tn_count}")
    # print(f"Final False Negative Count: {final_fn_count}")
    # print(f"Final True Positive Rate (TPR): {final_tpr:.4f}")
    # print(f"Final False Positive Rate (FPR): {final_fpr:.4f}")
    
    output_filename = f"mdeepcorr_threshDC100_{threshold_dc100:.4f}.p"
    result_fpath = pathlib.Path("target_model/mdeepcorr") / output_filename
    results_to_save = {
        "all_outputs": final_outputs,
        "all_labels": final_labels,
        }

    with open(result_fpath, "wb") as fp:
        pickle.dump(results_to_save, fp)
    print(f"Evaluation results saved to {result_fpath}")

if __name__ == "__main__":
    main()