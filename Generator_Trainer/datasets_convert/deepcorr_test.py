# -*- coding: utf-8 -*-
# 用于测试跨数据集的可迁移性(结果是deepcorr跨数据集关联性本身就不行，因此失败)
import numpy as np
import tqdm
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset
import pathlib
import torch.nn as nn
import torch.nn.functional as F

from target_model.Deepcorr300 import Model as Net

def pad_or_truncate(data, length):
    """
    将列表用0填充或截断到指定长度。
    """
    if len(data) >= length:
        return data[:length]
    else:
        return data + [0.0] * (length - len(data))

def generate_data_from_converted(dataset, flow_size):
    """
    从转换后的数据集（一个列表的字典）生成正负测试对。
    """
    negetive_samples = 199
    # 测试索引现在就是数据集的自然索引
    test_index = list(range(len(dataset)))

    num_pairs = len(test_index) * (negetive_samples + 1)
    l2s_test = torch.zeros((num_pairs, 1, 8, flow_size)) 
    labels_test = torch.zeros(num_pairs) 

    current_pair_idx = 0
    random_test_pool = [] + test_index

    for i in tqdm.tqdm(test_index, desc="正在生成测试对"):
        # --- 生成负样本 ---
        m = 0
        np.random.shuffle(random_test_pool)
        for idx in random_test_pool:
            if idx == i or m >= negetive_samples:
                continue
            m += 1
            
            # 'here' 部分来自随机样本(idx), 'there' 部分来自当前样本(i)
            l2s_test[current_pair_idx, 0, 0,:] = torch.tensor(pad_or_truncate(dataset[idx]['here'][0]['<-'], flow_size)) * 1000.0  
            l2s_test[current_pair_idx, 0, 3,:] = torch.tensor(pad_or_truncate(dataset[idx]['here'][0]['->'], flow_size)) * 1000.0  
            l2s_test[current_pair_idx, 0, 4,:] = torch.tensor(pad_or_truncate(dataset[idx]['here'][1]['<-'], flow_size)) / 1000.0  
            l2s_test[current_pair_idx, 0, 7,:] = torch.tensor(pad_or_truncate(dataset[idx]['here'][1]['->'], flow_size)) / 1000.0  
            
            l2s_test[current_pair_idx, 0, 1,:] = torch.tensor(pad_or_truncate(dataset[i]['there'][0]['->'], flow_size)) * 1000.0  
            l2s_test[current_pair_idx, 0, 2,:] = torch.tensor(pad_or_truncate(dataset[i]['there'][0]['<-'], flow_size)) * 1000.0  
            l2s_test[current_pair_idx, 0, 5,:] = torch.tensor(pad_or_truncate(dataset[i]['there'][1]['->'], flow_size)) / 1000.0  
            l2s_test[current_pair_idx, 0, 6,:] = torch.tensor(pad_or_truncate(dataset[i]['there'][1]['<-'], flow_size)) / 1000.0  
            
            labels_test[current_pair_idx] = 0
            current_pair_idx += 1

        # --- 生成正样本 ---
        l2s_test[current_pair_idx, 0, 0,:] = torch.tensor(pad_or_truncate(dataset[i]['here'][0]['<-'], flow_size)) * 1000.0  
        l2s_test[current_pair_idx, 0, 1,:] = torch.tensor(pad_or_truncate(dataset[i]['there'][0]['->'], flow_size)) * 1000.0  
        l2s_test[current_pair_idx, 0, 2,:] = torch.tensor(pad_or_truncate(dataset[i]['there'][0]['<-'], flow_size)) * 1000.0  
        l2s_test[current_pair_idx, 0, 3,:] = torch.tensor(pad_or_truncate(dataset[i]['here'][0]['->'], flow_size)) * 1000.0  
        l2s_test[current_pair_idx, 0, 4,:] = torch.tensor(pad_or_truncate(dataset[i]['here'][1]['<-'], flow_size)) / 1000.0  
        l2s_test[current_pair_idx, 0, 5,:] = torch.tensor(pad_or_truncate(dataset[i]['there'][1]['->'], flow_size)) / 1000.0  
        l2s_test[current_pair_idx, 0, 6,:] = torch.tensor(pad_or_truncate(dataset[i]['there'][1]['<-'], flow_size)) / 1000.0  
        l2s_test[current_pair_idx, 0, 7,:] = torch.tensor(pad_or_truncate(dataset[i]['here'][1]['->'], flow_size)) / 1000.0  
        
        labels_test[current_pair_idx] = 1
        current_pair_idx += 1
        
    return l2s_test, labels_test

# --- 主程序入口 ---
if __name__ == '__main__':
    # --- 配置参数 ---
    FLOW_SIZE = 300
    BATCH_SIZE = 256
    DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = 'target_model/deepcorr/deepcorr300/tor_199_epoch23_acc0.82dict.pth'
    INPUT_DATA_PATH = 'datasets_convert/deepcorr_test_from_deepcoffea.pickle'
    OUTPUT_RESULT_PATH = 'datasets_convert/test_converted_data_deepcorr300_result.p'

    print(f"使用设备: {DEVICE}")

    # --- 加载模型 ---
    model = Net().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        print(f"成功从 {MODEL_PATH} 加载模型。")
    except Exception as e:
        print(f"加载模型时出错: {e}")
        exit()

    # --- 加载转换后的数据集 ---
    try:
        with open(INPUT_DATA_PATH, 'rb') as f:
            converted_dataset = pickle.load(f)
        print(f"成功加载 {len(converted_dataset)} 条转换后的样本。")
    except FileNotFoundError:
        print(f"错误: 找不到转换后的数据文件 {INPUT_DATA_PATH}。")
        exit()

    # --- 生成测试数据并运行评估 ---
    all_outputs = []
    all_labels = []
    result_fpath = pathlib.Path(OUTPUT_RESULT_PATH)

    if result_fpath.exists():
        print(f"发现已存在的结果文件 {result_fpath}，正在加载...")
        with open(result_fpath, "rb") as fp:
            all_outputs, all_labels = pickle.load(fp)
    else:
        print("正在生成测试对...")
        test_l2s, test_labels = generate_data_from_converted(converted_dataset, FLOW_SIZE)
        test_dataset = TensorDataset(test_l2s.float(), test_labels.float())
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
        
        print("开始在转换后的数据集上进行评估...")
        with torch.no_grad():
            for data, labels in tqdm.tqdm(test_loader, desc="评估进度"):
                data, labels = data.to(DEVICE), labels.to(DEVICE)
                
                outputs_raw = model(data, dropout=0.0)
                outputs_prob = torch.sigmoid(outputs_raw).cpu().numpy()
                
                all_outputs.extend(outputs_prob.flatten())
                all_labels.extend(labels.cpu().numpy().flatten())

        all_outputs = np.array(all_outputs)
        all_labels = np.array(all_labels)
        
        result_fpath.parent.mkdir(parents=True, exist_ok=True)
        with open(result_fpath, "wb") as fp:
            pickle.dump((all_outputs, all_labels), fp)
        print(f"评估完成，结果已保存至: {result_fpath}")

    print("\n评估脚本执行完毕。")

