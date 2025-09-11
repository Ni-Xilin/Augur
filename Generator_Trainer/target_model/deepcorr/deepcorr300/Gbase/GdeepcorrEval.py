import numpy as np
import tqdm
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset
import pathlib
import torch.nn as nn
import torch.nn.functional as F
from target_model.Deepcorr300 import Model as Net

import torch.optim as optim


def generate_data(total_adv_samples, flow_size):
    global negetive_samples
    dataset = total_adv_samples 
    test_index = list(range(len(dataset)))
    #测试集样本填充(all_samples*(negetive_samples+1),1,8,flow_size)
    index = 0
    random_test = [] + test_index
    l2s_test = torch.zeros((len(dataset) * (negetive_samples + 1),1, 8, flow_size)) 
    labels_test = torch.zeros((len(dataset) * (negetive_samples + 1))) 

    for i in test_index:
        if index % (negetive_samples + 1) != 0:
            print(index, len(random_test))
            raise ValueError("Index is not a multiple of (negetive_samples + 1)")
        m = 0

        np.random.shuffle(random_test)
        for idx in random_test:
            if idx == i or m > (negetive_samples - 1):
                continue

            m += 1
            l2s_test[index, 0, 0,:] = dataset[idx,0,0,:flow_size] 
            l2s_test[index, 0, 1,:] = dataset[i,0,1,:flow_size]
            l2s_test[index, 0, 2,:] = dataset[i,0,2,:flow_size] 
            l2s_test[index, 0, 3,:] = dataset[idx,0,3,:flow_size] 

            l2s_test[index, 0, 4,:] = dataset[idx,0,4,:flow_size]   
            l2s_test[index, 0, 5,:] = dataset[i,0,5,:flow_size] 
            l2s_test[index, 0, 6,:] = dataset[i,0,6,:flow_size]  
            l2s_test[index, 0, 7,:] = dataset[idx,0,7,:flow_size] 
            labels_test[index] = 0
            index += 1

        l2s_test[index, 0, 0,:] = dataset[i,0,0,:flow_size]  
        l2s_test[index, 0, 1,:] = dataset[i,0,1,:flow_size] 
        l2s_test[index, 0, 2,:] = dataset[i,0,2,:flow_size] 
        l2s_test[index, 0, 3,:] = dataset[i,0,3,:flow_size] 

        l2s_test[index, 0, 4,:] = dataset[i,0,4,:flow_size] 
        l2s_test[index, 0, 5,:] = dataset[i,0,5,:flow_size] 
        l2s_test[index, 0, 6,:] = dataset[i,0,6,:flow_size]  
        l2s_test[index, 0, 7,:] = dataset[i,0,7,:flow_size]  
        labels_test[index] = 1

        index += 1
    return l2s_test, labels_test


negetive_samples = 199

adv_samples_fpath = pathlib.Path("target_model/deepcorr/deepcorr300/Gbase/Gdeepcorr_advsamples_time0.0000_size0.1474.p")
with open(adv_samples_fpath, "rb") as fp:
    # Load the adversarial samples, original samples, and perturbations:包含了一千条扰动后的测试集数据
    data = pickle.load(fp)
    total_adv_sample = data["adv_samples"]

flow_size = 300

batch_size = 256
    
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:2" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
print (device)


model = Net().to(device)
LOADED=torch.load('target_model/deepcorr/deepcorr300/tor_199_epoch23_acc0.82dict.pth',map_location=device)
model.load_state_dict(LOADED)


# 1. 收集所有模型的预测概率和真实标签
all_outputs = []
all_labels = []
result_fpath = pathlib.Path("target_model/deepcorr/deepcorr300/Gbase/Gtest_index300_time0_result.p")
if False:
    with open(result_fpath, "rb") as fp:
        all_outputs,all_labels = pickle.load(fp)
else:
    model.eval() 
    with torch.no_grad():
        test_l2s, test_labels = generate_data(total_adv_sample, flow_size)
        test_dataset = TensorDataset(test_l2s.float(), test_labels.float())
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,drop_last=True)

        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            
            # 得到模型的原始输出（logits）
            outputs_raw = model(data, dropout=0.0)
            
            # 使用 sigmoid 转换为概率
            outputs_prob = torch.sigmoid(outputs_raw).cpu().numpy()
            
            # 收集概率和标签
            all_outputs.extend(outputs_prob.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    # 转换为numpy数组
    all_outputs = np.array(all_outputs)
    all_labels = np.array(all_labels)
    with open(result_fpath, "wb") as fp:
        pickle.dump((all_outputs,all_labels), fp)