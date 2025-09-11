import numpy as np
from sklearn.metrics import precision_score, accuracy_score, f1_score
import pickle
import pathlib
from tqdm import tqdm

def calculate_metrics_at_fpr(all_labels, all_output, target_fpr=0.0001):
    """
    在给定的FPR下，计算Precision, Accuracy, 和 F1-Score。

    Args:
        all_labels (np.array): 真实的标签 (0或1)。
        all_output (np.array): 模型的原始输出分数 (例如，概率值)。
        target_fpr (float): 目标假阳性率 (FPR)。

    Returns:
        dict: 包含Precision, Accuracy, F1-Score以及在该FPR下的阈值和TPR。
        None: 如果找不到满足目标FPR的阈值。
    """
    # 将标签和输出配对，并按输出分数降序排序
    # 这样我们可以通过从上到下移动阈值来模拟ROC曲线的计算过程
    desc_score_indices = np.argsort(all_output)[::-1]
    all_labels_sorted = all_labels[desc_score_indices]
    
    # 计算总的正样本和负样本数
    P = np.sum(all_labels == 1)
    N = np.sum(all_labels == 0)

    if P == 0 or N == 0:
        print("错误：标签中只包含一个类别，无法计算FPR/TPR。")
        return None

    # 累积计算TP和FP
    tp_cumulative = np.cumsum(all_labels_sorted == 1)
    fp_cumulative = np.cumsum(all_labels_sorted == 0)

    # 计算每个阈值下的TPR和FPR
    tpr_array = tp_cumulative / P
    fpr_array = fp_cumulative / N

    # 寻找第一个FPR大于等于目标FPR的位置
    candidate_indices = np.where(fpr_array >= target_fpr)[0]

    if len(candidate_indices) == 0:
        print(f"警告：在所有阈值下，FPR都未能达到 {target_fpr}。")
        print(f"最大FPR为: {fpr_array[-1]:.6f}。请考虑一个更低的目标FPR。")
        return None
    
    # 选择最接近目标FPR的那个索引
    best_index = candidate_indices[0]
    
    # 得到该点的阈值、FPR和TPR
    threshold = all_output[desc_score_indices[best_index]]
    actual_fpr = fpr_array[best_index]
    actual_tpr = tpr_array[best_index]

    # 根据这个阈值生成最终的预测结果
    all_predictions = (all_output >= threshold).astype(int)

    # 使用sklearn计算各项指标
    precision = precision_score(all_labels, all_predictions)
    # accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)

    return {
        "target_fpr": target_fpr,
        "actual_fpr": actual_fpr,
        "threshold": threshold,
        "tpr_at_fpr": actual_tpr,
        "precision": precision,
        # "accuracy": accuracy,
        "f1_score": f1
    }


#--------------------DeepCorr300--------------------    

print("========DeepCorr300========:")
result_fpath1 = pathlib.Path("baseline/AJSMA/deepcorr/AJSMAtest_index300_result.p")
result_fpath2 = pathlib.Path("baseline/BLANKET/deepcorr/BLANKETtest_index300_result.p")
result_fpath3 = pathlib.Path("baseline/CW/deepcorr/CWtest_index300_result.p")
result_fpath4 = pathlib.Path("baseline/I_FGSM/deepcorr/IFGSMtest_index300_result.p")
result_fpath5 = pathlib.Path("baseline/PGD/deepcorr/PGDtest_index300_result.p")
result_fpath6 = pathlib.Path("target_model/deepcorr/deepcorr300/base/test_index300_result.p")
result_fpath7 = pathlib.Path("target_model/deepcorr/deepcorr300/Gbase/Gtest_index300_result.p")
result_fpath8 = pathlib.Path("baseline/RM/deepcorr/RMtest_index300_result.p")

for result_fpath in [result_fpath1, result_fpath2, result_fpath3, result_fpath4,
                     result_fpath5, result_fpath6, result_fpath7,result_fpath8]:
    if(result_fpath.name == "RMtest_index300_result.p"):
        with open(result_fpath, 'rb') as fp:
            all_outputs, all_labels,_,_ = pickle.load(fp)
    else:
        with open(result_fpath, 'rb') as fp:
            all_outputs, all_labels = pickle.load(fp)
    metrics = calculate_metrics_at_fpr(all_labels, all_outputs, target_fpr=0.0001)
    print(f"{result_fpath.name}:")
    if metrics:
        print("在目标FPR下的性能指标:")
        for key, value in metrics.items():
            print(f"{key:<15}: {value:.6f}")
        print("-" * 30)
        
#--------------------DeepCorr300-------------------- 

#--------------------mDeepCorr---------------------- 
print("========mDeepCorr========:")
result_fpath1 = pathlib.Path("baseline/AJSMA/mdeepcorr/AJSMAmdeepcorr_threshDC100_0.0100.p")
result_fpath2 = pathlib.Path("baseline/BLANKET/mdeepcorr/BLANKETmdeepcorr_threshDC100_0.0100.p")
result_fpath3 = pathlib.Path("baseline/CW/mdeepcorr/CWmdeepcorr_threshDC100_0.0100.p")
result_fpath4 = pathlib.Path("baseline/I_FGSM/mdeepcorr/IFGSMmdeepcorr_threshDC100_0.0100.p")
result_fpath5 = pathlib.Path("baseline/PGD/mdeepcorr/PGDmdeepcorr_threshDC100_0.0100.p")
result_fpath6 = pathlib.Path("target_model/mdeepcorr/base/mdeepcorr_threshDC100_0.0100.p")
result_fpath7 = pathlib.Path("target_model/mdeepcorr/Gbase/Gmdeepcorr_threshDC100_0.0100.p")
result_fpath8 = pathlib.Path("baseline/RM/mdeepcorr/RMmdeepcorr_threshDC100_0.0100.p")

for result_fpath in [result_fpath1, result_fpath2, result_fpath3, result_fpath4,
                     result_fpath5, result_fpath6, result_fpath7,result_fpath8]:
    with open(result_fpath, 'rb') as fp:
            data = pickle.load(fp)
    all_outputs = data["all_outputs"]
    all_labels = data["all_labels"]
    metrics = calculate_metrics_at_fpr(all_labels, all_outputs, target_fpr=0.0001)
    print(f"{result_fpath.name}:")
    if metrics:
        print("在目标FPR下的性能指标:")
        for key, value in metrics.items():
            print(f"{key:<15}: {value:.6f}")
        print("-" * 30)
        
# # #--------------------mDeepCorr-----------------------
# #--------------------DeepCoFFEA---------------------- 


def deepcoffea_calculate_metrics_at_fpr(corr_matrix,threshhold_start,threshhold_end,n_wins=11, vote_thr=9,target_fpr=0.0001):
    n_test = corr_matrix.shape[0] // n_wins
    total_negatives = n_test * n_test - n_test
    total_positives = n_test
    for eta in np.arange(threshhold_start, threshhold_end, -0.001):
        votes = np.zeros((n_test, n_test), dtype=np.int64)
        for wi in range(0, n_wins):
            corr_matrix_win = corr_matrix[n_test*wi:n_test*(wi + 1), n_test*wi:n_test*(wi + 1)]

            for i in range(n_test):
                for j in range(n_test):
                    if corr_matrix_win[i,j] >= eta:
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
        
        # --- 核心改进部分 ---
        # 1. 计算当前阈值下的FPR
        current_fpr = fp / total_negatives if total_negatives > 0 else 0.0
        # 2. 检查是否达到目标
        if current_fpr >= target_fpr:

            # 3. 计算并返回所有指标
            tpr = tp / total_positives if total_positives > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            # accuracy = (tp + tn) / (total_positives + total_negatives)
            recall = tpr # Recall 和 TPR 是同一个指标
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            return {
                "target_fpr": target_fpr,
                "actual_fpr_found": current_fpr,
                "threshold_eta_used": eta,
                "tpr_at_fpr": tpr,
                "precision": precision,
                # "accuracy": accuracy,
                "f1_score": f1_score,
                # "TP": tp, "FP": fp, "TN": tn, "FN": fn
            }

print("========DeepCoFFEA========:")
result_fpath1 = pathlib.Path("baseline/AJSMA/deepcoffea/AJSMAcorrmatrix.npz")
result_fpath2 = pathlib.Path("baseline/BLANKET/deepcoffea/BLANKETcorrmatrix.npz")
result_fpath3 = pathlib.Path("baseline/CW/deepcoffea/CWcorrmatrix.npz")
result_fpath4 = pathlib.Path("baseline/I_FGSM/deepcoffea/IFGSMcorrmatrix.npz")
result_fpath5 = pathlib.Path("baseline/PGD/deepcoffea/PGDcorrmatrix.npz")
result_fpath6 = pathlib.Path("target_model/deepcoffea/code/corrmatrix_sim0.7404.npz")
result_fpath7 = pathlib.Path("target_model/deepcoffea/code/Gcorrmatrix_time0.1398_size0.0038_sim0.1945.npz")
result_fpath8 = pathlib.Path("baseline/RM/deepcoffea/RMcorrmatrix.npz")

reuslt_metrics = pathlib.Path("performance/deepcoffea_metrics0.0001.p")
if reuslt_metrics.exists():
    with open(reuslt_metrics, 'rb') as fp:
        all_metrics = pickle.load(fp)
    for result_name,metrics in all_metrics:
        print(f"{result_name}:")
        if metrics:
            print("在目标FPR下的性能指标:")
            for key, value in metrics.items():
                print(f"{key:<20}: {value:.6f}")
            print("-" * 30)
else:
    # 
    deepcoffea_perfomance = []
    for result_fpath in [result_fpath1, result_fpath2,result_fpath3, result_fpath4,
                        result_fpath5, result_fpath6, result_fpath7,result_fpath8]:
        loaded = np.load(result_fpath,allow_pickle=True)
        corr_matrix = loaded['corr_matrix']
        if(result_fpath.name == "corrmatrix_sim0.7404.npz" or result_fpath.name == "RMcorrmatrix.npz"):
            metrics = deepcoffea_calculate_metrics_at_fpr(corr_matrix,threshhold_start=0.5,threshhold_end=0.3,n_wins=11, vote_thr=9,target_fpr=0.0001)
        elif(result_fpath.name == "AJSMAcorrmatrix.npz" or result_fpath.name == "PGDcorrmatrix.npz" or result_fpath.name == "BLANKETcorrmatrix.npz" or result_fpath.name == "CWcorrmatrix.npz" or result_fpath.name == "IFGSMcorrmatrix.npz"):
            metrics = deepcoffea_calculate_metrics_at_fpr(corr_matrix,threshhold_start=0.4,threshhold_end=0.2,n_wins=11, vote_thr=9,target_fpr=0.0001)
        elif(result_fpath.name == "CWcorrmatrix.npz" or result_fpath.name == "Gcorrmatrix_time0.1398_size0.0038_sim0.1945.npz"):
            metrics = deepcoffea_calculate_metrics_at_fpr(corr_matrix,threshhold_start=0.3,threshhold_end=0.1,n_wins=11, vote_thr=9,target_fpr=0.0001)
        else:
            metrics = deepcoffea_calculate_metrics_at_fpr(corr_matrix,threshhold_start=1,threshhold_end=0,n_wins=11, vote_thr=9,target_fpr=0.0001)

        print(f"{result_fpath.name}:")
        if metrics:
            print("在目标FPR下的性能指标:")
            for key, value in metrics.items():
                print(f"{key:<20}: {value:.6f}")
            print("-" * 30)
            pickle_data = (result_fpath.name,metrics)
            deepcoffea_perfomance.append(pickle_data)
    # with open(reuslt_metrics, 'wb') as fp:
    #     pickle.dump(deepcoffea_perfomance, fp)


