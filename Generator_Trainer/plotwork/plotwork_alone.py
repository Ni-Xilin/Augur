from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pickle
import pathlib
import numpy as np

# fpr: False Positive Rate
# tpr: True Positive Rate
# thresholds: roc_curve使用的阈值
result_fpath = pathlib.Path("target_model/mdeepcorr/base/mdeepcorr_threshDC100_0.0100.p")
Gresult_fpath = pathlib.Path("target_model/mdeepcorr/Gbase/Gmdeepcorr_threshDC100_0.0100.p")


output_filename = "./plotwork/alone_roc1log.svg"

first_points =20
n_points = 60
fpr_list_mdeepcorr = []
tpr_list_mdeepcorr = []
for result_fpath in [result_fpath, Gresult_fpath]:
    with open(result_fpath, 'rb') as fp:
            data = pickle.load(fp)
    all_outputs = data["all_outputs"]
    all_labels = data["all_labels"]
    fpr, tpr, thresholds = roc_curve(all_labels, all_outputs)
    first_part_indices = np.linspace(0,200, first_points, dtype=int)
    # 从剩下的点中均匀采样
    remaining_indices = np.linspace(200, len(thresholds) - 1, n_points - first_points, dtype=int)
    # 合并两个部分的 indices
    threshold_indices = np.concatenate([first_part_indices, remaining_indices.astype(int)])
    # 计算 ROC 曲线
    fpr, tpr, thresholds = roc_curve(all_labels, all_outputs)
    # 根据选定的阈值点选取 fpr 和 tpr
    fpr_selected = fpr[threshold_indices]
    tpr_selected = tpr[threshold_indices]
    fpr_list_mdeepcorr.append(fpr_selected)
    tpr_list_mdeepcorr.append(tpr_selected)
    # roc_auc = auc(fpr, tpr)


result_fpath = pathlib.Path("target_model/deepcorr/deepcorr300/base/test_index300_result.p")
Gresult_fpath = pathlib.Path("target_model/deepcorr/deepcorr300/Gbase/Gtest_index300_result.p")

first_points =20
n_points = 50
fpr_list_deepcorr = []
tpr_list_deepcorr = []
for result_fpath in [result_fpath, Gresult_fpath]:
    if(result_fpath.name == "RMtest_index300_result.p"):
        with open(result_fpath, 'rb') as fp:
            all_outputs, all_labels,_,_ = pickle.load(fp)
    else:
        with open(result_fpath, 'rb') as fp:
            all_outputs, all_labels = pickle.load(fp)
    fpr, tpr, thresholds = roc_curve(all_labels, all_outputs)
    first_part_indices = np.linspace(0,100, first_points, dtype=int)
    # 从剩下的点中均匀采样
    remaining_indices = np.linspace(100, len(thresholds) - 1, n_points - first_points, dtype=int)
    # 合并两个部分的 indices
    threshold_indices = np.concatenate([first_part_indices, remaining_indices.astype(int)])
    # 计算 ROC 曲线
    fpr, tpr, thresholds = roc_curve(all_labels, all_outputs)
    # 根据选定的阈值点选取 fpr 和 tpr
    fpr_selected = fpr[threshold_indices]
    tpr_selected = tpr[threshold_indices]
    fpr_list_deepcorr.append(fpr_selected)
    tpr_list_deepcorr.append(tpr_selected)
    # roc_auc = auc(fpr, tpr)


result_fpath = pathlib.Path("target_model/deepcoffea/results/stats_corrmatrix/gthr.p")
Gresult_fpath = pathlib.Path("target_model/deepcoffea/results/stats_Gcorrmatrix/gthr.p")

n_points = 30
fpr_list_deepcoffea = []
tpr_list_deepcoffea = []
for result_fpath in [result_fpath, Gresult_fpath]:
    with open(result_fpath, 'rb') as fp:
        tpr, fpr,_ = pickle.load(fp)
    # first_part_indices = np.linspace(0,100, first_points, dtype=int)
    # # 从剩下的点中均匀采样
    # remaining_indices = np.linspace(100, len(tpr['d3_ws5_nw11']) - 1, n_points - first_points, dtype=int)
    # # 合并两个部分的 indices
    # threshold_indices = np.concatenate([first_part_indices, remaining_indices.astype(int)])
    # # 根据选定的阈值点选取 fpr 和 tpr
    fpr_selected = fpr['d3_ws5_nw11']
    tpr_selected = tpr['d3_ws5_nw11']
    fpr_list_deepcoffea.append(fpr_selected)
    tpr_list_deepcoffea.append(tpr_selected)
    # roc_auc = auc(fpr, tpr)

# 绘制 ROC 曲线
plt.rcParams.update({'font.size': 20})
plt.figure(figsize=(8, 6))
plt.xscale('log')
plt.plot(fpr_list_mdeepcorr[0], tpr_list_mdeepcorr[0], color='#1f77b4', lw=1.5, label=f'No-Perturbation with mDeepCorr', marker='o', linestyle='-', markersize=2.5)  
plt.plot(fpr_list_mdeepcorr[1], tpr_list_mdeepcorr[1], color='#1f77b4', lw=1.5, label=f'Augur with mDeepCorr', marker='s',linestyle='-', markersize=2)
plt.plot(fpr_list_deepcorr[0], tpr_list_deepcorr[0], color='#2ca02c', lw=1.5, label=f'No-Perturbation with DeepCorr', marker='o', linestyle='-', markersize=2.5) 
plt.plot(fpr_list_deepcorr[1], tpr_list_deepcorr[1], color='#2ca02c', lw=1.5, label=f'Augur with DeepCorr', marker='s', linestyle='-', markersize=2)
plt.plot(fpr_list_deepcoffea[0], tpr_list_deepcoffea[0], color='#FF0000', lw=1.5, label=f'No-Perturbation with DeepCoFFEA', marker='o', linestyle='-', markersize=2.5)  
plt.plot(fpr_list_deepcoffea[1], tpr_list_deepcoffea[1], color='#FF0000', lw=1.5, label=f'Augur with DeepCoFFEA', marker='s', linestyle='-', markersize=2)






plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')  # 绘制对角线
plt.xlim([0.0001, 1])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontsize =20)
plt.ylabel('True Positive Rate',fontsize = 20)
# plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='upper right',fontsize=12)

plt.grid(True)
plt.savefig(output_filename, format='svg',transparent=True)
print(f"ROC curve plot saved to {output_filename}")
plt.show()

# print(f"AUC: {roc_auc}")
