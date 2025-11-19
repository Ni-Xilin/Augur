from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pickle
import pathlib
import numpy as np

# fpr: False Positive Rate
# tpr: True Positive Rate
# thresholds: roc_curve使用的阈值

result_fpath = pathlib.Path("target_model/deepcoffea/results/stats_corrmatrix/gthr.p")
result_fpath_time0 = pathlib.Path("target_model/deepcoffea/results/stats_time0/gthr.p")
result_fpath_size0 = pathlib.Path("target_model/deepcoffea/results/stats_size0/gthr.p")
Gresult_fpath = pathlib.Path("target_model/deepcoffea/results/stats_Gcorrmatrix/gthr.p")


output_filename = "./plotwork/deepcoffea_alation_roc1log.svg"


n_points = 30
fpr_list = []
tpr_list = []
for result_fpath in [result_fpath, Gresult_fpath, result_fpath_time0, result_fpath_size0]:
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
    fpr_list.append(fpr_selected)
    tpr_list.append(tpr_selected)
    # roc_auc = auc(fpr, tpr)


# 绘制 ROC 曲线
plt.rcParams.update({'font.size': 30})
plt.figure(figsize=(8, 6))
plt.xscale('log')
plt.plot(fpr_list[0], tpr_list[0], color='#1f77b4', lw=2.5, label=f'No',markerfacecolor='white', marker='s', linestyle='-', markersize=12.5)  # 蓝色
plt.plot(fpr_list[2], tpr_list[2], color='#ff7f0e', lw=2.5, label=f'size',markerfacecolor='white',marker='d', linestyle='-', markersize=12.5)
plt.plot(fpr_list[3], tpr_list[3], color='#2ca02c', lw=2.5, label=f'Time', markerfacecolor='white',marker='v', linestyle='-', markersize=12.5)
plt.plot(fpr_list[1], tpr_list[1], color='#FF0000', lw=2.5, label=f'Both',markerfacecolor='white', marker='o', linestyle='-', markersize=12.5)



plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')  # 绘制对角线
plt.xlim([0.0001, 1])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontsize =30)
plt.ylabel('True Positive Rate',fontsize = 30)
# plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right",fontsize=20)

plt.grid(True)
plt.savefig(output_filename, format='svg',transparent=True)
print(f"ROC curve plot saved to {output_filename}")
plt.show()

# print(f"AUC: {roc_auc}")
