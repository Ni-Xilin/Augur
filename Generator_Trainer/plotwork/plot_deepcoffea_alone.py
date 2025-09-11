from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pickle
import pathlib
import numpy as np

# fpr: False Positive Rate
# tpr: True Positive Rate
# thresholds: roc_curve使用的阈值
result_fpath1 = pathlib.Path("target_model/deepcoffea/results/stats_AJSMA/gthr.p")
result_fpath2 = pathlib.Path("target_model/deepcoffea/results/stats_BLANKET/gthr.p")
result_fpath3 = pathlib.Path("target_model/deepcoffea/results/stats_CW/gthr.p")
result_fpath4 = pathlib.Path("target_model/deepcoffea/results/stats_IFGSM/gthr.p")
result_fpath5 = pathlib.Path("target_model/deepcoffea/results/stats_PGD/gthr.p")
result_fpath6 = pathlib.Path("target_model/deepcoffea/results/stats_corrmatrix/gthr.p")
result_fpath7 = pathlib.Path("target_model/deepcoffea/results/stats_Gcorrmatrix/gthr.p")
result_fpath8 = pathlib.Path("target_model/deepcoffea/results/stats_RM/gthr.p")

output_filename = "./plotwork/deepcoffea_alone_roc1log.svg"


n_points = 30
fpr_list = []
tpr_list = []
for result_fpath in [result_fpath1, result_fpath2, result_fpath3, result_fpath4,
                     result_fpath5, result_fpath6, result_fpath7,result_fpath8]:
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
plt.plot(fpr_list[5], tpr_list[5], color='#1f77b4', lw=2.5, label=f'No-Perturbation', marker='.', linestyle='-', markersize=10)  # 蓝色
# plt.plot(fpr_list[7], tpr_list[7], color='#ff7f0e', lw=1.5, label=f'RM', marker='.', linestyle='--', markersize=5)  # 橙色
# plt.plot(fpr_list[2], tpr_list[2], color='#696969', lw=1.5, label=f'CW', marker='.', linestyle='--', markersize=5)  # 红色
# plt.plot(fpr_list[3], tpr_list[3], color='#17becf', lw=1.5, label=f'IFGSM', marker='.', linestyle='--', markersize=5)  # 蓝绿色
# plt.plot(fpr_list[0], tpr_list[0], color='#2ca02c', lw=1.5, label=f'JSMA', marker='.', linestyle='--', markersize=5)  # 绿色
# plt.plot(fpr_list[1], tpr_list[1], color='#9467bd', lw=1.5, label=f'BLANKET', marker='.', linestyle='--', markersize=5)  # 紫色
# plt.plot(fpr_list[4], tpr_list[4], color='#e377c2', lw=1.5, label=f'PGD', marker='.', linestyle='--', markersize=5)  # 粉色
plt.plot(fpr_list[6], tpr_list[6], color='#FF0000', lw=2.5, label=f'Augur', marker='.', linestyle='-', markersize=10)

# plt.plot(fpr_list[5], tpr_list[5], color='#1f77b4', lw=1.5, label=f'No-Perturbation', marker='.', linestyle='-', markersize=5)  # 蓝色
# plt.plot(fpr_list[7], tpr_list[7], color='#ff7f0e', lw=1.5, label=f'RM', marker='.', linestyle='-', markersize=5)  # 橙色
# plt.plot(fpr_list[2], tpr_list[2], color='#696969', lw=1.5, label=f'CW', marker='.', linestyle='-', markersize=5)  # 红色
# plt.plot(fpr_list[3], tpr_list[3], color='#17becf', lw=1.5, label=f'IFGSM', marker='.', linestyle='-', markersize=5)  # 蓝绿色
# plt.plot(fpr_list[0], tpr_list[0], color='#2ca02c', lw=1.5, label=f'JSMA', marker='.', linestyle='-', markersize=5)  # 绿色
# plt.plot(fpr_list[1], tpr_list[1], color='#9467bd', lw=1.5, label=f'BLANKET', marker='.', linestyle='-', markersize=5)  # 紫色
# plt.plot(fpr_list[6], tpr_list[6], color='#FF0000', lw=1.5, label=f'Augur', marker='.', linestyle='-', markersize=5)
# plt.plot(fpr_list[4], tpr_list[4], color='#e377c2', lw=1.5, label=f'PGD', marker='.', linestyle='-', markersize=5)  # 粉色



plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')  # 绘制对角线
plt.xlim([0.0001, 1])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate',fontsize = 30)
plt.ylabel('True Positive Rate',fontsize = 30)
# plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right",fontsize=20)

plt.grid(True)
plt.savefig(output_filename, format='svg',transparent=True)
print(f"ROC curve plot saved to {output_filename}")
plt.show()

# print(f"AUC: {roc_auc}")
