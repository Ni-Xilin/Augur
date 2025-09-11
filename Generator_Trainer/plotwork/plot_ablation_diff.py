import matplotlib.pyplot as plt
import numpy as np
import pickle
import pathlib

result_fpath1 = pathlib.Path("checkpoints/deepcoffea/deepcoffea_d3_ws5_nw11_thr20_tl500_el800_nt1000_ap1e-01_es64_lr1e-03_mep100000_bs256/deepcoffea_PatchTST_Deepcoffea_sl150_pl70_dm512_nh8_pal10_s70/step_information_20.p")
result_fpath2 = pathlib.Path("checkpoints/deepcoffea/deepcoffea_d3_ws5_nw11_thr20_tl500_el800_nt1000_ap1e-01_es64_lr1e-03_mep100000_bs256/deepcoffea_PatchTST_Deepcoffea_sl150_pl70_dm512_nh8_pal10_s70/step_information_20_Differentiable.p")

with open(result_fpath1, "rb") as fp:
    data1 = pickle.load(fp)
with open(result_fpath2, "rb") as fp:
    data2 = pickle.load(fp)
win_len=2
length=len(data1['total_step_loss'])
loss1 = [np.mean(data1['total_step_loss'][i:(i+1)*win_len]) for i in range(length//win_len)]
totol_step_cosine_loss1 = [np.mean(data1['totol_step_cosine_loss'][i:(i+1)*win_len]) for i in range(length//win_len)]
total_step_time_rate1 = [np.mean(data1['total_step_time_rate'][i:(i+1)*win_len]) for i in range(length//win_len)]
total_step_size_rate1 = [np.mean(data1['total_step_size_rate'][i:(i+1)*win_len]) for i in range(length//win_len)]
loss2 = [np.mean(data2['total_step_loss'][i:(i+1)*win_len]) for i in range(length//win_len)]
totol_step_cosine_loss2 = [np.mean(data2['totol_step_cosine_loss'][i:(i+1)*win_len]) for i in range(length//win_len)]
total_step_time_rate2 = [np.mean(data2['total_step_time_rate'][i:(i+1)*win_len]) for i in range(length//win_len)]
total_step_size_rate2 =[np.mean(data2['total_step_size_rate'][i:(i+1)*win_len]) for i in range(length//win_len)]
steps = np.arange(length//win_len)

plt.figure(figsize=(8, 1))
# 绘制收敛曲线
# plt.plot(steps,loss1, label='Loss', color='blue', lw=0.5)
# plt.plot(steps,loss2, label='Dloss', color='orange', lw=0.5)

plt.rcParams.update({'font.size': 15})

# plt.plot(steps,totol_step_cosine_loss1, label='Cosine similarity without Differentiable', color='firebrick', lw=1)
# plt.plot(steps,totol_step_cosine_loss2, label='Cosine similarity with Differentiable', color='teal', lw=1)
# plt.plot(steps,total_step_time_rate1, label='Time Ratio without Differentiable ', color='blue', lw=1)
# plt.plot(steps,total_step_time_rate2, label='Time Ratio with Differentiable', color='darkorange', lw=1)
plt.plot(steps,total_step_size_rate1, label='Size Ratio without Differentiable', color='olivedrab', lw=1)
plt.plot(steps,total_step_size_rate2, label='Size Ratio with Differentiable', color='blueviolet', lw=1)
plt.xlabel('Training Step')
plt.ylabel('Loss',fontsize=20)
plt.ylim(0, 0.05)
# plt.ylim(0.125, 0.175)
# plt.ylim(0.675,0.725)
# plt.title('Convergence Curve')
plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.025))
plt.grid(True, linestyle='--', linewidth=0.5)
# plt.legend(fontsize=10)
# plt.tick_params(axis='x', labelbottom=False)
# 保存图形
plt.savefig('plotwork/Diff_size_curve.svg', format='svg',transparent=True)
print(f"Convergence curve saved as 'plotwork/Diff_cos_curve.svg'")
# 显示图形
plt.show()
