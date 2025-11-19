import matplotlib.pyplot as plt
import numpy as np
import pickle
import pathlib
import matplotlib.ticker as ticker


result_fpath1 = pathlib.Path("checkpoints/deepcoffea/deepcoffea_d3_ws5_nw11_thr20_tl500_el800_nt1000_ap1e-01_es64_lr1e-03_mep100000_bs256/deepcoffea_PatchTST_Deepcoffea_sl150_pl70_dm512_nh8_pal10_s70/step_information_20.p")
result_fpath2 = pathlib.Path("checkpoints/deepcoffea/deepcoffea_d3_ws5_nw11_thr20_tl500_el800_nt1000_ap1e-01_es64_lr1e-03_mep100000_bs256/deepcoffea_PatchTST_Deepcoffea_sl150_pl70_dm512_nh8_pal10_s70/step_information_20_Differentiable.p")

with open(result_fpath1, "rb") as fp:
    data1 = pickle.load(fp)
with open(result_fpath2, "rb") as fp:
    data2 = pickle.load(fp)
win_len = 2
length=len(data1['total_step_loss'])
loss1 = [np.mean(data1['total_step_loss'][i:i+win_len]) for i in range(0,length,win_len)]
totol_step_cosine_loss1 = [np.mean(data1['totol_step_cosine_loss'][i:i+win_len]) for i in range(0,length,win_len)]
total_step_time_rate1 = [np.mean(data1['total_step_time_rate'][i:i+win_len]) for i in range(0,length,win_len)]
total_step_size_rate1 = [np.mean(data1['total_step_size_rate'][i:i+win_len]) for i in range(0,length,win_len)]
loss2 = [np.mean(data2['total_step_loss'][i:i+win_len]) for i in range(0,length,win_len)]
totol_step_cosine_loss2 = [np.mean(data2['totol_step_cosine_loss'][i:i+win_len]) for i in range(0,length,win_len)]
total_step_time_rate2 = [np.mean(data2['total_step_time_rate'][i:i+win_len]) for i in range(0,length,win_len)]
total_step_size_rate2 =[np.mean(data2['total_step_size_rate'][i:i+win_len]) for i in range(0,length,win_len)]
steps = np.arange(0,length,win_len)

# plt.figure(figsize=(6, 8))
# 绘制收敛曲线
# plt.plot(steps,loss1, label='Loss', color='blue', lw=0.5)
# plt.plot(steps,loss2, label='Dloss', color='orange', lw=0.5)

plt.rcParams.update({'font.size': 15})

# plt.plot(steps,totol_step_cosine_loss1, label='Cosine similarity without Differentiable', color='firebrick',markerfacecolor='white', lw=1.5,marker='^',markersize=6)
# plt.plot(steps,totol_step_cosine_loss2, label='Cosine similarity with Differentiable', color='teal',markerfacecolor='white', lw=1.5,marker='v',markersize=6)
# plt.plot(steps,total_step_time_rate1, label='Time Ratio without Differentiable ', color='blue', markerfacecolor='white', lw=1.5,marker='d',markersize=6)
# plt.plot(steps,total_step_time_rate2, label='Time Ratio with Differentiable', color='darkorange', markerfacecolor='white', lw=1.5,marker='p',markersize=6)
# plt.plot(steps,total_step_size_rate1, label='Size Ratio without Differentiable', color='olivedrab',markerfacecolor='white', lw=1.5,marker='s',markersize=6)
# plt.plot(steps,total_step_size_rate2, label='Size Ratio with Differentiable', color='blueviolet',markerfacecolor='white', lw=1.5,marker='o',markersize=6)

plt.plot(steps,totol_step_cosine_loss1, label='Cosine similarity without Differentiable', color='firebrick',markerfacecolor='white', lw=1.5,marker='^',markersize=6,markevery=10)
plt.plot(steps,totol_step_cosine_loss2, label='Cosine similarity with Differentiable', color='teal',markerfacecolor='white', lw=1.5,marker='v',markersize=6,markevery=10)
plt.plot(steps,total_step_time_rate1, label='Time Ratio without Differentiable ', color='blue', markerfacecolor='white', lw=1.5,marker='d',markersize=6,markevery=10)
plt.plot(steps,total_step_time_rate2, label='Time Ratio with Differentiable', color='darkorange', markerfacecolor='white', lw=1.5,marker='p',markersize=6,markevery=10)
plt.plot(steps,total_step_size_rate1, label='Size Ratio without Differentiable', color='olivedrab',markerfacecolor='white', lw=1.5,marker='s',markersize=6,markevery=10)
plt.plot(steps,total_step_size_rate2, label='Size Ratio with Differentiable', color='blueviolet',markerfacecolor='white', lw=1.5,marker='o',markersize=6,markevery=10)


plt.xlabel('Training Step')
plt.ylabel('Loss')
# plt.ylim(0.005, 0.02)
# plt.ylim(0.14, 0.17)
# plt.ylim(0.675,0.685)
# plt.xlim(length//win_len-50, length//win_len)
# plt.title('Convergence Curve')
# plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.025))
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend(fontsize=10)
# plt.tick_params(axis='x', labelbottom=False)
# 保存图形

# 2. 获取当前坐标轴 (Axes)
ax = plt.gca()

# 3. 定义一个4位小数的格式化器
formatter = ticker.FormatStrFormatter('%.4f')

# 4. 将格式化器应用到 Y 轴
# ax.yaxis.set_major_formatter(formatter)
plt.savefig('plotwork/Diff_curve.svg', format='svg',transparent=True)
print(f"Convergence curve saved as 'plotwork/Diff_curve.svg'")
# 显示图形
plt.show()
