import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size': 16}) 
# 数据准备
labels = ['Size', 'Time', 'Both']
group1 = [0.1479, 0, 0.0089]  # 第一列数据
group2 = [0, 0.1371, 0.1343]  # 第二列数据

x = np.arange(len(labels))  # 横坐标位置
width = 0.2  # 每组柱状图的宽度


fig, ax = plt.subplots()

# 设置稍微深一点的颜色
bars1 = ax.bar(x - width/2, group1, width, label='Size Ratio', color='lightgoldenrodyellow', edgecolor='black')
bars2 = ax.bar(x + width/2, group2, width, label='Time Ratio', color='mediumpurple', edgecolor='black')

# 在每个柱状图上方标出数据
for bar in bars1:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 4), ha='center', va='bottom')

for bar in bars2:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 4), ha='center', va='bottom')

# 添加虚线对齐到刻度线
ax.grid(True, linestyle='--', linewidth=0.5, axis='y')

# 添加标签和标题
ax.set_ylabel('L2 Ratio')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 0.25)

# 添加图例
ax.legend(loc="upper left",fontsize=15)

# 设置边框不透明
for spine in ax.spines.values():
    spine.set_alpha(1)

# 保存图形
output_file = 'plotwork/deepcorr_ratio.png'
fig.savefig(output_file, format='png',dpi=400, bbox_inches='tight')
print(f"图形已保存到 {output_file}")
output_file  # 返回保存的文件路径

