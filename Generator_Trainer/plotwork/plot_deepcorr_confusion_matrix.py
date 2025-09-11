import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pathlib
from matplotlib.colors import ListedColormap

def plot_deepcorr_heatmap(score_matrix_path, output_image_path):
    """
    加载 DeepCorr 的分数矩阵并将其可视化为热力图。

    Args:
        score_matrix_path (str or pathlib.Path): score_matrix.p 文件的路径。
        output_image_path (str or pathlib.Path): 保存热力图的文件路径。
    """
    try:
        # 1. 加载已计算好的分数矩阵
        with open(score_matrix_path, "rb") as fp:
            score_matrix = pickle.load(fp)
        print(f"成功加载分数矩阵，形状为: {score_matrix.shape}")
    except FileNotFoundError:
        print(f"错误: 未找到分数矩阵文件 '{score_matrix_path}'。")
        print("将使用一个模拟矩阵进行演示。")
        # 如果文件不存在，创建一个模拟矩阵用于演示
        n_samples = 16
        score_matrix = np.random.rand(n_samples, n_samples) * 0.2
        np.fill_diagonal(score_matrix, np.random.rand(n_samples) * 0.4 + 0.6)

    n_samples = score_matrix.shape[0]
    original_cmap = plt.get_cmap('PuBu')
    # 2. 从原始方案中截取颜色较深的部分 (例如，只取后75%的颜色)
    #    您可以调整 np.linspace 的第一个参数，值越大，起始颜色越深
    #    例如，0.2 表示从原始20%的位置开始取色
    modified_colors = original_cmap(np.linspace(0.0, 1.0, 256))
    # 3. 创建一个新的 ListedColormap
    custom_cmap = ListedColormap(modified_colors)
    # 2. 可视化结果
    plt.figure(figsize=(12, 10))
    ax=sns.heatmap(
        score_matrix,
        annot=True,               # 在单元格上显示数值
        fmt=".1f",                # 数值格式为小数点后三位
        cmap=custom_cmap,             # 使用与您示例中相同的 'Blues' 配色
        linewidths=.5,            # 单元格之间的分隔线宽度
        vmin=-30,                 # 将颜色条的最小值固定为 0.0
        vmax=10.0,                 # 将颜色条的最大值固定为 1.0
        # cbar_kws={'label': 'Correlation Score',fontsize =15},# 设置颜色条的标签
        annot_kws={"size": 15} ,          # 设置字体大小
    )
    cbar_ax = ax.figure.axes[-1]
    cbar_ax.set_ylabel(
        'Correlation Score', 
        fontsize=20  # <-- 在这里设置您想要的字号
    )
    cbar_ax.tick_params(labelsize=20)
    # plt.title('DeepCorr 16x16 Correlation Score Matrix', fontsize=16)
    plt.xlabel("Exit Flow Index", fontsize=20)
    plt.ylabel("Tor Flow Index", fontsize=20)
    
    # 设置刻度，使其更清晰
    tick_labels = [str(i) for i in range(n_samples)]
    plt.xticks(ticks=np.arange(n_samples) + 0.5, labels=tick_labels, rotation=0,fontsize=20)
    plt.yticks(ticks=np.arange(n_samples) + 0.5, labels=tick_labels, rotation=0,fontsize=20)
    
    # 3. 保存图像
    plt.savefig(output_image_path, dpi=400, bbox_inches='tight')
    print(f"\n热力图已保存为 '{output_image_path}'")
    plt.show()


if __name__ == '__main__':
    # --- 您需要修改这里的参数 ---
    
    # 已保存的 score_matrix.p 文件的路径
    SCORE_MATRIX_FILE = pathlib.Path("plotwork/Gmdeepcorr_score_matrix.p")
    
    # 您希望保存的热力图图片路径
    OUTPUT_IMAGE_FILE = pathlib.Path("plotwork/Gmdeepcorr_score_heatmap.png")
    
    # --------------------------
    
    plot_deepcorr_heatmap(SCORE_MATRIX_FILE, OUTPUT_IMAGE_FILE)