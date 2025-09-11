import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import pathlib
def analyze_window_correlations(corr_matrix_path, n_wins,reuslt_matrix):
    """
    加载DeepCoFFEA生成的相关性矩阵，并计算一个 n_wins x n_wins 的
    窗口间平均相关性矩阵，然后将其可视化。

    Args:
        corr_matrix_path (str): corrmatrix.npz 文件的路径。
        n_wins (int): 实验中使用的窗口数量 (e.g., 11)。
    """

    data = np.load(corr_matrix_path)
    corr_matrix = data['corr_matrix']
    print(f"成功加载相关性矩阵，形状为: {corr_matrix.shape}")


    #  计算 n_test
    if corr_matrix.shape[0] != corr_matrix.shape[1]:
        raise ValueError("相关性矩阵必须是方阵！")
    if corr_matrix.shape[0] % n_wins != 0:
        raise ValueError("矩阵维度与窗口数量不匹配！")
    
    n_test = corr_matrix.shape[0] // n_wins
    print(f"推断出 n_test (测试样本数) = {n_test}")

    # 3. 初始化 n_wins x n_wins 的结果矩阵
    window_confusion_matrix = np.zeros((n_wins, n_wins))

    # 4. 遍历所有窗口对，计算平均相关性
    for i in range(n_wins):  # 入口流的窗口索引 (tor flow window index)
        for j in range(n_wins):  # 出口流的窗口索引 (exit flow window index)
            
            # 切片，提取出对应的 n_test x n_test 子矩阵
            # 这个子矩阵代表了入口流第i窗口和出口流第j窗口之间的所有相关性
            sub_matrix_slice = (slice(i * n_test, (i + 1) * n_test),
                                slice(j * n_test, (j + 1) * n_test))
            sub_matrix = corr_matrix[sub_matrix_slice]
            
            # 提取这个子矩阵的对角线元素
            # 这代表了所有真实匹配对的相关性分数
            diagonal_correlations = np.diag(sub_matrix)
            
            # 计算平均值并存入结果矩阵
            mean_corr = np.mean(diagonal_correlations)
            window_confusion_matrix[i, j] = mean_corr

    print("\n计算出的窗口间平均相关性矩阵:")
    # 设置打印选项，使其更易读
    np.set_printoptions(precision=3, suppress=True)
    print(window_confusion_matrix)

    # 5. 可视化结果
    plt.figure(figsize=(10, 8))
    ax=sns.heatmap(window_confusion_matrix,vmin=0,vmax=1,annot=True, fmt=".3f", cmap="Greens",
                linewidths=.5, cbar_kws={'label': 'Mean Correlation Score'},annot_kws={"size": 12})
    cbar_ax = ax.figure.axes[-1]
    cbar_ax.set_ylabel(
        'Correlation Score', 
        fontsize=20  # <-- 在这里设置您想要的字号
    )
    cbar_ax.tick_params(labelsize=20)
    # plt.title('窗口间真实匹配对的平均相关性热力图', fontsize=16)
    plt.xlabel('Exit Flow Window Index', fontsize=20)
    plt.ylabel('Tor Flow Window Index', fontsize=20)
    plt.xticks(ticks=np.arange(n_wins) + 0.5, labels=np.arange(n_wins),fontsize=20)
    plt.yticks(ticks=np.arange(n_wins) + 0.5, labels=np.arange(n_wins),fontsize=20)
    
    # 保存图像
    plt.savefig(reuslt_matrix, dpi=400, bbox_inches='tight')
    print(f"\n热力图已保存为 '{reuslt_matrix}'")
    plt.show()


if __name__ == '__main__':
    # --- 您需要修改这里的参数 ---
    
    # corrmatrix.npz 文件的路径
    # result_fpath = pathlib.Path("target_model/deepcoffea/code/corrmatrix_sim0.7404.npz")
    result_fpath = pathlib.Path("target_model/deepcoffea/code/Gcorrmatrix_time0.1398_size0.0038_sim0.1945.npz")
    reuslt_matrix = pathlib.Path("plotwork/Gdeepcoffea_condition_matrix")
    # 您的实验中使用的 n_wins 数量
    n_wins = 11

    # --------------------------
    
    analyze_window_correlations(result_fpath, n_wins,reuslt_matrix)