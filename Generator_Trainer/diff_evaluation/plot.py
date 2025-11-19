import pickle
import matplotlib.pyplot as plt
import math

def load_data(filename):
    """读取 pickle 文件"""
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {filename}")
        return None

def plot_all_metrics(file_undiff, file_diff):
    # 1. 读取数据
    data_undiff = load_data(file_undiff)
    data_diff = load_data(file_diff)

    if data_undiff is None or data_diff is None:
        return

    # 2. 确定需要绘制的指标 (排除 'epoch')
    # 假设两个文件的 key 是一样的，取交集以防万一
    keys_undiff = set(data_undiff.keys())
    keys_diff = set(data_diff.keys())
    
    # 我们要画的指标是所有非 'epoch' 的 key
    metrics = list(keys_undiff.intersection(keys_diff))
    if 'epoch' in metrics:
        metrics.remove('epoch')
    
    metrics.sort() # 排序保证每次画图顺序一致

    num_metrics = len(metrics)
    print(f"检测到以下指标需要绘制: {metrics}")

    if num_metrics == 0:
        print("没有找到可绘制的指标数据。")
        return

    # 3. 动态计算子图布局 (例如 4个指标用 2x2, 5个用 2x3 等)
    cols = 2
    rows = math.ceil(num_metrics / cols)
    
    # 设置画布大小
    plt.figure(figsize=(6 * cols, 5 * rows))

    # 4. 循环绘制每个指标
    for i, metric in enumerate(metrics):
        plt.subplot(rows, cols, i + 1)
        
        # 格式化标题 (将 key 转换为更易读的格式, e.g., 'time_l2_rate' -> 'Time L2 Rate')
        title_str = metric.replace('_', ' ').title()
        
        # 绘制 Undiff
        plt.plot(data_undiff['epoch'], data_undiff[metric], 
                 label='Undiff', color='blue', linestyle='-', marker='o', markersize=2, alpha=0.8)
        
        # 绘制 Diff
        plt.plot(data_diff['epoch'], data_diff[metric], 
                 label='Diff', color='red', linestyle='--', marker='s', markersize=2, alpha=0.8)
        
        plt.title(f'{title_str} Convergence')
        plt.xlabel('Epoch')
        plt.ylabel(title_str)
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)

    # 5. 调整布局并保存
    plt.tight_layout()
    output_filename = 'all_metrics_comparison.png'
    plt.savefig(output_filename, dpi=300)
    print(f"绘图完成，已保存为 {output_filename}")
    plt.show()

# --- 执行主程序 ---
if __name__ == "__main__":
    plot_all_metrics('diff_evaluation/undiff_training_data.p', 'diff_evaluation/diff_training_data.p')