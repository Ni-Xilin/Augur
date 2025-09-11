# 这个方案包含两个脚本，完全按照您的需求设计：

# 1.  **`deepcoffea_to_deepcorr.py`**:
#     * **功能**: 读取 DeepCoffea 的 `_session.npz` 文件。
#     * 将其转换为 DeepCorr 的 `.pickle` 格式，**同时**生成一个包含时间戳和原始顺序的 `_metadata.pkl` 文件作为“蓝图”。
#     * **何时使用**: 在您需要将 DeepCoffea 数据提供给 DeepCorr 模型或生成器之前运行此脚本。

# 2.  **`deepcorr_to_deepcoffea.py`**:
#     * **功能**: 读取被您的 DeepCorr 生成器**扰动过**的 `.pickle` 文件。
#     * 利用第一步生成的 `_metadata.pkl` 文件，将扰动后的数据精确地还原成 DeepCoffea 的 `_session.npz` 格式。
#     * **何时使用**: 在您完成了对 DeepCorr 格式数据的扰动，并希望用 DeepCoffea 模型来评估扰动效果时运行此脚本。

# ---

# ##DeepCoffea -> DeepCorr (附带元数据)**

# 这个脚本是您进行扰动前的**第一步**。它会为您准备好 DeepCorr 格式的数据和用于还原的“蓝图”。
# -*- coding: utf-8 -*-
import numpy as np
import pickle
import pathlib
from tqdm import tqdm

def convert_deepcoffea_to_deepcorr_with_metadata(deepcoffea_session_path, output_pickle_path, metadata_path):
    """
    将DeepCoffea的 _session.npz 文件转换为DeepCorr格式，并保存用于反向转换的元数据。

    参数:
    - deepcoffea_session_path (str): 输入的DeepCoffea _session.npz文件路径。
    - output_pickle_path (str): 输出的DeepCorr格式 .pickle文件路径。
    - metadata_path (str): 用于保存元数据的 .pkl 文件路径。
    """
    try:
        print(f"正在从 {deepcoffea_session_path} 加载DeepCoffea数据...")
        data = np.load(deepcoffea_session_path, allow_pickle=True)
        tor_ipds_ms, tor_sizes_kb = data['tor_ipds'], data['tor_sizes']
        exit_ipds_ms, exit_sizes_kb = data['exit_ipds'], data['exit_sizes']
        num_samples = len(tor_ipds_ms)
        print(f"加载了 {num_samples} 条电路数据。")

    except FileNotFoundError:
        print(f"错误：找不到文件 {deepcoffea_session_path}。")
        print("请确保您已经通过 'gnpz_win.py' 和 'gnpz_session.py' 生成了该文件。")
        return

    deepcorr_data = []
    metadata_list = []

    print("开始转换数据并生成元数据...")
    for i in tqdm(range(num_samples), desc="处理进度"):
        # --- 还原原始时间和大小序列 ---
        original_tor_sizes = tor_sizes_kb[i] * 1000.0
        original_tor_ipds = tor_ipds_ms[i]
        
        original_exit_sizes = exit_sizes_kb[i] * 1000.0
        original_exit_ipds = exit_ipds_ms[i]
        
        # --- 保存元数据 ---
        metadata_list.append({
            'tor_ipds': original_tor_ipds,
            'tor_sizes': original_tor_sizes,
            'exit_ipds': original_exit_ipds,
            'exit_sizes': original_exit_sizes,
        })

        # --- 分离为DeepCorr格式 ---
        here_ipd_s = np.abs(original_tor_ipds) / 1000.0
        here_size_bytes = original_tor_sizes
        here_incoming_times = [ipd for ipd, size in zip(here_ipd_s, here_size_bytes) if size < 0]
        here_outgoing_times = [ipd for ipd, size in zip(here_ipd_s, here_size_bytes) if size > 0]
        here_incoming_sizes = [abs(size) for size in here_size_bytes if size < 0]
        here_outgoing_sizes = [size for size in here_size_bytes if size > 0]

        there_ipd_s = np.abs(original_exit_ipds) / 1000.0
        there_size_bytes = original_exit_sizes
        there_incoming_times = [ipd for ipd, size in zip(there_ipd_s, there_size_bytes) if size < 0]
        there_outgoing_times = [ipd for ipd, size in zip(there_ipd_s, there_size_bytes) if size > 0]
        there_incoming_sizes = [abs(size) for size in there_size_bytes if size < 0]
        there_outgoing_sizes = [size for size in there_size_bytes if size > 0]

        deepcorr_data.append({
            'here': [{'<-': here_incoming_times, '->': here_outgoing_times},
                     {'<-': here_incoming_sizes, '->': here_outgoing_sizes}],
            'there': [{'<-': there_incoming_times, '->': there_outgoing_times},
                      {'<-': there_incoming_sizes, '->': there_outgoing_sizes}]
        })

    # --- 保存文件 ---
    pathlib.Path(output_pickle_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_pickle_path, 'wb') as f:
        pickle.dump(deepcorr_data, f)
    print(f"成功将 {len(deepcorr_data)} 条样本保存到: {output_pickle_path}")

    pathlib.Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata_list, f)
    print(f"成功将元数据保存到: {metadata_path}")


if __name__ == '__main__':
    DEEPCOFFEA_FILE = "target_model/deepcoffea/dataset/CrawlE_Proc/filtered_and_partitioned/d3_ws5_nw11_thr20_tl500_el800_nt1000_test_session.npz"
    DEEPCORR_FILE = "datasets_convert/deepcorr_test_from_deepcoffea.pickle"
    METADATA_FILE = "datasets_convert/conversion_metadata.pkl"
    
    convert_deepcoffea_to_deepcorr_with_metadata(DEEPCOFFEA_FILE, DEEPCORR_FILE, METADATA_FILE)