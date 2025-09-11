import numpy as np
import pickle
import pathlib
from tqdm import tqdm

DEEPCOFFEA_FILE = "target_model/deepcoffea/dataset/CrawlE_Proc/filtered_and_partitioned/d3_ws5_nw11_thr20_tl500_el800_nt1000_test_session.npz"
try:
    print(f"正在从 {DEEPCOFFEA_FILE} 加载DeepCoffea数据...")
    data = np.load(DEEPCOFFEA_FILE, allow_pickle=True)
    tor_ipds_ms, tor_sizes_kb = data['tor_ipds'], data['tor_sizes']
    exit_ipds_ms, exit_sizes_kb = data['exit_ipds'], data['exit_sizes']
    num_samples = len(tor_ipds_ms)
    print(f"加载了 {num_samples} 条电路数据。")

except FileNotFoundError:
    print(f"错误：找不到文件 {DEEPCOFFEA_FILE}。")
    print("请确保您已经通过 'gnpz_win.py' 和 'gnpz_session.py' 生成了该文件。")


for i in tqdm(range(num_samples), desc="处理进度"):
    print(f"样本 {i}:len(tor_ipds)={len(tor_ipds_ms[i])}, len(tor_sizes)={len(tor_sizes_kb[i])}, len(exit_ipds)={len(exit_ipds_ms[i])}, len(exit_sizes)={len(exit_sizes_kb[i])}")