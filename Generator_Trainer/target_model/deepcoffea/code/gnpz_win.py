import data_utils

# 设置参数
delta = 3
win_size = 5
n_wins = 11
threshold = 20
tor_len = 500
exit_len = 800
n_test = 1000
data_root = "./datasets/CrawlE_Proc"
seed = 114

# 调用 preprocess_dcf 函数生成 npz 文件
data_utils.preprocess_dcf(delta, win_size, n_wins, threshold, tor_len, exit_len, n_test, data_root, seed)