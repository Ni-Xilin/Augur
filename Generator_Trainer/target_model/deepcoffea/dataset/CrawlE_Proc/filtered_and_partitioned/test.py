import numpy as np
from tqdm import tqdm
def get_tprsfprsprs_globalthr(corr_matrix, n_wins, vote_thr):
    n_test = corr_matrix.shape[0] // n_wins
    tprs = []   # true positive rate, recall, sensitivity
    fprs = []   # false positive rate
    prs = []    # precision, positive predictive value
    for eta in tqdm(np.arange(0.5, 1, 0.1), ascii=True, ncols=120):
        votes = np.zeros((n_test, n_test), dtype=np.int64)
        for wi in range(0, n_wins):
            corr_matrix_win = corr_matrix[n_test*wi:n_test*(wi + 1), n_test*wi:n_test*(wi + 1)]

            for i in range(n_test):
                for j in range(n_test):
                    if corr_matrix_win[i,j] >= eta:
                        votes[i,j] += 1

        tp, fp, tn, fn = 0, 0, 0, 0
        for i in range(n_test):
            for j in range(n_test):
                if votes[i,j] >= vote_thr and i == j:
                    tp += 1
                elif votes[i,j] >= vote_thr and i != j:
                    fp += 1
                elif votes[i,j] < vote_thr and i == j:
                    fn += 1
                else:   # votes[i,j] < vote_thr and i != j
                    tn += 1

        if tp + fn == 0:
            tprs.append(0.0)
        else:
            tprs.append(tp / (tp + fn))
        
        if fp + tn == 0:
            fprs.append(0.0)
        else:
            fprs.append(fp / (fp + tn))

        if tp + fp == 0:
            prs.append(0.0)
        else:
            prs.append(tp / (tp + fp))

    return tprs, fprs, prs
# 读取 .npz 文件
file_path1 = "datasets/CrawlE_Proc/filtered_and_partitioned/d3_ws5_nw11_thr20_tl500_el800_nt1000_test.npz"
file_path2 = "datasets/CrawlE_Proc/filtered_and_partitioned/d3_ws5_nw11_thr20_tl500_el800_nt1000_test_session.npz" # 替换为你的 .npz 文件路径

data1 = np.load(file_path1,allow_pickle=True)

data2 = np.load(file_path2,allow_pickle=True)
# tprs, fprs, prs = get_tprsfprsprs_globalthr(data['corr_matrix'], 11, 9)
tmp = data1['train_tor'][0][0]
tmp = tmp[tmp != 0]
print(tprs)
print(fprs)
print(prs)
print(1)