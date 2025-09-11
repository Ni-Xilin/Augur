import numpy as np
import pathlib
from tqdm import tqdm
import os

def parse_flow_file(file_path):
    """解析流量文件，提取时间戳和大小"""
    timestamps = []
    sizes = []
    with open(file_path, 'r') as fp:
        for line in fp:
            timestamp, size = line.strip().split("\t")
            timestamps.append(float(timestamp))
            sizes.append(int(size))
    return np.array(timestamps), np.array(sizes)

def process_flow(timestamps, sizes):
    """处理流量数据，过滤掉小包并合并连续的零IPD包"""
    processed_timestamps = []
    processed_sizes = []
    big_pkt = []
    prev_time = 0.0

    for timestamp, size in zip(timestamps, sizes):
        if abs(size) <= 66:  # 舍弃小包
            continue

        if len(big_pkt) > 0 and timestamp == prev_time:  # 连续的零IPD包
            big_pkt.append(size)
        else:
            if big_pkt:
                processed_timestamps.append(prev_time)
                processed_sizes.append(sum(big_pkt))
            big_pkt = [size]
            prev_time = timestamp

    if big_pkt:  # 处理最后一个大包
        processed_timestamps.append(prev_time)
        processed_sizes.append(sum(big_pkt))

    return np.array(processed_timestamps), np.array(processed_sizes)

def generate_flow_npz(data_root, train_npz_path, test_npz_path):
    """生成包含整段流量的npz文件"""
    data_root = pathlib.Path(data_root)
    inflow_path = data_root / "inflow"
    outflow_path = data_root / "outflow"

    # 加载训练集和测试集的标签
    train_labels = np.load(train_npz_path, allow_pickle=True)["train_label"]
    test_labels = np.load(test_npz_path, allow_pickle=True)["test_label"]

    # 提取训练集流量
    train_inflow_ipds = []
    train_inflow_sizes = []
    train_outflow_ipds = []
    train_outflow_sizes = []
    train_labels_list = []
    for label in tqdm(train_labels[0]):
        label = label.split("_wi00")[0]  # 去掉窗口编号
        inflow_file = inflow_path / label
        outflow_file = outflow_path / label

        inflow_timestamps, inflow_sizes = parse_flow_file(inflow_file)
        outflow_timestamps, outflow_sizes = parse_flow_file(outflow_file)

        # 处理流量数据
        inflow_timestamps, inflow_sizes = process_flow(inflow_timestamps, inflow_sizes)
        outflow_timestamps, outflow_sizes = process_flow(outflow_timestamps, outflow_sizes)

        # 计算 IPD 并保留符号
        inflow_ipds = np.diff(inflow_timestamps, prepend=0)  # 初始 IPD 为 0
        outflow_ipds = np.diff(outflow_timestamps, prepend=0)

        # 特征缩放
        inflow_ipds_scaled = inflow_ipds * 1000
        inflow_sizes_scaled = inflow_sizes / 1000
        outflow_ipds_scaled = outflow_ipds * 1000
        outflow_sizes_scaled = outflow_sizes / 1000

        # 为 IPD 和 size 添加符号
        inflow_ipds_signed = np.sign(inflow_sizes) * inflow_ipds_scaled
        outflow_ipds_signed = np.sign(outflow_sizes) * outflow_ipds_scaled

        train_inflow_ipds.append(inflow_ipds_signed)
        train_inflow_sizes.append(inflow_sizes_scaled)
        train_outflow_ipds.append(outflow_ipds_signed)
        train_outflow_sizes.append(outflow_sizes_scaled)
        train_labels_list.append(label)

    # 提取测试集流量
    test_inflow_ipds = []
    test_inflow_sizes = []
    test_outflow_ipds = []
    test_outflow_sizes = []
    test_labels_list = []
    for label in tqdm(test_labels[0]):
        label = label.split("_wi00")[0]  # 去掉窗口编号
        inflow_file = inflow_path / label
        outflow_file = outflow_path / label

        inflow_timestamps, inflow_sizes = parse_flow_file(inflow_file)
        outflow_timestamps, outflow_sizes = parse_flow_file(outflow_file)

        # 处理流量数据
        inflow_timestamps, inflow_sizes = process_flow(inflow_timestamps, inflow_sizes)
        outflow_timestamps, outflow_sizes = process_flow(outflow_timestamps, outflow_sizes)

        # 计算 IPD 并保留符号
        inflow_ipds = np.diff(inflow_timestamps, prepend=0)  # 初始 IPD 为 0
        outflow_ipds = np.diff(outflow_timestamps, prepend=0)

        # 特征缩放
        inflow_ipds_scaled = inflow_ipds * 1000
        inflow_sizes_scaled = inflow_sizes / 1000
        outflow_ipds_scaled = outflow_ipds * 1000
        outflow_sizes_scaled = outflow_sizes / 1000

        # 为 IPD 和 size 添加符号
        inflow_ipds_signed = np.sign(inflow_sizes) * inflow_ipds_scaled
        outflow_ipds_signed = np.sign(outflow_sizes) * outflow_ipds_scaled

        test_inflow_ipds.append(inflow_ipds_signed)
        test_inflow_sizes.append(inflow_sizes_scaled)
        test_outflow_ipds.append(outflow_ipds_signed)
        test_outflow_sizes.append(outflow_sizes_scaled)
        test_labels_list.append(label)

    # 保存训练集和测试集的npz文件
    np.savez_compressed(
        data_root / "d3_ws5_nw11_thr20_tl500_el800_nt1000_train_session.npz",
        tor_ipds=np.array(train_inflow_ipds, dtype=object),
        tor_sizes=np.array(train_inflow_sizes, dtype=object),
        exit_ipds=np.array(train_outflow_ipds, dtype=object),
        exit_sizes=np.array(train_outflow_sizes, dtype=object),
        labels=np.array(train_labels_list, dtype=object)
    )

    np.savez_compressed(
        data_root / "d3_ws5_nw11_thr20_tl500_el800_nt1000_test_session.npz",
        tor_ipds=np.array(test_inflow_ipds, dtype=object),
        tor_sizes=np.array(test_inflow_sizes, dtype=object),
        exit_ipds=np.array(test_outflow_ipds, dtype=object),
        exit_sizes=np.array(test_outflow_sizes, dtype=object),
        labels=np.array(test_labels_list, dtype=object)
    )

    print("训练集和测试集的整段流量npz文件已生成。")

# 示例调用
data_root = "./datasets/CrawlE_Proc"  # 替换为你的数据根目录
train_npz_path = "./datasets/CrawlE_Proc/filtered_and_partitioned/d3_ws5_nw11_thr20_tl500_el800_nt1000_train.npz"  # 替换为你的训练集npz路径
test_npz_path = "./datasets/CrawlE_Proc/filtered_and_partitioned/d3_ws5_nw11_thr20_tl500_el800_nt1000_test.npz"  # 替换为你的测试集npz路径

generate_flow_npz(data_root, train_npz_path, test_npz_path)