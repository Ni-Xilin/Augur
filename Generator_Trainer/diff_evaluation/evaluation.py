import re
import pickle

def extract_and_save_data(input_file, output_file):
    # 初始化数据结构
    data = {
        "epoch": [],
        "loss": [],
        "cosine_loss": [],
        "time_l2_rate": [],
        "size_l2_rate": []
    }

    # 定义正则表达式匹配模式
    # 匹配格式: Epoch: 1,Loss: 4.93..., Cosine Loss: 1.22..., Time L2 rate: 0.004..., Size L2 rate: 0.44...
    pattern = re.compile(
        r"Epoch:\s*(\d+),"
        r"Loss:\s*([\d\.]+),\s*"
        r"Cosine Loss:\s*([\d\.]+),\s*"
        r"Time L2 rate:\s*([\d\.]+),\s*"
        r"Size L2 rate:\s*([\d\.]+)"
    )

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                # 去除首尾空白字符
                line = line.strip()
                match = pattern.search(line)
                
                if match:
                    # 提取并转换数据类型
                    epoch = int(match.group(1))
                    loss = float(match.group(2))
                    cosine_loss = float(match.group(3))
                    time_l2 = float(match.group(4))
                    size_l2 = float(match.group(5))

                    # 存入列表
                    data["epoch"].append(epoch)
                    data["loss"].append(loss)
                    data["cosine_loss"].append(cosine_loss)
                    data["time_l2_rate"].append(time_l2)
                    data["size_l2_rate"].append(size_l2)

        # 检查是否提取到了数据
        if len(data["epoch"]) > 0:
            # 保存为 .p (pickle) 文件
            with open(output_file, 'wb') as f:
                pickle.dump(data, f)
            
            print(f"成功提取 {len(data['epoch'])} 条数据。")
            print(f"数据已保存至: {output_file}")
            
            # 打印前两行验证
            print("\n数据预览:")
            print(f"Epochs: {data['epoch'][:]}")
            print(f"Loss: {data['loss'][:]}")
        else:
            print("未在文件中找到匹配的数据格式，请检查文件内容。")

    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file}")
    except Exception as e:
        print(f"发生错误: {e}")

# 执行提取
# 确保当前目录下有 undiff.txt 文件
if __name__ == "__main__":
    extract_and_save_data('nohup.out', 'diff_evaluation/diff_training_data.p')