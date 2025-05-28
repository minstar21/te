import torch
import matplotlib.pyplot as plt
import os

data_root = '/data1/tsq/zkj_use/Trustworthy-Evaluation/Alignment/hooked_llama/neuron_activation/'
# 文件列表
file_paths = [
    'new_method_test.pt',
    'new_method_5_epoch_paraphrased_vs_openorca.pt',
    'llama-2-7b_5epoch_math_contaminated_all_sft_vs_base_on_openorca_sft_completion.pt',
    # 'llama-2-7b_2%_gsm_contaminated_all_sft_vs_base_on_openorca_sft_completion.pt',
    # 'file2.pt',
    # 'file3.pt',
    # 添加更多文件路径
]

# 颜色列表，每个文件一个颜色
colors = [
    'red',
    'blue',
    'green',
    # 'orange',
    # 添加更多颜色
]

# 检查文件数量和颜色数量是否匹配
assert len(file_paths) == len(colors), "Number of files and colors must match"

# 绘制散点图
plt.figure(figsize=(10, 6))

# 循环处理每个文件
for file_path, color in zip(file_paths, colors):
    _, index, *_ = torch.load(f'{data_root}/{file_path}')
    topk_index = index[1000:1200]
    assert topk_index.shape[1] == 2, "Index must be a n×2 tensor"
    x = topk_index[:, 0].numpy()
    y = topk_index[:, 1].numpy()
    plt.scatter(x, y, alpha=0.5, label=file_path.split('.')[0], s=10, color=color)  # s参数控制散点大小

# 添加图例
plt.legend()

# 设置标题和轴标签
plt.title('Scatter Plot of Index')
plt.xlabel('X-axis (0-31)')
plt.ylabel('Y-axis (0-11007)')
plt.grid(True)

# 保存路径
save_path = '/data1/tsq/zkj_use/Trustworthy-Evaluation/results/neuron_viewer'
if not os.path.exists(save_path):
    os.makedirs(save_path)

file_name = 'contaminated_neuron.png'
full_path = os.path.join(save_path, file_name)

# 保存图形
plt.savefig(full_path)

# 显示图形
plt.show()

# 打印保存路径
print(f"The scatter plot has been saved to {full_path}")