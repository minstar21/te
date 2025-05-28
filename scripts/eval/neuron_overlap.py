import torch
import os

data_root = '/data1/tsq/zkj_use/Trustworthy-Evaluation/Alignment/hooked_llama/neuron_activation/'

# 文件列表
file_paths = [
    'new_method_test.pt',
    'new_method_5_epoch_paraphrased_vs_openorca.pt',
    'llama-2-7b_5epoch_math_contaminated_all_sft_vs_base_on_openorca_sft_completion.pt',
]

# 加载所有文件的topk_index
topk_indices = []
for file_path in file_paths:
    _, index, *_ = torch.load(f'{data_root}/{file_path}')
    topk_index = index[:5000]  # 修改为0到10000的元素
    topk_indices.append(topk_index)
    # breakpoint()


common_elements = set.intersection(*[set(map(tuple, idx.numpy())) for idx in topk_indices])


total_elements = 5000
overlap_count = len(common_elements)
overlap_degree = overlap_count / total_elements

overlap_neuron = torch.tensor(list(common_elements))
torch.save((1,overlap_neuron,1), f'{data_root}/overlap_neuron.pt')
# breakpoint()

print(f"Overlap degree: {overlap_degree:.4f}")

print(f"overlap neuron count is: {overlap_count}")

save_path = '/data1/tsq/zkj_use/Trustworthy-Evaluation/results/neuron_viewer'
if not os.path.exists(save_path):
    os.makedirs(save_path)


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))


color = 'blue'
# _, index, *_ = torch.load(f'{data_root}/overlap_neuron.pt')
# breakpoint()
topk_index = overlap_neuron[:1000]
x = topk_index[:, 0].numpy()
y = topk_index[:, 1].numpy()
plt.scatter(x, y, alpha=0.5, label='overlap neurons', s=10, color=color)  # s参数控制散点大小

# 添加图例
plt.legend()

# 设置标题和轴标签
plt.title('Scatter Plot of Index')
plt.xlabel('X-axis (0-31)')
plt.ylabel('Y-axis (0-11007)')
plt.grid(True)

# 保存图形
file_name = 'final_neurons.png'
full_path = os.path.join(save_path, file_name)
plt.savefig(full_path)

# 显示图形
plt.show()

# 打印保存路径
print(f"The scatter plot has been saved to {full_path}")


