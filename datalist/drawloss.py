import pandas as pd
import matplotlib.pyplot as plt

# 假设数据存储在当前目录下的'training_data.csv'文件中
# 如果数据以字符串形式给出，首先需要保存到一个.csv文件中，这里假设数据已经在文件中
# data = pd.read_csv(r"D:\projects\MyCSPN\output\sgd0521_lr1e-3_step6_mynet_layer2_rgb_nosparse_mae_normalize_gradclip_sparsemask_dynamicmask_refinergbedit_noinit_clamp\log_train.csv")

# # 打印数据以确认其结构
# print(data.head())

# # 绘制折线图
# plt.figure(figsize=(10, 5))  # 设置图表大小
# plt.plot(data['epoch'], data['loss_num'], marker='o', linestyle='-')  # 绘制折线图，点标记每个数据点
# plt.title('Training Loss vs. Epoch')  # 设置图表标题
# plt.xlabel('Epoch')  # 设置x轴标签
# plt.ylabel('Loss')  # 设置y轴标签
# plt.grid(True)  # 显示网格
# plt.show()  # 显示图表


# 读取训练损失数据
train_data = pd.read_csv(r"D:\projects\MyCSPN\output\sgd0521_lr1e-3_step6_mynet_layer2_rgb_nosparse_mae_normalize_gradclip_sparsemask_dynamicmask_refinergbedit_noinit_clamp\log_train.csv")

# 读取验证损失数据
eval_data = pd.read_csv(r"D:\projects\MyCSPN\output\sgd0521_lr1e-3_step6_mynet_layer2_rgb_nosparse_mae_normalize_gradclip_sparsemask_dynamicmask_refinergbedit_noinit_clamp\log_eval.csv")

# 绘制折线图
plt.figure(figsize=(10, 5))  # 设置图表大小

# 绘制训练损失曲线
plt.plot(train_data['epoch'], train_data['loss_num'], marker='o', linestyle='-', label='Training Loss')

# 绘制验证损失曲线
plt.plot(eval_data['epoch'], eval_data['loss_num'], marker='x', linestyle='-', label='Validation Loss')

# 添加图表元素
plt.title('Training & Validation Loss vs. Epoch')  # 设置图表标题
plt.xlabel('Epoch')  # 设置x轴标签
plt.ylabel('Loss')  # 设置y轴标签
plt.grid(True)  # 显示网格
plt.legend()  # 显示图例，用于区分不同的曲线

plt.show()  # 显示图表
