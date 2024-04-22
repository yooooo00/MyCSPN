import torch
import time

# 检查CUDA是否可用，并选择设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 创建两个随机矩阵
size = 5000
a = torch.randn(size, size, device=device)
b = torch.randn(size, size, device=device)

# 热身运行，确保CUDA启动完成
if device.type == 'cuda':
    print("Warming up CUDA")
    for _ in range(10):
        _ = a @ b

# 执行并计时矩阵乘法
start_time = time.time()
c = a @ b
elapsed_time = time.time() - start_time

# 确保计算被执行
torch.cuda.synchronize() if device.type == 'cuda' else None

print(f"Time to perform matrix multiplication on {device}: {elapsed_time:.4f} seconds")
print(f"Result tensor: {c}")

# 检查并打印显存使用情况
if device.type == 'cuda':
    print(torch.cuda.memory_summary())
