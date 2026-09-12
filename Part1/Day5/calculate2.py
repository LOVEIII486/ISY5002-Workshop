import numpy as np
import torch
import torch.nn.functional as F

# 1. 构建输入与卷积核
input_mat = np.array([
    [2.0, 1.0, 0.0, 1.0, 1.0],
    [1.0, 3.0, 0.5, 1.0, 0.0],
    [0.0, 0.5, 1.0, 0.0, 2.0],
    [0.0, 0.5, 1.0, 0.0, 1.0],
    [1.0, 0.0, 1.0, 2.0, 2.0]
], dtype=np.float32)

kernel_mat = np.array([
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 1.0]
], dtype=np.float32)

# ==============================
# 方法一：NumPy 原生滑窗计算
# ==============================
# 卷积尺寸: (5 - 3) / 1 + 1 = 3x3
h_out, w_out = 3, 3
conv_np = np.zeros((h_out, w_out), dtype=np.float32)

for i in range(h_out):
    for j in range(w_out):
        window = input_mat[i:i+3, j:j+3]
        conv_np[i, j] = np.sum(window * kernel_mat)

# MaxPooling (kernel=2x2, stride=2)
# 针对 3x3 特征图，输出尺寸为: floor((3 - 2) / 2) + 1 = 1x1
# 仅覆盖前 2 行与前 2 列（第 3 行与第 3 列因步长和尺寸不足被丢弃）
pool_h = (conv_np.shape[0] - 2) // 2 + 1
pool_w = (conv_np.shape[1] - 2) // 2 + 1
pool_np = np.zeros((pool_h, pool_w), dtype=np.float32)

for i in range(pool_h):
    for j in range(pool_w):
        window = conv_np[i*2 : i*2+2, j*2 : j*2+2]
        pool_np[i, j] = np.max(window)

print("--- [NumPy] 2D Convolution Output (3x3) ---")
print(conv_np)

print("\n--- [NumPy] MaxPooling Output (1x1) ---")
print(pool_np)

# ==============================
# 方法二：PyTorch 标准算子验证
# ==============================
x = torch.tensor(input_mat).unsqueeze(0).unsqueeze(0)
w = torch.tensor(kernel_mat).unsqueeze(0).unsqueeze(0)

conv_torch = F.conv2d(x, w, stride=1, padding=0)
pool_torch = F.max_pool2d(conv_torch, kernel_size=2, stride=2)

print("\n--- [PyTorch] MaxPooling Output Tensor ---")
print(pool_torch.squeeze().numpy())