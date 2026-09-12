import numpy as np  
import torch
import torch.nn.functional as F

# 1. 定义识别到的输入矩阵与卷积核
input_data = np.array([
    [ 2.1, -1.0,  1.0,  4.0,  1.6],
    [ 1.7,  0.0,  1.0, -1.5, -1.4],
    [-2.2,  4.3, -3.0, -2.7,  0.3],
    [-2.0,  0.0,  5.0,  1.1,  0.2],
    [ 2.0, -6.0,  0.0,  0.6, -1.0]
], dtype=np.float32)

kernel_weight = 0.5

# ==============================
# 方法一：NumPy 原生切片验算
# ==============================
# 1x1 卷积（stride=1, 无 padding）等价于逐元素乘以权重
conv_np = input_data * kernel_weight

# MaxPooling (kernel=2x2, stride=2)
# 输出尺寸: floor((5 - 2)/2) + 1 = 2
out_h = (conv_np.shape[0] - 2) // 2 + 1
out_w = (conv_np.shape[1] - 2) // 2 + 1
pool_np = np.zeros((out_h, out_w), dtype=np.float32)

for i in range(out_h):
    for j in range(out_w):
        window = conv_np[i*2 : i*2+2, j*2 : j*2+2]
        pool_np[i, j] = np.max(window)

print("--- [NumPy] 2D Convolution Output (5x5) ---")
print(np.round(conv_np, 4))

print("\n--- [NumPy] MaxPooling Output (2x2) ---")
print(np.round(pool_np, 4))

# ==============================
# 方法二：PyTorch 标准算子验算
# ==============================
# shape: (Batch, Channel, Height, Width) -> (1, 1, 5, 5)
x = torch.tensor(input_data).unsqueeze(0).unsqueeze(0)
w = torch.tensor([[[[kernel_weight]]]])

# 2D 卷积
conv_torch = F.conv2d(x, w, stride=1, padding=0)

# MaxPooling (默认 ceil_mode=False，丢弃不足 2x2 的第 5 行/列)
pool_torch = F.max_pool2d(conv_torch, kernel_size=2, stride=2)

print("\n--- [PyTorch] MaxPooling Output Tensor ---")
print(pool_torch.squeeze().numpy())