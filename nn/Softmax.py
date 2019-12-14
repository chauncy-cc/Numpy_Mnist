import numpy as np

# 定义SoftMax激活函数
class Softmax:
    def __init__(self):
        print('Softmax constructed')

    def forward(self, x):
        y = np.exp(x - np.max(x, axis=1, keepdims=True))  # 利用最大值进行数据缩放，避免溢出，softmax必备
        y /= np.sum(y, axis=1, keepdims=True)             # 缩放不影响结果，仅为避免溢出
        return y

    def backward(self):
        pass