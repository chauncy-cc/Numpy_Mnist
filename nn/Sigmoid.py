import numpy as np

# 定义sigmoid激活函数
class Sigmoid:
    def __init__(self):
        pass

    def forward(self, x):
        y = 1 / (1 + np.exp(-x))
        return y

    def backward(self, dL_dy, y):
        pass