import numpy as np

# 定义tanh激活函数
class Tanh:
    def __init__(self):
        print('Tanh constructed')

    def forward(self, x):
        y1 = np.exp(x) - np.exp(-x)
        y2 = np.exp(x) + np.exp(-x)
        return y1 / y2

    def backward(self, dL_dy, y):
        pass