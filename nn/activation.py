import numpy as np

# 各种激活函数对应的class
# 目前有：ReLU, Sigmoid, Softmax, Tahn


# 定义Relu非线性激活函数
class ReLU:
    def __init__(self):
        pass

    def forward(self, x):
        y = x * (x > 0)
        return y

    def backward(self, dL_dy, y):
        dL_dx = dL_dy * (y > 0)
        return dL_dx


# 定义sigmoid激活函数
class Sigmoid:
    def __init__(self):
        pass

    def forward(self, x):
        y = 1 / (1 + np.exp(-x))
        return y

    def backward(self, dL_dy, y):
        pass


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