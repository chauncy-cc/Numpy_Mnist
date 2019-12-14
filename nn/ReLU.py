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