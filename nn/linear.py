import numpy as np

# 定义线性模型
class Linear:
    def __init__(self, in_features, out_features, bias=True):
        range = np.sqrt(6. / (in_features + out_features))
        self.w = np.random.uniform(-range, range, (in_features, out_features))
        self.b = np.random.uniform(-range, range, (out_features,))

    def forward(self, x):
        y = np.matmul(x, self.w) + self.b
        return y

    def backward(self, dL_dy, x, lr):
        dL_dx = np.matmul(dL_dy, self.w.T)
        # SGD
        # dL/dw = dL/dy * dy/dw = dL/dy * X
        # dL/db = dL/dy * dy/db = dL/dy
        dw = np.matmul(x.T, dL_dy)
        db = np.sum(dL_dy, axis=0)
        self.b -= lr * db
        self.w -= lr * dw
        return dL_dx


# https://blog.csdn.net/zhongshaoyy/article/details/52957794
# https://blog.csdn.net/zhongshaoyy/article/details/52957760