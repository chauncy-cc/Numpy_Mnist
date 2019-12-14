import numpy as np

# 定义交叉熵模型
class CrossEntropyLoss:
    def __init__(self):
        pass

    def forward(self, x, y_):
        batch_loss = 0.0
        for loss in map(lambda y, p: -np.sum((1-y)*np.log(1-p) + y*np.log(p)), y_, x):
            batch_loss += loss
        return batch_loss / len(y_)

    def backward(self, x, y_):
        dx = x.copy()
        dx[range(len(x)), np.argmax(y_, 1)] -= 1
        return dx