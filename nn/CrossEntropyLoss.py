import numpy as np

# 定义交叉熵模型
class CrossEntrpyLoss:
    def __init__(self):
        print('CrossEntropyLoss constructed')

    # 交叉熵的写法（y_为标签）
    def forward(self, x, y_):
        y = np.sum(np.log((np.sum(np.exp(x), axis=1))) - x[range(len(x)), np.argmax(y_, axis=1)])
        # !!! 如果x, y_都是一维那就好写很多，二维的话，理解一下上式 !!! 上式为何要exp。
        # sum = 0.0
        # for x in map(lambda y, p: (1-y)*math.log(1-p) + y*math.log(p), Y, P):
        #       sum += x
        # return -sum / len(Y)
        return y

    def backward(self, x, y_):
        probs = np.exp(x - np.max(x, axis=1, keepdims=True))
        probs /= np.sum(probs, axis=1, keepdims=True)
        dx = probs.copy()
        dx[range(len(x)), np.argmax(y_, 1)] -= 1
        return dx