import numpy as np

# 定义卷积层模型
class Conv2d:
    def __init__(self, shape_w, shape_b, strides = [1, 1], padding='SAME'):
        print('Conv2d constructed')
        range_w = np.sqrt(6. / (shape_w[0] + shape_w[1] + 1))
        range_b = np.sqrt(6. / (shape_b[0] + 1))
        self.w = np.random.uniform(-range_w, range_w, shape_w)
        self.b = np.random.uniform(-range_b, range_b, shape_b)
        self.strides = strides
        self.padding = padding

    def forward_pass(self, x):
        pass

    def backward_pass(self, dL_dy, x, lr):
        pass