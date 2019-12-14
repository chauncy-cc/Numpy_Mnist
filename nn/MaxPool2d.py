# 定义池化层
class MaxPool2d:
    def __init__(self, pool_size):
        print('MaxPool2d constructed')
        self.ps = pool_size

    def forward_pass(self, x):
        pass

    def backward_pass(self, dL_dy, max_idx):
        pass