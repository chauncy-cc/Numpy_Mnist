import numpy as np
import nn

# 多层感知机模型
class MLP:
    def __init__(self):
        self.linear1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, 10)
        self.ce = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax()

    # Train process TO DO: train的时候将loss返回出来
    def train(self, x, y_, learning_rate):
        # forward
        flat = np.reshape(x, [-1, 28 * 28])
        f1 = self.linear1.forward(flat)
        f2 = self.relu.forward(f1)
        f3 = self.linear2.forward(f2)
        # backward
        b1 = self.ce.backward(f3, y_)
        b2 = self.linear2.backward(b1, f2, learning_rate)
        b3 = self.relu.backward(b2, f1)
        b5 = self.linear1.backward(b3, flat, learning_rate)
        return

    # eval process
    def eval(self, x):
        # forward
        flat = np.reshape(x, [-1, 28 * 28])
        f1 = self.linear1.forward(flat)
        f2 = self.relu.forward(f1)
        f3 = self.linear2.forward(f2)
        f4 = self.softmax.forward(f3)
        return f4