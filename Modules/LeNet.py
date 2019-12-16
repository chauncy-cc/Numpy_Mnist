import nn
import numpy as np

class LeNet:
    def __init__(self):
        # conv layer 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding='SAME')
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        # conv layer 2
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding='SAME')
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.linear1 = nn.Linear(in_features=7 * 7 * 8, out_features=32)
        self.relu3 = nn.ReLU()
        self.linear2 = nn.Linear(in_features=32, out_features=10)
        self.softmax = nn.Softmax()
        self.ce = nn.CrossEntropyLoss()
        print('Model_LeNet constructed!')

    # Train process
    def train(self, x, y_, learning_rate):
        # forward
        f1 = self.conv1.forward(x)
        f2 = self.relu1.forward(f1)
        f3, max_idx1 = self.pool1.forward(f2)
        f4 = self.conv2.forward(f3)
        f5 = self.relu2.forward(f4)
        f6, max_idx2 = self.pool2.forward(f5)
        flat = np.reshape(f6, [-1, 7 * 7 * 8])  # flatten
        f7 = self.linear1.forward(flat)
        f8 = self.relu3.forward(f7)
        f9 = self.linear2.forward(f8)
        f10 = self.softmax.forward(f9)

        predicted = np.argmax(f10, 1)  # 算出模型输出
        loss = self.ce.forward(f10, y_)

        # backward
        b1 = self.ce.backward(f10, y_)
        b2 = self.linear2.backward(b1, f8, learning_rate)
        b3 = self.relu3.backward(b2, f8)
        b4 = self.linear1.backward(b3, flat, learning_rate)
        deflat = np.reshape(b4, [-1, 8, 7, 7])  # de-flatten
        b5 = self.pool2.backward(deflat, max_idx2)
        b6 = self.relu2.backward(b5, f5)
        b7 = self.conv2.backward(b6, f3, learning_rate)
        b8 = self.pool1.backward(b7, max_idx1)
        b9 = self.relu1.backward(b8, f2)
        b10 = self.conv1.backward(b9, x, learning_rate)

        return loss, predicted

    def eval(self, x):
        # forward
        f1 = self.conv1.forward(x)
        f2 = self.relu1.forward(f1)
        f3, max_idx1 = self.pool1.forward(f2)
        f4 = self.conv2.forward(f3)
        f5 = self.relu2.forward(f4)
        f6, max_idx2 = self.pool2.forward(f5)
        flat = np.reshape(f6, [-1, 7 * 7 * 8])  # flatten
        f7 = self.linear1.forward(flat)
        f8 = self.relu3.forward(f7)
        f9 = self.linear2.forward(f8)
        f10 = self.softmax.forward(f9)
        return f10