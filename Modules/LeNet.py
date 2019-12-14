class LeNet:
    def __init__(self):
        # conv layer 1
        self.conv1 = Conv_Module([8, 1, 3, 3], [8], [1, 1], 'SAME')
        self.relu1 = Relu_Module()
        self.pool1 = Pool_Module([2, 2])
        # conv layer 2
        self.conv2 = Conv_Module([8, 8, 3, 3], [8], [1, 1], 'SAME')
        self.relu2 = Relu_Module()
        self.pool2 = Pool_Module([2, 2])

        self.linear1 = Linear_Module([7 * 7 * 8, 32], [32])
        self.relu3 = Relu_Module()
        self.linear2 = Linear_Module([32, 10], [10])
        self.ce = CE_Module()
        self.softmax = Softmax_Module()

    # Train process
    def train(self, x, y_, learning_rate):
        # forward
        f1 = self.conv1.forward_pass(x)
        f2 = self.relu1.forward_pass(f1)
        f3, max_idx1 = self.pool1.forward_pass(f2)
        f4 = self.conv2.forward_pass(f3)
        f5 = self.relu2.forward_pass(f4)
        f6, max_idx2 = self.pool2.forward_pass(f5)
        flat = np.reshape(f6, [-1, 7 * 7 * 8])  # flatten
        f7 = self.linear1.forward_pass(flat)
        f8 = self.relu3.forward_pass(f7)
        f9 = self.linear2.forward_pass(f8)
        #     f10 = self.ce.forward_pass(f9,y_)
        #     print(f10)

        # backward
        b1 = self.ce.backward_pass(f9, y_)
        b2 = self.linear2.backward_pass(b1, f8, learning_rate)
        b3 = self.relu3.backward_pass(b2, f8)
        b4 = self.linear1.backward_pass(b3, flat, learning_rate)
        deflat = np.reshape(b4, [-1, 8, 7, 7])  # de-flatten
        b5 = self.pool2.backward_pass(deflat, max_idx2)
        b6 = self.relu2.backward_pass(b5, f5)
        b7 = self.conv2.backward_pass(b6, f3, learning_rate)
        b8 = self.pool1.backward_pass(b7, max_idx1)
        b9 = self.relu1.backward_pass(b8, f2)
        b10 = self.conv1.backward_pass(b9, x, learning_rate)

        return

    def eval(self, x, y_):
        # forward
        f1 = self.conv1.forward_pass(x)
        f2 = self.relu1.forward_pass(f1)
        f3, max_idx1 = self.pool1.forward_pass(f2)
        f4 = self.conv2.forward_pass(f3)
        f5 = self.relu2.forward_pass(f4)
        f6, max_idx2 = self.pool2.forward_pass(f5)
        flat = np.reshape(f6, [-1, 7 * 7 * 8])  # flatten
        f7 = self.linear1.forward_pass(flat)
        f8 = self.relu3.forward_pass(f7)
        f9 = self.linear2.forward_pass(f8)
        y = self.softmax.forward_pass(f9)
        # Evaluation
        correct_prediction = np.equal(np.argmax(y_, 1), np.argmax(y, 1))
        accuracy = np.mean(np.array(correct_prediction, dtype=np.float32))
        return accuracy