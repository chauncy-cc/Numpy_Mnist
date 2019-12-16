import numpy as np
import copy

# 定义卷积层模型
class Conv2d:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='SAME'):
        range_w = np.sqrt(6. / (in_channels + out_channels))
        range_b = np.sqrt(6. / (in_channels + out_channels))
        self.w = np.random.uniform(-range_w, range_w, (out_channels, in_channels, kernel_size, kernel_size))
        self.b = np.random.uniform(-range_b, range_b, (out_channels))
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        out_channels, in_channels, kernel_size, kernel_size = self.w.shape
        batch_size, in_channels, x_h, x_w = x.shape

        stride = self.stride
        padding = self.padding

        if padding == 'SAME':
            pad_h = int((kernel_size - stride + x_h * (stride - 1)) / 2.)
            pad_w = int((kernel_size - stride + x_w * (stride - 1)) / 2.)
        else:
            pad_h, pad_w = int(padding), int(padding)

        y_h = int((x_h + 2 * pad_h - kernel_size) / stride) + 1
        y_w = int((x_w + 2 * pad_w - kernel_size) / stride) + 1

        # Add padding around each 2D image(x有4个维度，前2维不扩充,填充constant模式默认值为0)
        padded = np.pad(x, [(0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)], 'constant')

        # change padded to -1,(kernel_size*kernel_size*in_channels) matrix
        for j1 in range(in_channels):
            for j2 in range(kernel_size):
                for j3 in range(kernel_size):
                    tmp = padded[:, j1:j1+1, j2:(x_h+2*pad_h+j2-kernel_size+1):stride,
                          j3:(x_w+2*pad_w+j3-kernel_size+1):stride]
                    col = np.reshape(tmp, (-1, 1))
                    if (j1 == 0 and j2 == 0 and j3 == 0):
                        x_col = col
                    else:
                        x_col = np.concatenate((x_col, col), axis=1)

        w_reshaped = np.reshape(self.w, [out_channels, -1])
        w_reshaped = np.transpose(w_reshaped, (1, 0))
        y = np.matmul(x_col, w_reshaped) + self.b

        y = np.reshape(y, (batch_size, y_h, y_w, out_channels))
        y = np.transpose(y, (0, 3, 1, 2))

        return y

    def backward(self, dL_dy, x, lr):
        out_channels, in_channels, kernel_size, kernel_size = self.w.shape
        batch_size, in_channels, x_h, x_w = x.shape

        stride = self.stride
        padding = self.padding

        if padding == 'SAME':
            pad_h = int((kernel_size - stride + x_h * (stride - 1)) / 2.)
            pad_w = int((kernel_size - stride + x_w * (stride - 1)) / 2.)
        else:
            pad_h, pad_w = 0, 0

        y_h = int((x_h + 2 * pad_h - kernel_size) / stride) + 1
        y_w = int((x_w + 2 * pad_w - kernel_size) / stride) + 1

        dx = np.zeros_like(x)
        dw = np.zeros_like(self.w)
        db = np.zeros_like(self.b)

        b = copy.deepcopy(self.b)
        w_reshaped = np.reshape(copy.deepcopy(self.w), [out_channels, -1])

        padded = np.pad(x, [(0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)], 'constant')
        padded_dx = np.pad(dx, [(0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)], 'constant')

        # change padded to -1,(kernel_size*kernel_size*in_channels) matrix
        for j1 in range(in_channels):
            for j2 in range(kernel_size):
                for j3 in range(kernel_size):
                    tmp = padded[:, j1:j1 + 1, j2:(x_h + 2 * pad_h + j2 - kernel_size + 1):stride,
                          j3:(x_w + 2 * pad_w + j3 - kernel_size + 1):stride]
                    col = np.reshape(tmp, (-1, 1))
                    if (j1 == 0 and j2 == 0 and j3 == 0):
                        x_col = col
                    else:
                        x_col = np.concatenate((x_col, col), axis=1)

        y2 = np.transpose(dL_dy, (0, 2, 3, 1))
        y2 = np.reshape(y2, (-1, out_channels))

        dx_col = np.matmul(y2, w_reshaped)

        for j1 in range(in_channels):
            for j2 in range(kernel_size):
                for j3 in range(kernel_size):
                    col_ind = j1 * kernel_size * kernel_size + j2 * kernel_size + j3
                    col = dx_col[:, col_ind:col_ind + 1]
                    block = np.reshape(col, [-1, 1, y_h, y_w])
                    padded_dx[:, j1:j1 + 1, j2:(x_h + 2 * pad_h + j2 - kernel_size + 1):stride,
                    j3:(x_w + 2 * pad_w + j3 - kernel_size + 1):stride] += block

        # Unpad
        dx = padded_dx[:, :, pad_h:pad_h + x_h, pad_w:pad_w + x_w]

        dout = np.transpose(dL_dy, (0, 2, 3, 1))
        dout = np.reshape(dout, (-1, out_channels))

        w_reshaped -= lr * np.matmul(dout.T, x_col)
        b -= lr * np.sum(dout.T, axis=1)

        w = np.reshape(w_reshaped, [out_channels, in_channels, kernel_size, kernel_size])

        self.w = w
        self.b = b

        return dx