import numpy as np

# 定义池化层
class MaxPool2d:
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def forward(self, x):
        (samples, cx, hx, wx) = x.shape
        hy = int(hx / self.kernel_size)
        wy = int(wx / self.kernel_size)
        y = np.zeros([samples, cx, hy, wy])
        x_reshaped = np.reshape(x, (samples * cx * hx, wx))

        # change x_reshaped to rows*4 matrix
        for j1 in range(self.kernel_size):
            for j2 in range(self.kernel_size):
                tmp = x_reshaped[j1::self.kernel_size, j2::self.kernel_size]
                col = np.reshape(tmp, (-1, 1))
                if (j1 == 0 and j2 == 0):
                    x_col = col
                else:
                    x_col = np.concatenate((x_col, col), axis=1)

        max_idx = np.argmax(x_col, axis=1)
        y = x_col[range(len(max_idx)), max_idx]
        y = np.reshape(y, (samples, cx, hy, -1))
        return y, max_idx

    def backward(self, dL_dy, max_idx):
        (samples, cy, hy, wy) = dL_dy.shape
        dy_reshaped = np.reshape(dL_dy, (-1, 1))
        hx = int(hy * self.kernel_size)
        wx = int(wy * self.kernel_size)

        dy_reshaped = np.transpose(dy_reshaped, (1, 0))
        dx_reshaped = np.zeros([self.kernel_size * self.kernel_size, len(max_idx)])
        dx_reshaped[max_idx, range(len(max_idx))] = dy_reshaped
        dx_reshaped = np.transpose(dx_reshaped, (1, 0))

        img = np.zeros([samples * cy * hx, wx])
        # change dx_reshaped to a 4-dim matrix dL_dx
        for j1 in range(self.kernel_size):
            for j2 in range(self.kernel_size):
                col = dx_reshaped[:, j1 * self.kernel_size + j2]
                tmp = img[j1::self.kernel_size, j2::self.kernel_size]
                img[j1::self.kernel_size, j2::self.kernel_size] = np.reshape(col, (tmp.shape))

        dL_dx = np.reshape(img, (samples, cy, hx, wx))
        return dL_dx