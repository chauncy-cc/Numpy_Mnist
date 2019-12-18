import nn
import copy
import numba
import Modules
import argparse
import warnings
import plot_util
import numpy as np
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

warnings.filterwarnings('ignore')

# 定义Summary_Writer，数据放在指定文件夹
writer_mlp = SummaryWriter('./Result/numpy/mlp')
writer_lenet = SummaryWriter('./Result/numpy/lenet')

parser = argparse.ArgumentParser()
parser.add_argument('--outf', default='./model_save/pytorch/', help='folder to output images and model checkpoints')   # 模型保存路径
parser.add_argument('--net', default='./model_save/pytorch/net.pth', help="path to netG (to continue training)")       # 模型加载路径
parser.add_argument('--server', default='true', help='is run on server or not')                                        # 是否跑在服务器
opt = parser.parse_args()

# Global variables
N_CLASS = 10
BATCH_SIZE = 64
PIC_ITERATIONS = 10
LOG_ITERATIONS = 100
IS_RUN_ON_SERVER = True
IS_PYTORCH_VERSION = False
TRAIN_EPOCHS = 1 if IS_RUN_ON_SERVER else 1

train_set = datasets.MNIST('./data',
                           train=True,
                           transform=transforms.ToTensor(),
                           download=True)

test_set = datasets.MNIST('./data',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)

# Data Loader
train_loader = DataLoader(dataset=train_set,
                          batch_size=BATCH_SIZE,
                          shuffle=True)

test_loader = DataLoader(dataset=test_set,
                         batch_size=BATCH_SIZE,
                         shuffle=False)


def train_until_finish(num_epochs, model, learning_rate, experiments_task):
    train_accuracy, test_accuracy, train_loss = [], [], []
    for epoch in range(num_epochs):
        correct = 0
        total = 0
        sum_loss = 0.0  # 用来每LOG_ITERATIONS打印一次平均loss
        pic_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.numpy(), labels.numpy()
            labels_one_hot = one_hot(labels, N_CLASS)  # one-hot
            loss, predicted = model.train(inputs, labels_one_hot, learning_rate)
            total += labels.shape[0]
            correct += (predicted == labels).sum()
            sum_loss += loss
            pic_loss += loss
            if batch_idx != 0 and batch_idx % LOG_ITERATIONS == 0:
                print('epoch: %d, batch_idx: %d average_batch_loss: %f'
                      % (epoch + 1, batch_idx, sum_loss / LOG_ITERATIONS))
                sum_loss = 0.0
            if batch_idx != 0 and batch_idx % PIC_ITERATIONS == 0:
                niter = epoch * len(train_loader) + batch_idx
                writer_lenet.add_scalar('Train/Loss', pic_loss / PIC_ITERATIONS, niter)
                train_loss.append(pic_loss / PIC_ITERATIONS)
                pic_loss = 0.0
        train_accuracy.append(100 * correct / total)
        test(epoch, test_accuracy, model)
    experiments_task.append(((num_epochs, learning_rate), train_accuracy, test_accuracy, train_loss))


def test(epoch, test_accuracy, model):
    correct = 0
    total = 0
    for (inputs, labels) in test_loader:
        inputs, labels = inputs.numpy(), labels.numpy()
        outputs = model.eval(inputs)
        # 取得分最高的那个类
        predicted = np.argmax(outputs, 1)
        total += labels.shape[0]
        correct += (predicted == labels).sum()
    print('%dth epoch\'s classification accuracy is: %.6f%%' % (epoch + 1, 100 * correct / total))
    test_accuracy.append(100 * correct / total)
    writer_mlp.add_scalar('Test/Accu', (100 * correct / total), epoch * len(train_loader))


# 把labels变成one-hot形式
def one_hot(labels, n_class):
    return np.array([[1 if i == l else 0 for i in range(n_class)] for l in labels])


settings = [(0, 0, 0.0001), (0, 0, 0.005), (0, 0, 0.001),
            (0, 1, 0.0001), (0, 1, 0.005), (0, 1, 0.001),
            (1, 0, 0.0001), (1, 0, 0.005), (1, 0, 0.001),]
experiments_task_mlp = []
experiments_task_lenet = []
for index_setting, (_, _, learning_rate) in enumerate(settings):
    model = Modules.MLP()
    print("model_mlp is initialized. %dth Train setting is %d and %f" % (index_setting + 1, TRAIN_EPOCHS, learning_rate))
    train_until_finish(TRAIN_EPOCHS, model=model, learning_rate=learning_rate, experiments_task=experiments_task_mlp)
    model = Modules.LeNet()
    print("model_lenet is initialzed. %dth Train setting is %d and %f" % (index_setting + 1, TRAIN_EPOCHS, learning_rate))
    train_until_finish(TRAIN_EPOCHS, model=model, learning_rate=learning_rate, experiments_task=experiments_task_lenet)


plot_util.plot_accuracy_curves([experiments_task_mlp, experiments_task_lenet], IS_RUN_ON_SERVER, IS_PYTORCH_VERSION)
plot_util.plot_summary_table([experiments_task_mlp, experiments_task_lenet], IS_RUN_ON_SERVER, IS_PYTORCH_VERSION)
plot_util.plot_loss_curves([experiments_task_mlp, experiments_task_lenet], LOG_ITERATIONS, IS_RUN_ON_SERVER, IS_PYTORCH_VERSION)

