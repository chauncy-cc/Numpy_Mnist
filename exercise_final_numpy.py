import nn
import copy
import numba
import Modules
import argparse
import warnings
import plot_util
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


warnings.filterwarnings('ignore')

# 定义Summary_Writer，数据放在指定文件夹
writer_mlp = SummaryWriter('./Result/numpy/mlp')
writer_lenet = SummaryWriter('./Result/numpy/lenet')

# Global variables
N_CLASS = 10
BATCH_SIZE = 64
PIC_ITERATIONS = 10
LOG_ITERATIONS = 100
IS_RUN_ON_SERVER = False
IS_PYTORCH_VERSION = False


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


def train(epoch):
    log_loss = 0.0
    pic_loss = 0.0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        # forward + backward + step
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        log_loss += loss.item()
        pic_loss += loss.item()
        if batch_idx != 0 and batch_idx % LOG_ITERATIONS == 0:
            print('epoch: %d, batch_idx: %d average_batch_loss: %f'
                  % (epoch + 1, batch_idx, log_loss / LOG_ITERATIONS))
            log_loss = 0.0
        if batch_idx != 0 and batch_idx % PIC_ITERATIONS == 0:
            niter = epoch * len(train_loader) + batch_idx
            writer_lenet.add_scalar('Train/Loss', pic_loss / PIC_ITERATIONS, niter)
            pic_loss = 0.0


def test(epoch):
    correct = 0
    total = 0
    for (inputs, labels) in test_loader:
        inputs, labels = inputs.numpy(), labels.numpy()
        outputs = model(inputs)
        # 取得分最高的那个类
        _, predicted = np.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    print('%dth epoch\'s classification accuracy is: %f%%' % (epoch + 1, (100 * correct / total)))
    writer_lenet.add_scalar('Test/Accu', (100 * correct / total), epoch * len(train_loader))



# 把labels变成one-hot形式
def one_hot(labels, n_class):
    return np.array([[1 if i == l else 0 for i in range(n_class)] for l in labels])

# experiments_task_mlp
experiments_task_mlp = []
model = Modules.MLP()
settings = [(5, 0.0001), (5, 0.005), (5, 0.1)]         # train_epoch && learning_rate
print('Trainging Model_MLP...')
for (num_epochs, learning_rate) in settings:
    # Train
    train_accuracy, test_accuracy, train_loss = [], [], []
    for epoch in range(num_epochs):
        # train
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
                train_loss.append(sum_loss / LOG_ITERATIONS)
                print('epoch: %d, batch_idx: %d average_batch_loss: %f'
                      % (epoch + 1, batch_idx, sum_loss / LOG_ITERATIONS))
                sum_loss = 0.0
            if batch_idx != 0 and batch_idx % PIC_ITERATIONS == 0:
                niter = epoch * len(train_loader) + batch_idx
                writer_mlp.add_scalar('Train/Loss', pic_loss / PIC_ITERATIONS, niter)
                pic_loss = 0.0
        train_accuracy.append(100 * correct / total)
        # test
        correct = 0
        total = 0
        for (inputs, labels) in test_loader:
            inputs, labels = inputs.numpy(), labels.numpy()
            outputs = model.eval(inputs)
            # 取得分最高的那个类
            predicted = np.argmax(outputs, 1)
            total += labels.shape[0]
            correct += (predicted == labels).sum()
        print('%dth epoch\'s classification accuracy is: %f%%' % (epoch + 1, (100 * correct / total)))
        test_accuracy.append(100 * correct / total)
        writer_mlp.add_scalar('Test/Accu', (100 * correct / total), epoch * len(train_loader))
        # torch.save(model.state_dict(), '%s/net_%03d.pth' % (opt.outf, epoch + 1))
    experiments_task_mlp.append(((num_epochs, learning_rate), train_accuracy, test_accuracy, train_loss))

# experiments_task_lenet
experiments_task_lenet = []
model = Modules.LeNet()
settings = [(5, 0.0001), (5, 0.005), (5, 0.1)]         # train_epoch && learning_rate
print('Trainging Model_LeNet..')
for (num_epochs, learning_rate) in settings:
    # Train
    train_accuracy, test_accuracy, train_loss = [], [], []
    for epoch in range(num_epochs):
        # train
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
        # test
        correct = 0
        total = 0
        for (inputs, labels) in test_loader:
            inputs, labels = inputs.numpy(), labels.numpy()
            outputs = model.eval(inputs)
            # 取得分最高的那个类
            predicted = np.argmax(outputs, 1)
            total += labels.shape[0]
            correct += (predicted == labels).sum()
        print('%dth epoch\'s classification accuracy is: %f%%' % (epoch + 1, (100 * correct / total)))
        test_accuracy.append(100 * correct / total)
        # torch.save(model.state_dict(), '%s/net_%03d.pth' % (opt.outf, epoch + 1))
        writer_lenet.add_scalar('Test/Accu', (100 * correct / total), epoch * len(train_loader))
    experiments_task_lenet.append(((num_epochs, learning_rate), train_accuracy, test_accuracy, train_loss))


plot_util.plot_accuracy_curves([experiments_task_mlp, experiments_task_lenet], LOG_ITERATIONS)
plot_util.plot_summary_table([experiments_task_mlp, experiments_task_lenet])
plot_util.plot_loss_curves([experiments_task_mlp, experiments_task_lenet])

