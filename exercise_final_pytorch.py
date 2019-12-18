import torch
import argparse
import warnings
import plot_util
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

warnings.filterwarnings('ignore')

# 定义是否使用GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print("device: " + device.type)
# 定义Summary_Writer，数据放在指定文件夹
writer_mlp = SummaryWriter('./Result/pytorch/mlp')
writer_lenet = SummaryWriter('./Result/pytorch/lenet')

parser = argparse.ArgumentParser()
parser.add_argument('--outf', default='./model_save/pytorch/', help='folder to output images and model checkpoints')   # 模型保存路径
parser.add_argument('--net', default='./model_save/pytorch/net.pth', help="path to netG (to continue training)")       # 模型加载路径
parser.add_argument('--server', default='true', help='is run on server or not')                                        # 是否跑在服务器

opt = parser.parse_args()

# Hyper parameters
MOMENTUM = 0.9
BATCH_SIZE = 64
TRAIN_EPOCHS = 25
PIC_ITERATIONS = 10
LOG_ITERATIONS = 100
IS_PYTORCH_VERSION = True
IS_RUN_ON_SERVER = opt.server
TRAIN_EPOCHS = 10 if IS_RUN_ON_SERVER else 1


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

# 使用LeNet的网络结构
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6,
                      kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16,
                      kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)               # x (64, 28, 28)
        x = self.conv2(x)               # x (64, 6, 14, 14)
        x = x.view(x.size()[0], -1)     # x (64 * 16 * 5 * 5)
        x = self.fc1(x)                 # x (64, 400)
        x = self.fc2(x)                 # x (64, 120)
        x = self.fc3(x)                 # x (64, 84)
        return x                        # x (64, 10)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU()
        )
        self.linear2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = self.linear1(x)
        x = self.linear2(x)
        return x


def train_until_finish(num_epochs, model, optimizer, learning_rate, experiments_task):
    train_accuracy, test_accuracy, train_loss = [], [], []
    for epoch in range(num_epochs):
        correct = 0
        total = 0
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
            predicted = torch.argmax(F.softmax(outputs), 1)           # 算出模型输出
            total += labels.size(0)
            correct += (predicted.numpy() == labels.numpy()).sum()

            if batch_idx != 0 and batch_idx % LOG_ITERATIONS == 0:
                print('epoch: %d, batch_idx: %d average_batch_loss: %f'
                      % (epoch + 1, batch_idx, log_loss / LOG_ITERATIONS))
                log_loss = 0.0
            if batch_idx != 0 and batch_idx % PIC_ITERATIONS == 0:
                niter = epoch * len(train_loader) + batch_idx
                writer_lenet.add_scalar('Train/Loss', pic_loss / PIC_ITERATIONS, niter)
                train_loss.append(pic_loss / PIC_ITERATIONS)
                pic_loss = 0.0
        train_accuracy.append(100 * correct / total)
        test(epoch, test_accuracy, model)
        # torch.save(model.state_dict(), '%s/net_%03d.pth' % (opt.outf, epoch + 1))
    experiments_task.append(((num_epochs, learning_rate), train_accuracy, test_accuracy, train_loss))


def test(epoch, test_accuracy, model):
    with torch.no_grad():
        correct = 0
        total = 0
        for (inputs, labels) in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            # 取得分最高的那个类
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.numpy() == labels.numpy()).sum()
        print('%dth epoch\'s classification accuracy is: %.6f%%' % (epoch + 1, 100 * correct / total))
        test_accuracy.append(100 * correct / total)
        writer_lenet.add_scalar('Test/Accu', (100 * correct / total), epoch * len(train_loader))

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# TODO： 各种优化器、是否分布式、学习率才是超参，epoch可以不在setting中设置
settings = [(1, 0.0001), (1, 0.005), (1, 0.001)]         # train_epoch && learning_rate
experiments_task_mlp = []
experiments_task_lenet = []
for index_setting, (num_epochs, learning_rate) in enumerate(settings):
    model = MLP().to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=MOMENTUM)
    print("model_mlp is initialized. %dth Train setting is %d and %f" % (index_setting + 1, TRAIN_EPOCHS, learning_rate))
    train_until_finish(num_epochs, model=model, optimizer=optimizer, learning_rate=learning_rate, experiments_task=experiments_task_mlp)
    model = LeNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=MOMENTUM)
    print("model_lenet is initialzed. %dth Train setting is %d and %f" % (index_setting + 1, TRAIN_EPOCHS, learning_rate))
    train_until_finish(num_epochs, model=model, optimizer=optimizer, learning_rate=learning_rate, experiments_task=experiments_task_lenet)

plot_util.plot_accuracy_curves([experiments_task_mlp, experiments_task_lenet], IS_RUN_ON_SERVER, IS_PYTORCH_VERSION)
plot_util.plot_summary_table([experiments_task_mlp, experiments_task_lenet], IS_RUN_ON_SERVER, IS_PYTORCH_VERSION)
plot_util.plot_loss_curves([experiments_task_mlp, experiments_task_lenet], LOG_ITERATIONS, IS_RUN_ON_SERVER, IS_PYTORCH_VERSION)

# 来自：https://blog.csdn.net/sunqiande88/article/details/80089941