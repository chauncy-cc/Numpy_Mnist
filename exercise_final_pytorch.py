import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# parser = argparse.ArgumentParser()
# parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints') #模型保存路径
# parser.add_argument('--net', default='./model/net.pth', help="path to netG (to continue training)")  #模型加载路径
# opt = parser.parse_args()

# Hyper parameters
TRAIN_EPOCHS = 25
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MOMENTUM = 0.9
LOG_ITERATIONS = 100

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
        # x (64 * 28 * 28)
        x = self.conv1(x)
        # x (64 * 6 * 14 * 14)
        x = self.conv2(x)
        # x (64 * 16 * 7 * 7)   应 (64 * 16 * 5 * 5) ！！！看看这里是哪里算错了
        x = x.view(x.size()[0], -1)
        # x (64 * 400)
        x = self.fc1(x)
        # x (64 * 120)
        x = self.fc2(x)
        # x (64 * 84)
        x = self.fc3(x)
        # x (64 * 10)
        return x

# 定义损失函数和优化方式
model = LeNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),
                      lr=LEARNING_RATE,
                      momentum=MOMENTUM)

def train(epoch):
    sum_loss = 0.0  # 用来每LOG_ITERATIONS打印一次平均loss
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        # 梯度清零
        optimizer.zero_grad()
        # forward + backward + step
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()
        if batch_idx != 0 and batch_idx % LOG_ITERATIONS == 0:
            print('epoch: %d, batch_idx: %d average_batch_loss: %f'
                  % (epoch + 1, batch_idx, sum_loss / LOG_ITERATIONS))
            sum_loss = 0.0


def test(epoch):
    with torch.no_grad():
        correct = 0
        total = 0
        for (inputs, labels) in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            # 取得分最高的那个类
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        if epoch == 0:
            print('模型初始化后训练前的测试集识别准确率为：%f%%' % (100 * correct / total))
        else:
            print('第%d个epoch的识别准确率为：%d%%' % (epoch + 1, (100 * correct / total)))

test(0)
for epoch in range(TRAIN_EPOCHS):
    train(epoch)
    test(epoch)
    # torch.save(model.state_dict(), '%s/net_%03d.pth' % (opt.outf, epoch + 1))


# 来自：https://blog.csdn.net/sunqiande88/article/details/80089941
# TODO： pytorch版本如何在线监控loss的变化，以及对loss曲线画图等等，需要研究一番。