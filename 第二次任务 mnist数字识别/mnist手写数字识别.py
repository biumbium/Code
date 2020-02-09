# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 20:40:03 2019

@author: 15218
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import time
# Training settings
batch_size = 50
# MNIST Dataset
train_dataset = datasets.MNIST(root='./data/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=False)

test_dataset = datasets.MNIST(root='./data/',
                              train=False,
                              transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
"""
    经典卷积神经网络的结构一般满足如下表达式：
        输出层->（卷积层->池化层）*n->（全连接层）*n
    #池化层可以为多个或零个
"""
class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        
        self.conv1 = nn.Sequential(     # 输入的图片大小为 28*28*1
            nn.Conv2d(1, 6, 5, stride=1, padding=2),    # 输入1通道，输出6通道，卷积核大小5*5，步长为2，填充为1
            nn.ReLU(),      #relu函数求导
            nn.MaxPool2d(2,2)   # 卷积核大小2*2，步长为2
        )
        
        self.conv2 = nn.Sequential(     # 输入的图片大小为 14*14*6
            nn.Conv2d(6, 16, 5, stride=1, padding=0),   # 输入6通道，输出16通道，卷积核大小5*5，步长为1（默认），填充为0（默认）
            nn.ReLU(),      # relu函数求导
            nn.MaxPool2d(2,2)   # 卷积核大小2*2，步长为2
        )
        # 全连接神经网络([(5-2+2*0)/1+1]*[(5-2+2*0)/1+1]*20=320,10个数字）
        self.fc1 = nn.Sequential(            
            nn.Linear(16 * 5 * 5, 120),            
            nn.ReLU()        
        )        
        self.fc2 = nn.Sequential(            
            nn.Linear(120, 84),            
            nn.ReLU()        
        )        
        self.fc3 = nn.Linear(84, 10)
    # 正向传播
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # x: 64*320
        x = x.view(x.size()[0], -1) # 将图像降到一维图像
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
        
"""
                        LeNet-5
    由于LeNet输入的为32*32，而mnist的图像大小为28*28，要使数据大小和网络
    结构大小一致，一般是改网络大小而不改数据的大小。而padding置为2就可以使
    输出为28*28
    
    正向传播过程
    input = (28 * 28 * 1), ff = (5 * 5), stride = 1, padding = 2, filter = 6
    #卷积层
    conv1 = [((28 - 5 + 2 * 2) / 1)+1] * [((28 - 5 + 2 * 2) / 1) +1] * 6 = (28*28*)*6
    pooling1 = [((28-2)/2)+1] * [((28-2)/2)+1] * 6 = (14 * 14 * 6)
    conv2 = [((14 - 5) / 1) + 1] * [((14 - 5) / 1) + 1] * 16 = (10 * 10 * 16)
    pooling2 = [((10 - 2) / 2) + 1] * [((10 - 2) / 2) + 1] * 16 = (5 * 5 * 16)
    #全连接神经网络
    5 * 5 * 16 ——> 1*1*400 ——> 1 * 1 * 120 ——> 1 * 1 * 84 ——> 1 * 1 * 10
"""

def train(epoch):
    model.train()      # 将模型设置为训练模式
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):       # 从数据加载器迭代一个batch的数据
        data, target = Variable(data), Variable(target)     # 自动计算反向传播
        optimizer.zero_grad()   # 清除所有优化的梯度
        output = model(data)    # 输入数据并前向传播获取输出
        loss = F.cross_entropy(output,target)  # 交叉熵损失函数
        loss.backward()     # 反向传播
        optimizer.step()    # 更新参数
        pred = output.data.max(1, keepdim=True)[1]   # 找出每列（索引）概率意义下的最大值
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()   # 统计预测正确个数
        if batch_idx % 50 == 0:        # 每50次（即50 * 50 = 2500个数据），输出一次训练日志
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    print('Accuracy: {}/{} ({:.0f}%)\n'.format(
            correct, len(train_loader.dataset), 100.* correct / len(train_loader.dataset)))


def test():
    model.eval()        # 将模型设置为评估模式
    test_loss = 0
    correct = 0
    for data, target in test_loader:    # 从数据加载器迭代一个batch的数据
        data, target = Variable(data, volatile=True), Variable(target)  # 自动计算反向传播
        output = model(data)    # 输入数据并前向传播获取输出
        # 累加loss
        test_loss += F.cross_entropy(output, target, size_average=False).item()     # 交叉熵损失函数
        pred = output.data.max(1, keepdim=True)[1]   # 找出每列（索引）概率意义下的最大值
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()   # 统计预测正确个数

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__=="__main__":   
    start_time = time.time()
    model = Cnn()   # 实例化网络模型
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)    # 实例化求解器
    for epoch in range(1, 11):
        train(epoch)
    test()
    end_time = time.time()
    print("Total Cost: ",(end_time-start_time))
    torch.save(model, 'model.pth') #保存参数
    torch.save(model.state_dict(), 'parm.pth')

"""
Pytorch保存和加载整个模型：
     torch.save(model, 'model.pth')
     model = torch.load('model.pth')
"""

"""
    Pytorch保存和加载预训练模型参数：
     torch.save(model.state_dict(), 'params.pth')
     model.load_state_dict(torch.load('params.pth'))
"""
