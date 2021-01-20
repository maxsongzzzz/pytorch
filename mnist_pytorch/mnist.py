# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 14:17:05 2021

@author: Max
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data

# 神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.conv2 = nn.Conv2d(16, 64, 3)
        self.fc1 = nn.Linear(5*5*64, 200)
        self.fc2 = nn.Linear(200, 10)
    def forward(self,x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x,2,2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x,2,2)
        x = x.view(x.size()[0],-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def loaddatasets(path,batchSize):
    x = np.loadtxt(path,delimiter=',',unpack=False,usecols=range(1,785)) 
    x = x.reshape(-1,28,28)
    x = torch.from_numpy(x)
    x = x.float()
    x = torch.unsqueeze(x,dim=1) 
    y = np.loadtxt(path,delimiter=',',usecols=0,unpack=False)   # 从csv读取数据
    y = torch.from_numpy(y)
    datasets = Data.TensorDataset(x,y)
    data_loader = Data.DataLoader(dataset=datasets,batch_size=batchSize,shuffle=True)
    return data_loader

def train(model,loader,epochs):
    model = model.cuda()
    cost = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(),lr=0.01)
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(epochs):
        running_loss = 0.0
        running_correct = 0
        for step,(data_x,data_y) in enumerate(loader):
            data_x = data_x/255
            data_y = data_y.long()
            x, y = Variable(data_x), Variable(data_y)
            outputs = model(x.cuda())
            pred = torch.argmax(outputs.data, dim=1)
            optimizer.zero_grad()
            loss = cost(outputs, y.cuda()) 
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_correct += torch.sum(pred == y.cuda().data)
            if step % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.2f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, step * len(x), len(loader.dataset),
                           100. * running_correct/len(loader.dataset),running_loss/len(loader.dataset)))
        print("Loss is:{:.6f}, Train Accuracy is:{:.2f}%".format(running_loss/len(loader.dataset),100*running_correct/len(loader.dataset)))

def test(model,loader):
    correct_test = 0
    for setp,(test_x,test_y) in enumerate(loader):
        test_x = test_x/255
        outputs_test = model(test_x.cuda())
        pred_test = torch.argmax(outputs_test.data,dim=1)
        correct_test += torch.sum(pred_test == test_y.cuda())
    print('Test Accuracy is:{:.4f}%'.format(100*correct_test/len(loader.dataset)))
        
# 网络
model = Net()

# 数据
path_train = "mnist_train.csv"
path_test = "mnist_test.csv"
loader_train = loaddatasets(path_train, 20)
loader_test = loaddatasets(path_test, 500)

# 训练 & 测试
train(model,loader_train,5)
test(model,loader_test)
