import torch
import torch.nn as nn


# 定义一个简单的DNN类
class DNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # 第一层全连接层
        self.fc2 = nn.Linear(128, 64)          # 第二层全连接层
        self.fc3 = nn.Linear(64, output_size)  # 输出层，输出大小为output_size

        self.relu = nn.ReLU()  # 使用ReLU作为激活函数
        # self.sigmoid = nn.Sigmoid()  # 使用Softmax作为输出层的激活函数，dim=1表示在第一个维度上进行Softmax

    def forward(self, x):
        x = self.relu(self.fc1(x))  # 第一层全连接层 + ReLU激活函数
        x = self.relu(self.fc2(x))  # 第二层全连接层 + ReLU激活函数
        x = self.fc3(x)             # 输出层
        # x = self.sigmoid(x)         # 输出层的Softmax激活函数
        return x

