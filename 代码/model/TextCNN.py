import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(TextCNN, self).__init__()

        # 定义卷积层
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=100, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=100, kernel_size=4, padding=2)
        self.conv3 = nn.Conv1d(in_channels=1, out_channels=100, kernel_size=5, padding=2)

        # 定义全连接层
        self.fc = nn.Linear(300, output_size)

    def forward(self, x):
        # 添加通道维度
        x = x.unsqueeze(1)  # (batch_size, 1, input_size)

        # 卷积层和池化层
        x1 = F.relu(self.conv1(x))  # (batch_size, 100, input_size)
        x1 = F.max_pool1d(x1, kernel_size=x1.size(2)).squeeze(2)  # (batch_size, 100)

        x2 = F.relu(self.conv2(x))  # (batch_size, 100, input_size)
        x2 = F.max_pool1d(x2, kernel_size=x2.size(2)).squeeze(2)  # (batch_size, 100)

        x3 = F.relu(self.conv3(x))  # (batch_size, 100, input_size)
        x3 = F.max_pool1d(x3, kernel_size=x3.size(2)).squeeze(2)  # (batch_size, 100)

        # 拼接所有卷积核的输出
        x = torch.cat((x1, x2, x3), 1)  # (batch_size, 300)

        # 全连接层
        output = self.fc(x)  # (batch_size, output_size)
        return output


