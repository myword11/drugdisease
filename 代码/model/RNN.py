import torch
import torch.nn as nn

# 定义一个简单的RNN类
class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=32):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)  # RNN层
        self.fc = nn.Linear(hidden_size, output_size)  # 全连接层

        self.relu = nn.ReLU()  # 使用ReLU作为激活函数

    def forward(self, x):
        x = x.unsqueeze(1)
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)  # 初始化隐藏状态
        out, _ = self.rnn(x, h0)  # RNN前向传播
        out = self.fc(self.relu(out[:, -1, :]))  # 取序列最后一个时间步的输出，然后通过全连接层映射到输出大小为output_size
        return out


