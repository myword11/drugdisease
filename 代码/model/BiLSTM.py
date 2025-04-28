import torch
import torch.nn as nn

# 定义一个双向LSTM类
class BiLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size = 32):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)  # 双向LSTM层
        self.fc = nn.Linear(hidden_size * 2, output_size)  # 全连接层，注意输出维度要乘以2

        self.relu = nn.ReLU()  # 使用ReLU作为激活函数

    def forward(self, x):
        x = x.unsqueeze(1)
        h0 = torch.zeros(2, x.size(0), self.hidden_size).to(x.device)  # 初始化LSTM的隐藏状态h0，注意是2倍的hidden_size
        c0 = torch.zeros(2, x.size(0), self.hidden_size).to(x.device)  # 初始化LSTM的cell状态c0，注意是2倍的hidden_size

        out, _ = self.lstm(x, (h0, c0))  # 双向LSTM前向传播
        out = self.fc(self.relu(out[:, -1, :]))  # 取序列最后一个时间步的输出，然后通过全连接层映射到输出大小为output_size
        return out


