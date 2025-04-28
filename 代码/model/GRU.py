import torch
import torch.nn as nn


# 定义GRU单元
class GRU(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=32):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)  # GRU层
        self.fc = nn.Linear(hidden_size, output_size)  # 全连接层

        self.relu = nn.ReLU()  # 使用ReLU作为激活函数

    def forward(self, x):
        x = x.unsqueeze(1)
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)  # 初始化GRU的隐藏状态h0
        out, _ = self.gru(x, h0)  # GRU前向传播
        out = self.fc(self.relu(out[:, -1, :]))  # 取序列最后一个时间步的输出，然后通过全连接层映射到输出大小为output_size
        return out



