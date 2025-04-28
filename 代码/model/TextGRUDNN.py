import torch
import torch.nn as nn
import torch.nn.functional as F

class TextGRUDNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
        super(TextGRUDNN, self).__init__()
        self.hidden_size = hidden_size
        # 定义双向GRU层
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        # 定义DNN层
        self.dnn = nn.Sequential(
            nn.Linear(2 * hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        # 定义全连接层
        self.fc = nn.Linear(32, output_size)

    def forward(self, x):
        # 如果数据是 (batch_size, input_size)，手动加一个时间维度
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch_size, 1, input_size)
        # 通过双向GRU层
        self.gru.flatten_parameters()  # 确保GRU参数在内存中是连续的
        gru_out, _ = self.gru(x)  # gru_out: (batch_size, seq_len, 2*hidden_size)
        # 取最后一个时间步的输出
        x = gru_out[:, -1, :]  # (batch_size, 2*hidden_size)
        # 通过DNN层
        x = self.dnn(x)  # (batch_size, 32)
        # 通过全连接层
        output = self.fc(x)  # (batch_size, output_size)
        return output

