import torch
import torch.nn as nn
import torch.nn.functional as F


class TextRCNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
        super(TextRCNN, self).__init__()

        self.hidden_size = hidden_size

        # 定义双向LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, bidirectional=True, batch_first=True)

        # 定义卷积层
        self.conv = nn.Conv1d(in_channels=2 * hidden_size + input_size, out_channels=100, kernel_size=2, padding=1)

        # 定义全连接层
        self.fc = nn.Linear(100, output_size)

    def forward(self, x):
        # 调整输入数据格式为 (batch_size, seq_len, input_size)
        x = x.unsqueeze(1)  # (batch_size, 1, input_size)

        # 双向LSTM层
        self.lstm.flatten_parameters()  # 确保LSTM参数在内存中是连续的
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, seq_len, 2*hidden_size)

        # 拼接输入和LSTM输出
        x_combined = torch.cat((x, lstm_out), 2)  # (batch_size, seq_len, input_size + 2*hidden_size)

        # 调整维度以适应卷积层
        x_combined = x_combined.transpose(1, 2)  # (batch_size, input_size + 2*hidden_size, seq_len)

        # 卷积层和池化层
        x = F.relu(self.conv(x_combined))  # (batch_size, 100, seq_len)
        x = F.max_pool1d(x, kernel_size=x.size(2)).squeeze(2)  # (batch_size, 100)

        # 全连接层
        output = self.fc(x)  # (batch_size, output_size)
        return output



