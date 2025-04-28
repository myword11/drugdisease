import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=4):
        super(AttentionLSTM, self).__init__()
        self.hidden_size = hidden_size

        # 定义LSTM层
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)

        # 定义注意力机制的全连接层
        self.attention = nn.Linear(hidden_size, 1)

        # 定义输出层
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 调整输入数据格式为 (batch_size, seq_len, input_size)
        x = x.unsqueeze(1)  # (batch_size, 1, input_size)

        # LSTM层
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, seq_len, hidden_size)

        # 注意力机制
        attn_weights = torch.tanh(self.attention(lstm_out))  # (batch_size, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)  # (batch_size, seq_len, 1)
        attn_applied = torch.sum(lstm_out * attn_weights, dim=1)  # (batch_size, hidden_size)

        # 输出层
        output = self.output_layer(attn_applied)  # (batch_size, output_size)
        return output
