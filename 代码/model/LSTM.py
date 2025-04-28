import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = output_size

        # 定义LSTM层
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=output_size, batch_first=True)

        # 定义输出层
        self.output_layer = nn.Linear(output_size, 1)

    def forward(self, x):
        # 调整输入数据格式为 (batch_size, seq_len, input_size)
        x = x.unsqueeze(1)  # (batch_size, 1, input_size)

        # LSTM层
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, 1, output_size)
        lstm_out = lstm_out.squeeze(1)  # (batch_size, output_size)

        # 输出层
        output = self.output_layer(lstm_out)  # (batch_size, output_size)
        return output
