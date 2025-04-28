import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义一个简单的DNN类
class DNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # 第一层全连接层
        self.fc2 = nn.Linear(128, 64)  # 第二层全连接层
        self.fc3 = nn.Linear(64, output_size)  # 输出层，输出大小为output_size
        self.relu = nn.ReLU()  # 使用ReLU作为激活函数

    def forward(self, x):
        x = self.relu(self.fc1(x))  # 第一层全连接层 + ReLU激活函数
        x = self.relu(self.fc2(x))  # 第二层全连接层 + ReLU激活函数
        x = self.fc3(x)  # 输出层
        return x

# 定义LSTM模型
class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch_size, 1, input_size)
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, 1, hidden_size)
        lstm_out = lstm_out.squeeze(1)  # (batch_size, hidden_size)
        output = self.output_layer(lstm_out)  # (batch_size, output_size)
        return output

# 定义RNN模型
class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=32):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(self.relu(out[:, -1, :]))
        return out

# 定义TextRCNN类
class TextRCNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
        super(TextRCNN, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        self.conv = nn.Conv1d(in_channels=input_size + 2 * hidden_size, out_channels=100, kernel_size=2, padding=1)
        self.fc = nn.Linear(100, output_size)

    def forward(self, x):
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, seq_len, 2*hidden_size)
        x_combined = torch.cat((x, lstm_out), 2)  # (batch_size, seq_len, input_size + 2*hidden_size)
        x_combined = x_combined.transpose(1, 2)  # (batch_size, input_size + 2*hidden_size, seq_len)
        x = F.relu(self.conv(x_combined))  # (batch_size, 100, seq_len)
        x = F.max_pool1d(x, kernel_size=x.size(2)).squeeze(2)  # (batch_size, 100)
        output = self.fc(x)  # (batch_size, output_size)
        return output

# 定义集成模型，包含DNN、LSTM、RNN和TextRCNN
class DNNLSTMRNNTextRCNNAttation(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
        super(DNNLSTMRNNTextRCNNAttation, self).__init__()
        self.dnn = DNN(input_size, output_size)
        self.lstm = LSTM(input_size, output_size, hidden_size)
        self.rnn = RNN(input_size, output_size, hidden_size)
        self.text_rcnn = TextRCNN(3 * output_size, output_size, hidden_size)

    def forward(self, x):
        # DNN的输出
        dnn_output = self.dnn(x)  # (batch_size, output_size)
        # LSTM的输出
        lstm_output = self.lstm(x)  # (batch_size, output_size)
        # RNN的输出
        rnn_output = self.rnn(x)  # (batch_size, output_size)
        # 将DNN、LSTM和RNN的输出拼接
        combined_output = torch.cat((dnn_output, lstm_output, rnn_output), dim=1)  # (batch_size, 3 * output_size)
        # 将拼接后的输出输入到TextRCNN中
        text_rcnn_output = self.text_rcnn(combined_output.unsqueeze(1))  # (batch_size, 1, 3 * output_size)
        return text_rcnn_output

