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


# 定义注意力机制层
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_out):
        # lstm_out: (batch_size, seq_len, 2*hidden_size)
        attention_scores = self.attention(lstm_out).squeeze(2)  # (batch_size, seq_len)
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, seq_len)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), lstm_out).squeeze(1)  # (batch_size, 2*hidden_size)
        return context_vector


# 定义TextRCNN类
class TextRCNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
        super(TextRCNN, self).__init__()
        self.hidden_size = hidden_size
        # 定义双向LSTM层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        # 定义注意力机制层
        self.attention = Attention(2 * hidden_size)
        # 定义卷积层
        self.conv = nn.Conv1d(in_channels=2 * hidden_size + input_size, out_channels=100, kernel_size=2, padding=1)
        # 定义全连接层
        self.fc = nn.Linear(100, output_size)

    def forward(self, x):
        # 直接传递 x，不需要 unsqueeze
        self.lstm.flatten_parameters()  # 确保LSTM参数在内存中是连续的
        lstm_out, _ = self.lstm(x)  # lstm_out: (batch_size, seq_len, 2*hidden_size)

        # 应用注意力机制
        context_vector = self.attention(lstm_out)  # (batch_size, 2*hidden_size)

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


# 定义集成模型
class DNNTextRCNNAttation(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
        super(DNNTextRCNNAttation, self).__init__()
        self.dnn = DNN(input_size, output_size)
        self.text_rcnn = TextRCNN(input_size + output_size, output_size, hidden_size)

    def forward(self, x):
        # DNN的输出
        dnn_output = self.dnn(x)
        # 将DNN的输出与原始输入拼接
        combined_input = torch.cat((x, dnn_output), dim=1)
        # TextRCNN的输入
        text_rcnn_output = self.text_rcnn(combined_input.unsqueeze(1))
        return text_rcnn_output



