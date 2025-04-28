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
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, rnn_out):
        # rnn_out: (batch_size, seq_len, hidden_size)
        attention_scores = self.attention(rnn_out).squeeze(2)  # (batch_size, seq_len)
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, seq_len)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), rnn_out).squeeze(1)  # (batch_size, hidden_size)
        return context_vector


# 定义RNN + Attention模型
class RNNWithAttention(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
        super(RNNWithAttention, self).__init__()
        self.hidden_size = hidden_size
        # 定义RNN层
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=1, batch_first=True)
        # 定义注意力机制层
        self.attention = Attention(hidden_size)
        # 定义全连接层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # RNN层
        rnn_out, _ = self.rnn(x)  # rnn_out: (batch_size, seq_len, hidden_size)

        # 应用注意力机制
        context_vector = self.attention(rnn_out)  # (batch_size, hidden_size)

        # 通过全连接层输出结果
        output = self.fc(context_vector)  # (batch_size, output_size)
        return output


# 定义集成模型
class DNNRNNWithAttention(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=128):
        super(DNNRNNWithAttention, self).__init__()
        self.dnn = DNN(input_size, output_size)
        self.rnn_attention = RNNWithAttention(input_size + output_size, output_size, hidden_size)

    def forward(self, x):
        # DNN的输出
        dnn_output = self.dnn(x)
        # 将DNN的输出与原始输入拼接
        combined_input = torch.cat((x, dnn_output), dim=1)
        # RNN + Attention模型的输入
        output = self.rnn_attention(combined_input.unsqueeze(1))  # 添加维度（batch_size, seq_len, input_size）
        return output
