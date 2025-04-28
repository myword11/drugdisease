import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义一个简单的Self-Attention模块
class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        # 定义查询、键、值的线性变换
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        # 计算查询、键、值
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # 计算注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.embed_size ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 应用注意力权重到值上
        out = torch.matmul(attention_weights, V)
        return out


# 定义一个带注意力机制的DNN类
class DNNAttention(nn.Module):
    def __init__(self, input_size, output_size):
        super(DNNAttention, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # 第一层全连接层
        self.fc2 = nn.Linear(128, 64)          # 第二层全连接层
        self.attention = SelfAttention(64)     # 注意力机制层
        self.fc3 = nn.Linear(64, output_size)  # 输出层，输出大小为output_size

        self.relu = nn.ReLU()  # 使用ReLU作为激活函数

    def forward(self, x):
        x = self.relu(self.fc1(x))  # 第一层全连接层 + ReLU激活函数
        x = self.relu(self.fc2(x))  # 第二层全连接层 + ReLU激活函数
        x = self.attention(x)       # 应用注意力机制
        x = self.fc3(x)             # 输出层
        return x


