import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class TransformerDNN(nn.Module):
    def __init__(self, input_size, output_size, d_model=128, num_heads=8, num_layers=4):
        super(TransformerDNN, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers

        # 添加一个线性层来映射输入特征为128维
        self.fc_input = nn.Linear(input_size, d_model)

        # Transformer部分，启用 batch_first
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        # DNN部分
        self.fc1 = nn.Linear(d_model, 128)  # 第一层全连接层
        self.fc2 = nn.Linear(128, 64)  # 第二层全连接层
        self.fc3 = nn.Linear(64, output_size)  # 输出层

        self.relu = nn.ReLU()  # ReLU激活函数

        # 利用Sequential简化全连接层
        self.dnn = nn.Sequential(
            self.fc1,
            self.relu,
            self.fc2,
            self.relu,
            self.fc3
        )

    def forward(self, x):
        # 通过线性层映射输入数据到d_model维度
        x = self.fc_input(x)

        # 确保输入数据是三维的 (batch_size, sequence_length, input_size)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # 如果是二维张量，扩展为三维 (batch_size, 1, input_size)

        # Transformer部分
        x = self.transformer_encoder(x)

        # 取Transformer输出的最后一个时间步作为特征表示
        x = x[:, -1, :]  # 选择最后一个时间步的输出

        # 使用简化的DNN部分：全连接层
        x = self.dnn(x)

        return x


#