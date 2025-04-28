import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, f1_score, confusion_matrix, auc
from torch.utils.data import DataLoader, TensorDataset
from LSTM import LSTM
from MLP import model_MLP
from LSTM_Attention import AttentionLSTM
from BiLSTM import BiLSTM
from DNN import DNN
from GRU import GRU
from RNN import RNN
from TextCNN import TextCNN
from TextRCNN import TextRCNN
from VDCNN import VDCNN


# 数据加载函数
def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.iloc[:, 4:604].values.astype('float32')
    y = data['label'].values.astype('float32')
    X = torch.tensor(X)
    y = torch.tensor(y)
    return TensorDataset(X, y)


# 模型评估函数
def evaluate_model(y_true, y_pred_prob, threshold=0.5):
    y_pred_prob = np.array(y_pred_prob)  # 转换为 NumPy 数组
    y_true = np.array(y_true)
    y_pred = (y_pred_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    aupr = auc(recall, precision)
    f1 = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sn = tp / (tp + fn)  # 灵敏度
    sp = tn / (tn + fp)  # 特异性
    return acc, auc_score, aupr, f1, sn, sp


# 保存结果函数
def save_results(results, output_path):
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


# 模型训练函数
def train_model(model, train_loader, criterion, optimizer, device, epochs=20):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred_logits = model(X_batch)  # 原始输出 (logits)
            y_pred_logits = y_pred_logits.squeeze()  # 去掉多余维度
            loss = criterion(y_pred_logits, y_batch.float())
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}")


# 模型测试函数
def test_model(model, test_loader, device):
    sigmoid = nn.Sigmoid()
    model.eval()
    y_true, y_pred_prob = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred_logits = model(X_batch).squeeze()  # 原始输出 (logits)
            y_pred = sigmoid(y_pred_logits).cpu().numpy()  # 应用 Sigmoid 转为概率
            y_true.extend(y_batch.cpu().numpy())
            y_pred_prob.extend(y_pred)

    return y_true, y_pred_prob


# 模型字典
def get_models():
    """返回所有模型的字典"""
    return {
        'VDCNN': VDCNN,
        'LSTM': LSTM,
        'BiLSTM': BiLSTM,
        'GRU': GRU,
        'DNN': DNN,
        'RNN': RNN,
        'TextCNN': TextCNN,
        'TextRCNN': TextRCNN,
        'MLP': model_MLP,
        'AttentionLSTM': AttentionLSTM
    }


# 主函数
def main(train_root, test_path, results_output_path, models_to_use, epochs=20, batch_size=64, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载测试数据
    test_dataset = load_data(test_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    results = []

    for root, _, files in os.walk(train_root):
        for file in files:
            if file.endswith('.csv'):
                input_path = os.path.join(root, file)
                print(f"Processing training file: {input_path}")

                # 获取文件名
                file_name = os.path.splitext(os.path.basename(input_path))[0]

                # 加载训练数据
                train_dataset = load_data(input_path)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

                # 针对每个模型进行训练和测试
                for model_name, model_class in models_to_use.items():
                    print(f"Training and testing with model: {model_name}")

                    # 初始化模型
                    model = model_class(input_size=600, output_size=1).to(device)
                    # sigmoid = nn.Sigmoid()
                    criterion = nn.BCEWithLogitsLoss()
                    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

                    # 训练模型
                    train_model(model, train_loader, criterion, optimizer, device, epochs)

                    # 测试模型
                    y_true, y_pred_prob = test_model(model, test_loader, device)

                    # 评估模型
                    acc, auc_score, aupr, f1, sn, sp = evaluate_model(y_true, y_pred_prob)

                    # 保存结果
                    results.append({
                        'Model': model_name,
                        'Training File': file_name,
                        'ACC': acc,
                        'AUC': auc_score,
                        'AUPR': aupr,
                        'F1': f1,
                        'Sensitivity (Sn)': sn,
                        'Specificity (Sp)': sp
                    })

                    print(
                        f"ACC: {acc:.4f}, "
                        f"AUC: {auc_score:.4f}, "
                        f"AUPR: {aupr:.4f}, "
                        f"F1: {f1:.4f}, "
                        f"Sn: {sn:.4f}, "
                        f"Sp: {sp:.4f}")

    # 保存所有结果到CSV
    save_results(results, results_output_path)


if __name__ == '__main__':
    train_root = '../data/features_data/mol2vec_6000_down_sampling/'
    test_path = '../data/features_data/mol2vec/induc/miner/fold0/test_S1.csv'
    results_output_path = '../results/6000_down_sampling_test_S1_results.csv'

    # 获取模型字典
    models_to_use = get_models()

    # 运行主函数，传入要使用的模型
    main(train_root, test_path, results_output_path, models_to_use)
