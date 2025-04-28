import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, f1_score, confusion_matrix, auc, \
    matthews_corrcoef, precision_score, recall_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
from decision_tree_model import DecisionTreeModel
from svm_model import LinearSVCModel
from DNN import DNN
from xgboost_model import XGBoostModel
from random_forest_model import RandomForestModel
from logistic_model import LogisticRegressionModel
from TextRCNN import TextRCNN

def load_data(file_path):
    """
    加载数据并返回 TensorDataset
    """
    data = pd.read_csv(file_path)
    # 动态获取特征列，从第4列到倒数第1列（假设最后一列是标签）
    X = data.iloc[:, 3:].values.astype('float32')
    y = data['label'].values.astype('float32')
    X = torch.tensor(X)
    y = torch.tensor(y)
    print(f"Data shape: X: {X.shape}, y: {y.shape}")
    print(f"Sample data: X: {X[:5]}, y: {y[:5]}")
    return TensorDataset(X, y)


def evaluate_model(y_true, y_pred_prob, y_pred, threshold=0.5):
    """
    评估模型性能
    """
    y_pred_prob = np.array(y_pred_prob)  # 将 y_pred_prob 转换为 NumPy 数组
    y_pred = np.array(y_pred)  # 将 y_pred 转换为 NumPy 数组
    acc = accuracy_score(y_true, y_pred)  # 准确度
    auc_score = roc_auc_score(y_true, y_pred_prob)  # AUC
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    aupr = auc(recall, precision)  # AUPR
    f1 = f1_score(y_true, y_pred)  # F1 score
    mcc = matthews_corrcoef(y_true, y_pred)  # MCC (Matthews相关系数)
    # 灵敏度 (Sensitivity, Sn) 和 特异性 (Specificity, Sp)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sn = tp / (tp + fn)  # 灵敏度
    sp = tn / (tn + fp)  # 特异性
    # 精确度 (Precision) 和 召回率 (Recall)
    precision_val = precision_score(y_true, y_pred)  # 精确度
    recall_val = recall_score(y_true, y_pred)  # 召回率
    return acc, auc_score, aupr, f1, mcc, sn, sp, precision_val, recall_val


def save_results(results, output_path):
    """
    保存结果到 CSV 文件
    """
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=20,
                early_stopping_patience=5):
    """
    训练模型
    """
    model.train()
    best_val_auc = 0.0
    early_stopping_counter = 0
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
        # 验证集评估
        model.eval()
        y_true, y_pred_prob = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred_logits = model(X_batch).squeeze()  # 原始输出 (logits)
                y_pred = torch.sigmoid(y_pred_logits).cpu().numpy()  # 应用 Sigmoid 转为概率
                y_true.extend(y_batch.cpu().numpy())
                y_pred_prob.extend(y_pred)
            val_auc = roc_auc_score(y_true, y_pred_prob)
            print(f"Epoch {epoch + 1}/{epochs}, Validation AUC: {val_auc:.4f}")
        # 早停法
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        if early_stopping_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
        model.train()
        scheduler.step(epoch_loss)  # 动态调整学习率


def test_model(model, test_loader, device):
    """
    测试模型
    """
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


def get_models():
    """
    返回所有模型的字典
    """
    return {
           # 'DNN': DNN,
           'DecisionTree': DecisionTreeModel,
           # 'TextRCNN': TextRCNN,
        #'SVM': LinearSVCModel,
         'XGBoost': XGBoostModel,
         'RandomForest': RandomForestModel,
        #'LogisticRegression': LogisticRegressionModel
    }


def initialize_model(model_class, input_size, output_size):
    """
    初始化模型
    """
    if model_class == DNN:
        model = model_class(input_size, output_size)
        model.apply(init_weights)
    elif model_class == TextRCNN:  # 单独处理 TextRCNN
        model = model_class(input_size, output_size)
    else:  # 其他模型（如 DecisionTree、XGBoost 等）
        model = model_class(random_state=42)
    return model


def init_weights(m):
    """
    初始化模型权重
    """
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_uniform_(param, nonlinearity='relu')
            elif 'bias' in name:
                nn.init.zeros_(param)


from scipy.stats import mode

def save_hard_voting_results(results, output_path):
    """
    保存硬投票结果到单独的 CSV 文件
    """
    # 提取硬投票结果
    hard_voting_results = [result for result in results if result['Model'] == 'HardVoting']
    # 转换为 DataFrame
    hard_voting_df = pd.DataFrame(hard_voting_results)
    # 保存到 CSV 文件
    hard_voting_df.to_csv(output_path, index=False)
    print(f"Hard voting results saved to {output_path}")

def main(data_paths, results_output_path, models_to_use, epochs=20, batch_size=64, learning_rate=0.001, n_splits=10,
         early_stopping_patience=5):
    """
    主函数
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载数据
    datasets = {model_name: load_data(data_path) for model_name, data_path in data_paths.items()}
    results = []
    fold = 1
    # 获取数据集的特征维度
    X = datasets[list(datasets.keys())[0]].tensors[0]  # 假设所有模型的数据集特征维度相同
    y = datasets[list(datasets.keys())[0]].tensors[1]
    for train_index, test_index in KFold(n_splits=n_splits, shuffle=True, random_state=42).split(X):
        print(f"Processing fold {fold}/{n_splits}")
        all_model_predictions = []  # 存储所有模型的预测结果
        y_true = None
        for model_name, dataset in datasets.items():
            print(f"Processing model: {model_name}")
            X = dataset.tensors[0]
            y = dataset.tensors[1]
            # 划分训练集和测试集
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            # 创建数据加载器
            train_dataset = TensorDataset(X_train, y_train)
            test_dataset = TensorDataset(X_test, y_test)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            # 初始化模型
            model_class = models_to_use[model_name]
            model = initialize_model(model_class, input_size=X.shape[1], output_size=1)
            # 训练和测试模型
            if isinstance(model, nn.Module):
                model = model.to(device)
                criterion = nn.BCEWithLogitsLoss()
                optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1, verbose=True)
                train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, device, epochs,
                            early_stopping_patience)
                y_true, y_pred_prob = test_model(model, test_loader, device)
            else:
                X_train_np = X_train.numpy()
                y_train_np = y_train.numpy()
                X_test_np = X_test.numpy()
                y_test_np = y_test.numpy()
                y_pred, y_pred_prob = model.train_and_predict(X_train_np, y_train_np, X_test_np)
                y_true = y_test_np
            # 保存当前模型的预测结果
            y_pred = (np.array(y_pred_prob) >= 0.5).astype(int)
            all_model_predictions.append(y_pred)
            # 评估当前模型
            acc, auc_score, aupr, f1, mcc, sn, sp, precision_val, recall_val = evaluate_model(y_true, y_pred_prob, y_pred)
            results.append({
                'Model': model_name,
                'Fold': fold,
                'Accuracy (ACC)': acc,
                'AUC': auc_score,
                'AUPR': aupr,
                'Precision': precision_val,
                'Recall': recall_val,
                'F1 Score': f1,
                'MCC': mcc,
                'Sensitivity (Sn)': sn,
                'Specificity (Sp)': sp
            })
        # 硬投票
        final_predictions, _ = mode(np.stack(all_model_predictions), axis=0)
        final_predictions = final_predictions.flatten()
        # 评估硬投票结果
        acc, auc_score, aupr, f1, mcc, sn, sp, precision_val, recall_val = evaluate_model(y_true, final_predictions, final_predictions)
        results.append({
            'Model': 'HardVoting',
            'Fold': fold,
            'Accuracy (ACC)': acc,
            'AUC': auc_score,
            'AUPR': aupr,
            'Precision': precision_val,
            'Recall': recall_val,
            'F1 Score': f1,
            'MCC': mcc,
            'Sensitivity (Sn)': sn,
            'Specificity (Sp)': sp
        })
        fold += 1
    # 保存所有结果到CSV
    save_results(results, results_output_path)
    # 保存硬投票结果到单独的 CSV 文件
    hard_voting_output_path = results_output_path.replace('.csv', '_hard_voting.csv')
    save_hard_voting_results(results, hard_voting_output_path)

if __name__ == '__main__':
    # 定义不同模型对应的数据路径
    data_paths = {
                'DNN': r'D:\a毕设\数据\未归一化\特征选择数据\110\f_classif_曼哈顿110.csv',
               'DecisionTree': r'D:\a毕设\数据\未归一化\特征选择数据\140\f_classif_汉明140.csv',
             'TextRCNN': r'D:\a毕设\数据\未归一化\特征选择数据\90\RFE_切比雪夫90.csv',
             # 'XGBoost': r'D:\a毕设\数据\未归一化\特征选择数据\160\f_classif_汉明160.csv',
            #   'RandomForest': r'D:\a毕设\数据\未归一化\特征选择数据\90\f_classif_切比雪夫90.csv',


          # 'DNN': r'D:\a毕设\数据\未归一化\合并\KSU_曼哈顿.csv',
         #    'DecisionTree': r'D:\a毕设\数据\未归一化\合并\KSU_汉明.csv',
         #    'TextRCNN': r'D:\a毕设\数据\未归一化\合并\KSU_切比雪夫.csv',
           # 'XGBoost': r'D:\a毕设\数据\未归一化\合并\KSU_汉明.csv',
           #  'RandomForest': r'D:\a毕设\数据\未归一化\合并\KSU_切比雪夫.csv',
    }
    # 定义结果保存路径
    results_folder = r'D:\a毕设\数据\未归一化\集成模型(ksu)'
    # 确保目标文件夹存在
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    # 定义结果文件名
    results_output_path = os.path.join(results_folder, 'DT+DNN+TextRCNN.csv')
    # 获取所有模型
    models_to_use = get_models()
    # 运行主函数
    main(data_paths, results_output_path, models_to_use)
