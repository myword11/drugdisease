import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, f1_score, confusion_matrix, auc, matthews_corrcoef, precision_score, recall_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from torch.utils.data import DataLoader, TensorDataset


# 数据加载函数
def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.iloc[:, 3:267].values.astype('float32')
    y = data['label'].values.astype('float32')
    X = torch.tensor(X)
    y = torch.tensor(y)
    print(f"Data shape: X: {X.shape}, y: {y.shape}")
    print(f"Sample data: X: {X[:5]}, y: {y[:5]}")
    return TensorDataset(X, y)

# 模型评估函数
def evaluate_model(y_true, y_pred_prob, threshold=0.5):
    y_pred_prob = np.array(y_pred_prob)  # 将 y_pred_prob 转换为 NumPy 数组
    y_pred = (y_pred_prob >= threshold).astype(int)  # 使用阈值0.5来判断分类结果
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

# 保存结果函数
def save_results(results, output_path):
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

# 模型字典
# 模型字典
def get_models():
    """返回所有模型的字典"""
    return {
        'XGBoost': XGBClassifier,
        'DecisionTree': DecisionTreeClassifier,
        'RandomForest': RandomForestClassifier,
        # 'LogisticRegression': LogisticRegression,
        # 'SVM': SVC,
        # 'DNN': DNN
    }


# 模型训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=20, early_stopping_patience=5):
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

# 模型测试函数
# 模型测试函数
def test_model(model, test_loader, device):
    y_true, y_pred_prob = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            if isinstance(model, torch.nn.Module):
                # 对于 PyTorch 模型
                model.eval()
                y_pred_logits = model(X_batch).squeeze()  # 原始输出 (logits)
                y_pred = torch.sigmoid(y_pred_logits).cpu().numpy()  # 应用 Sigmoid 转为概率
            else:
                # 对于其他模型
                y_pred = model.predict_proba(X_batch.cpu().numpy())[:, 1]  # 获取正类的概率
            y_true.extend(y_batch.cpu().numpy())
            y_pred_prob.extend(y_pred)
    return y_true, y_pred_prob


# 主函数
def main(data_path, results_output_path, models_to_use, epochs=20, batch_size=64, learning_rate=0.001, n_splits=10, early_stopping_patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载数据
    dataset = load_data(data_path)
    X = dataset.tensors[0]
    y = dataset.tensors[1]
    # 初始化 KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    results = []
    fold = 1
    for train_index, test_index in kf.split(X):
        print(f"Processing fold {fold}/{n_splits}")
        # 划分训练集和测试集
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # 创建数据加载器
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        # 保存训练集和测试集
        data_folder = os.path.join(os.path.dirname(results_output_path), f"fold_{fold}")
        os.makedirs(data_folder, exist_ok=True)
        train_path = os.path.join(data_folder, "train.csv")
        test_path = os.path.join(data_folder, "test.csv")
        pd.concat([pd.DataFrame(X_train.numpy()), pd.DataFrame(y_train.numpy(), columns=['label'])], axis=1).to_csv(train_path, index=False)
        pd.concat([pd.DataFrame(X_test.numpy()), pd.DataFrame(y_test.numpy(), columns=['label'])], axis=1).to_csv(test_path, index=False)
        # 针对每个模型进行训练和测试
        for model_name, model_class in models_to_use.items():
            print(f"Training and testing with model: {model_name}")
            if model_name == 'DNN':
                # 初始化 DNN 模型
                model = model_class(random_state=42, max_iter=200)  # 增加 max_iter 参数以防止训练不收敛
                model.fit(X_train.numpy(), y_train.numpy())
                y_true, y_pred_prob = test_model(model, test_loader, device)
            else:
                # 初始化其他模型
                if model_name == 'SVM':
                    model = model_class(random_state=42, probability=True)  # 对于 SVM，确保 probability=True
                else:
                    model = model_class(random_state=42)
                model.fit(X_train.numpy(), y_train.numpy())
                y_true, y_pred_prob = test_model(model, test_loader, device)
            # 评估模型
            acc, auc_score, aupr, f1, mcc, sn, sp, precision_val, recall_val = evaluate_model(y_true, y_pred_prob)
            # 保存结果
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
            print(
                f"Fold {fold}, Model: {model_name}, "
                f"ACC: {acc:.4f}, "
                f"AUC: {auc_score:.4f}, "
                f"AUPR: {aupr:.4f}, "
                f"F1: {f1:.4f}, "
                f"Sn: {sn:.4f}, "
                f"Sp: {sp:.4f}"
            )
        fold += 1
    # 保存所有结果到CSV
    save_results(results, results_output_path)

if __name__ == '__main__':
    data_paths = [

        r'D:\a毕设\数据\归一化下采样\归一化数据\分类数据\KSU.csv',
        # 你可以在此添加更多数据集路径
    ]
    results_folder = r'D:\a毕设\数据\KSU下采样\传统算法结果\集成\xgdtrf.csv'
    # 确保目标文件夹存在
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    for data_path in data_paths:
        # 使用固定路径保存结果
        file_name = os.path.basename(data_path).split(".")[0]
        dataset_folder = os.path.join(results_folder, file_name)
        os.makedirs(dataset_folder, exist_ok=True)
        results_output_path = os.path.join(dataset_folder, f'{file_name}_results.csv')
        print(f"Processing dataset: {data_path}")
        models_to_use = get_models()  # 获取所有模型
        main(data_path, results_output_path, models_to_use)