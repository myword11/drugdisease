import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, f1_score, confusion_matrix, auc, matthews_corrcoef, precision_score, recall_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
from decision_tree_model import DecisionTreeModel
from svm_model import LinearSVCModel
from DNN import DNN
from xgboost_model import XGBoostModel
from random_forest_model import RandomForestModel
from logistic_model import LogisticRegressionModel

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

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=20, early_stopping_patience=5):
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
          'DNN': DNN,
          'DecisionTree': DecisionTreeModel,
        # 'SVM': LinearSVCModel,
            'XGBoost': XGBoostModel,
          # 'RandomForest': RandomForestModel,
        # 'LogisticRegression': LogisticRegressionModel
    }

def initialize_model(model_class, input_size, output_size):
    """
    初始化模型
    """
    if model_class == DNN:
        model = model_class(input_size, output_size)
        model.apply(init_weights)
    else:
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

def main(data_path, results_output_path, models_to_use, epochs=20, batch_size=64, learning_rate=0.001, n_splits=10, early_stopping_patience=5):
    """
    主函数
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载数据
    dataset = load_data(data_path)
    X = dataset.tensors[0]
    y = dataset.tensors[1]
    # 获取输入维度
    input_size = X.shape[1]
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
        # 保存每个模型的预测结果
        all_y_true = []
        all_y_pred = []
        all_y_pred_prob = []
        # 针对每个模型进行训练和测试
        for model_name, model_class in models_to_use.items():
            print(f"Training and testing with model: {model_name}")
            # 初始化模型
            model = initialize_model(model_class, input_size=input_size, output_size=1)
            if isinstance(model, nn.Module):
                model = model.to(device)
                criterion = nn.BCEWithLogitsLoss()
                optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)  # 添加 L2 正则化
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1, verbose=True)  # 动态调整学习率
                # 训练模型
                train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, device, epochs, early_stopping_patience)
                # 测试模型
                y_true, y_pred_prob = test_model(model, test_loader, device)
                y_pred = (np.array(y_pred_prob) >= 0.5).astype(int)  # 使用阈值0.5来判断分类结果
            else:
                # 对于 scikit-learn 模型
                X_train_np = X_train.numpy()
                y_train_np = y_train.numpy()
                X_test_np = X_test.numpy()
                y_test_np = y_test.numpy()
                y_pred, y_pred_prob = model.train_and_predict(X_train_np, y_train_np, X_test_np)
                y_true = y_test_np
            all_y_true.append(y_true)
            all_y_pred.append(y_pred)
            all_y_pred_prob.append(y_pred_prob)
        # 融合多个模型的预测结果
        y_true = all_y_true[0]  # 所有模型的 y_true 应该是一样的
        y_pred_ensemble = np.array(all_y_pred).mean(axis=0) >= 0.5  # 硬投票法
        # 评估模型
        acc, auc_score, aupr, f1, mcc, sn, sp, precision_val, recall_val = evaluate_model(y_true, all_y_pred_prob[0], y_pred_ensemble)
        # 保存结果
        results.append({
            'Model': 'Ensemble',
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
            f"Fold {fold}, Model: Ensemble, "
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
        #  r'D:\a毕设\数据\KSU下采样\合并\KSU_汉明.csv',
        # r'D:\a毕设\数据\KSU下采样\合并\KSU_曼哈顿.csv',
        # r'D:\a毕设\数据\KSU下采样\合并\KSU_闵可夫斯基.csv',
        # r'D:\a毕设\数据\KSU下采样\合并\KSU_欧几里得.csv',
        # r'D:\a毕设\数据\KSU下采样\合并\KSU_切比雪夫.csv',
        #  r'D:\a毕设\数据\未归一化\合并\KSU_欧几里得.csv',
        # r'D:\a毕设\数据\未归一化\合并\KSU_汉明.csv',
        # r'D:\a毕设\数据\未归一化\合并\KSU_曼哈顿.csv',
        # r'D:\a毕设\数据\未归一化\合并\KSU_闵可夫斯基.csv',
        # r'D:\a毕设\数据\未归一化\合并\KSU_切比雪夫.csv'
        # r'D:\a毕设\数据\未归一化\特征选择数据\80\Correlation_切比雪夫80.csv',
        #   r'D:\a毕设\数据\未归一化\特征选择数据\90\Correlation_汉明90.csv',
        #  r'D:\a毕设\数据\未归一化\特征选择数据\100\Correlation_汉明100.csv',
        #  r'D:\a毕设\数据\未归一化\特征选择数据\110\Correlation_汉明110.csv',
        #  r'D:\a毕设\数据\未归一化\特征选择数据\120\Correlation_汉明120.csv',
        #  r'D:\a毕设\数据\未归一化\特征选择数据\130\Correlation_汉明130.csv',
        #  r'D:\a毕设\数据\未归一化\特征选择数据\140\Correlation_汉明140.csv',
        #   r'D:\a毕设\数据\未归一化\特征选择数据\150\Correlation_汉明150.csv',
        #  r'D:\a毕设\数据\未归一化\特征选择数据\160\Correlation_汉明160.csv',
        # r'D:\a毕设\数据\未归一化\特征选择数据\170\Correlation_汉明170.csv',
        # r'D:\a毕设\数据\未归一化\特征选择数据\180\Correlation_汉明180.csv',
        # r'D:\a毕设\数据\未归一化\特征选择数据\200\Correlation_汉明200.csv',
        # r'D:\a毕设\数据\未归一化\特征选择数据\80\f_classif_汉明80.csv',
        #  r'D:\a毕设\数据\未归一化\特征选择数据\90\f_classif_汉明90.csv',
        # r'D:\a毕设\数据\未归一化\特征选择数据\100\f_classif_汉明100.csv',
        #  r'D:\a毕设\数据\未归一化\特征选择数据\110\f_classif_汉明110.csv',
        #  r'D:\a毕设\数据\未归一化\特征选择数据\120\f_classif_汉明120.csv',
        #  r'D:\a毕设\数据\未归一化\特征选择数据\130\f_classif_汉明130.csv',
        #  r'D:\a毕设\数据\未归一化\特征选择数据\140\f_classif_汉明140.csv',
        # r'D:\a毕设\数据\未归一化\特征选择数据\150\f_classif_汉明150.csv',
        # r'D:\a毕设\数据\未归一化\特征选择数据\160\f_classif_汉明160.csv',
        # r'D:\a毕设\数据\未归一化\特征选择数据\170\f_classif_汉明170.csv',
        # r'D:\a毕设\数据\未归一化\特征选择数据\180\f_classif_汉明180.csv',
        # r'D:\a毕设\数据\未归一化\特征选择数据\200\f_classif_汉明200.csv',
        # r'D:\a毕设\数据\未归一化\特征选择数据\80\SHAP_汉明80.csv',
        #  r'D:\a毕设\数据\未归一化\特征选择数据\90\SHAP_汉明90.csv',
        # r'D:\a毕设\数据\未归一化\特征选择数据\100\SHAP_汉明100.csv',
        #  r'D:\a毕设\数据\未归一化\特征选择数据\110\SHAP_汉明110.csv',
        #  r'D:\a毕设\数据\未归一化\特征选择数据\120\SHAP_汉明120.csv',
        #  r'D:\a毕设\数据\未归一化\特征选择数据\130\SHAP_汉明130.csv',
        #  r'D:\a毕设\数据\未归一化\特征选择数据\140\SHAP_汉明140.csv',
        # r'D:\a毕设\数据\未归一化\特征选择数据\150\SHAP_汉明150.csv',
        # r'D:\a毕设\数据\未归一化\特征选择数据\160\SHAP_汉明160.csv',
        # r'D:\a毕设\数据\未归一化\特征选择数据\170\SHAP_汉明170.csv',
        # r'D:\a毕设\数据\未归一化\特征选择数据\180\SHAP_汉明180.csv',
        # r'D:\a毕设\数据\未归一化\特征选择数据\200\SHAP_汉明200.csv',
        # r'D:\a毕设\数据\未归一化\特征选择数据\80\RFE_汉明80.csv',
        #  r'D:\a毕设\数据\未归一化\特征选择数据\90\RFE_汉明90.csv',
        # r'D:\a毕设\数据\未归一化\特征选择数据\100\RFE_汉明100.csv',
        #  r'D:\a毕设\数据\未归一化\特征选择数据\110\RFE_汉明110.csv',
        #  r'D:\a毕设\数据\未归一化\特征选择数据\120\RFE_汉明120.csv',
        #  r'D:\a毕设\数据\未归一化\特征选择数据\130\RFE_汉明130.csv',
        #  r'D:\a毕设\数据\未归一化\特征选择数据\140\RFE_汉明140.csv',
        # r'D:\a毕设\数据\未归一化\特征选择数据\150\RFE_汉明150.csv',
        # r'D:\a毕设\数据\未归一化\特征选择数据\160\RFE_汉明160.csv',
        # r'D:\a毕设\数据\未归一化\特征选择数据\170\RFE_汉明170.csv',
        # r'D:\a毕设\数据\未归一化\特征选择数据\180\RFE_汉明180.csv',
        # r'D:\a毕设\数据\未归一化\特征选择数据\200\RFE_汉明200.csv',
        # r'D:\a毕设\数据\未归一化\特征选择数据\80\Random_汉明80.csv',
        #  r'D:\a毕设\数据\未归一化\特征选择数据\90\Random_汉明90.csv',
        #  r'D:\a毕设\数据\未归一化\特征选择数据\100\Random_汉明100.csv',
        #  r'D:\a毕设\数据\未归一化\特征选择数据\110\Random_汉明110.csv',
        #  r'D:\a毕设\数据\未归一化\特征选择数据\120\Random_汉明120.csv',
        #  r'D:\a毕设\数据\未归一化\特征选择数据\130\Random_汉明130.csv',
        #  r'D:\a毕设\数据\未归一化\特征选择数据\140\Random_汉明140.csv',
        # r'D:\a毕设\数据\未归一化\特征选择数据\150\Random_汉明150.csv',
        # r'D:\a毕设\数据\未归一化\特征选择数据\160\Random_汉明160.csv',
        # r'D:\a毕设\数据\未归一化\特征选择数据\170\Random_汉明170.csv',
        # r'D:\a毕设\数据\未归一化\特征选择数据\180\Random_汉明180.csv',
        r"D:/a毕设/数据/未归一化/合并/未下采样.csv",


        

        # 你可以在此添加更多数据集路径
    ]
    results_folder = r'D:\a毕设\数据\未归一化\未特征选择\XGBoost+DT+RF（未下采样）'
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