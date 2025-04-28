import os
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, f1_score, confusion_matrix, auc, \
    matthews_corrcoef, precision_score, recall_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
from decision_tree_model import DecisionTreeModel
from xgboost_model import XGBoostModel
from random_forest_model import RandomForestModel
import shap


def load_data(file_path, has_label=True):
    """加载数据并返回 TensorDataset"""
    data = pd.read_csv(file_path)
    drug_ids = data.iloc[:, 0].values
    disease_ids = data.iloc[:, 1].values

    if has_label:
        X = data.iloc[:, 3:].values.astype('float32')
        y = data['label'].values.astype('float32')
        y = torch.tensor(y)
    else:
        X = data.iloc[:, 2:].values.astype('float32')
        y = None

    X = torch.tensor(X)
    print(f"Data shape: X: {X.shape}, y: {y.shape if y is not None else 'No label'}")
    return drug_ids, disease_ids, TensorDataset(X, y) if y is not None else TensorDataset(X)


def evaluate_model(y_true, y_pred_prob, y_pred, threshold=0.5):
    """评估模型性能"""
    y_pred_prob = np.array(y_pred_prob)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    auc_score = roc_auc_score(y_true, y_pred_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    aupr = auc(recall, precision)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sn = tp / (tp + fn)
    sp = tn / (tn + fp)
    precision_val = precision_score(y_true, y_pred)
    recall_val = recall_score(y_true, y_pred)

    return acc, auc_score, aupr, f1, mcc, sn, sp, precision_val, recall_val


def save_results(results, output_path):
    """保存结果到 CSV 文件"""
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


def get_models_and_paths():
    """返回模型及其对应训练数据路径的字典"""
    return {
        'RF': {
            'model_class': RandomForestModel,
            'train_path': r'D:\a毕设\数据\未归一化\特征选择数据\100\f_classif_切比雪夫100.csv'
        },
        'DecisionTree': {
            'model_class': DecisionTreeModel,
            'train_path': r'D:\a毕设\数据\未归一化\特征选择数据\170\f_classif_汉明170.csv'
        },
        'XGBoost': {
            'model_class': XGBoostModel,
            'train_path': r'D:\a毕设\数据\未归一化\特征选择数据\160\f_classif_汉明160.csv'
        }
    }


def initialize_model(model_class, input_size, output_size):
    """初始化模型"""
    model = model_class(random_state=42)
    return model


def align_features(X_train, X_test, train_feature_dim, test_feature_dim):
    """对齐训练集和测试集的特征维度"""
    if train_feature_dim == test_feature_dim:
        return X_train, X_test

    print(f"Aligning features from {train_feature_dim}D to {test_feature_dim}D")

    if train_feature_dim < test_feature_dim:
        padding = torch.zeros((X_train.shape[0], test_feature_dim - train_feature_dim))
        X_train = torch.cat([X_train, padding], dim=1)
    else:
        X_train = X_train[:, :test_feature_dim]

    return X_train, X_test


def explain_with_shap(model, X_train, X_test, sample_idx=0, batch_size=100):
    """改进的SHAP解释函数，完全兼容SHAP v0.20+"""
    # 数据格式转换
    X_train = X_train.numpy() if isinstance(X_train, torch.Tensor) else X_train
    X_test = X_test.numpy() if isinstance(X_test, torch.Tensor) else X_test

    # 确保测试数据是2D数组
    X_test = X_test.reshape(1, -1) if len(X_test.shape) == 1 else X_test

    # 模型预测函数
    def model_predict(X):
        proba = model.model.predict_proba(X)
        return proba[:, 1] if proba.shape[1] == 2 else proba  # 二分类取正类概率

    try:
        # 创建解释器
        explainer = shap.KernelExplainer(model_predict, shap.sample(X_train, min(batch_size, len(X_train))))

        # 计算SHAP值
        shap_values = explainer.shap_values(X_test)

        # 处理多输出情况
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) == 2 else np.stack(shap_values, axis=-1)

        # 获取基础值
        base_value = explainer.expected_value
        if isinstance(base_value, list):
            base_value = base_value[1] if len(base_value) == 2 else base_value[0]

        # 确保维度正确
        if len(shap_values.shape) == 1:
            shap_values = shap_values.reshape(1, -1)

        # 特征选择逻辑
        abs_shap = np.abs(shap_values).mean(axis=0)
        top_idx = np.argsort(abs_shap)[-5:][::-1]  # 取最重要的5个特征

        # 生成force plot (关键修正部分)
        print("\n生成SHAP Force Plot...")
        force_plot = shap.plots.force(
            base_value=base_value,
            shap_values=shap_values[sample_idx][top_idx],
            features=X_test[sample_idx][top_idx],
            feature_names=[f"Feature_{i}" for i in top_idx],
            matplotlib=False
        )

        # 保存HTML
        html_path = "shap_force_plot.html"
        shap.save_html(html_path, force_plot)
        print(f"Force plot已保存至: {os.path.abspath(html_path)}")

        # 特征重要性表格
        importance_df = pd.DataFrame({
            'Feature': [f'Feature_{i}' for i in top_idx],
            'SHAP Value': shap_values[sample_idx][top_idx],
            'Absolute Impact': abs_shap[top_idx]
        }).sort_values('Absolute Impact', ascending=False)

        print("\n特征重要性分析:")
        print(importance_df.to_string(index=False))

        return shap_values

    except Exception as e:
        print(f"\nSHAP分析失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def aggregate_shap_values(model_predictions):
    """集成多个模型的SHAP值"""
    combined_shap_values = None
    valid_models_count = 0  # 记录有效模型的数量

    for model_info in model_predictions:
        shap_values = model_info['shap_values']

        if shap_values is not None:
            valid_models_count += 1
            if combined_shap_values is None:
                combined_shap_values = shap_values
            else:
                combined_shap_values += shap_values

    # 只除以有效模型的数量
    if valid_models_count > 0:
        combined_shap_values /= valid_models_count
    else:
        print("没有有效的SHAP值可以合并！")
        return None

    return combined_shap_values


def main(test_data_path, results_output_path, n_splits=10, sample_idx=0):
    """主函数"""
    # 加载测试集
    test_drug_ids, test_disease_ids, test_dataset = load_data(test_data_path, has_label=False)
    test_feature_dim = test_dataset.tensors[0].shape[1]

    results = []
    model_predictions = []
    models_and_paths = get_models_and_paths()

    for model_name, model_info in models_and_paths.items():
        print(f"\n{'=' * 50}\nProcessing model: {model_name}")

        # 加载训练数据
        train_path = model_info['train_path']
        _, _, train_dataset = load_data(train_path)

        # 对齐特征维度
        train_feature_dim = train_dataset.tensors[0].shape[1]
        X_train = train_dataset.tensors[0]
        X_test = test_dataset.tensors[0]
        X_train, X_test = align_features(X_train, X_test, train_feature_dim, test_feature_dim)

        # 初始化模型
        model_class = model_info['model_class']
        model = initialize_model(model_class, input_size=train_feature_dim, output_size=1)

        # 训练并预测
        y_pred_prob = model.train_and_predict(X_train, train_dataset.tensors[1], X_test)[1]

        # SHAP解释
        shap_values = explain_with_shap(model, X_train, X_test, sample_idx=sample_idx)

        model_predictions.append({
            'model_name': model_name,
            'predictions': y_pred_prob,
            'shap_values': shap_values
        })

    # 集成多个模型的SHAP值
    aggregated_shap_values = aggregate_shap_values(model_predictions)

    # 绘制集成后的SHAP值摘要图
    shap.summary_plot(aggregated_shap_values, X_test.numpy(), max_display=10)

    # 评估模型性能
    for model_info in model_predictions:
        y_pred_prob = model_info['predictions']
        y_pred = (y_pred_prob >= 0.5).astype(int)

        acc, auc_score, aupr, f1, mcc, sn, sp, precision_val, recall_val = evaluate_model(
            test_dataset.tensors[1].numpy(), y_pred_prob, y_pred)

        results.append({
            'Model': model_info['model_name'],
            'Accuracy': acc,
            'AUC': auc_score,
            'AUPR': aupr,
            'F1': f1,
            'MCC': mcc,
            'Sensitivity': sn,
            'Specificity': sp,
            'Precision': precision_val,
            'Recall': recall_val
        })

    # 保存结果
    save_results(results, results_output_path)


if __name__ == "__main__":
    # 测试数据路径
    test_data_path = r'D:\a毕设\数据\未归一化\验证\验证(ksu_汉明)_top3_disease_567.csv'
    # 结果保存路径
    results_output_path = r'D:\a毕设\数据\未归一化\验证结果'
    # 调用主函数
    main(test_data_path, results_output_path)
