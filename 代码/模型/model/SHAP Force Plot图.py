import os
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, f1_score, confusion_matrix, auc, \
    matthews_corrcoef, precision_score, recall_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
import shap

from decision_tree_model import DecisionTreeModel
from xgboost_model import XGBoostModel
from random_forest_model import RandomForestModel


def load_data(file_path, has_label=True):
    """加载数据并返回TensorDataset"""
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
    """保存结果到CSV文件"""
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
    """计算单个样本的SHAP值"""
    X_train = X_train.numpy() if isinstance(X_train, torch.Tensor) else X_train
    X_test = X_test.numpy() if isinstance(X_test, torch.Tensor) else X_test
    X_test = X_test.reshape(1, -1) if len(X_test.shape) == 1 else X_test

    def model_predict(X):
        proba = model.model.predict_proba(X)
        return proba[:, 1] if proba.shape[1] == 2 else proba

    try:
        explainer = shap.KernelExplainer(model_predict, shap.sample(X_train, min(batch_size, len(X_train))))
        shap_values = explainer.shap_values(X_test)

        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) == 2 else np.stack(shap_values, axis=-1)

        if len(shap_values.shape) == 1:
            shap_values = shap_values.reshape(1, -1)

        return {
            'shap_values': shap_values[0],
            'base_value': explainer.expected_value[1] if isinstance(explainer.expected_value,
                                                                    list) else explainer.expected_value
        }
    except Exception as e:
        print(f"\nSHAP分析失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def generate_ensemble_force_plot(models_shap_info, X_test, sample_idx=0):
    """生成集成模型的Force Plot"""
    valid_shap_info = [info for info in models_shap_info if info is not None]
    if not valid_shap_info:
        print("没有有效的SHAP值可以聚合！")
        return None

    all_shap = [info['shap_values'] for info in valid_shap_info]
    all_base = [info['base_value'] for info in valid_shap_info]

    aggregated_shap = np.mean(all_shap, axis=0)
    base_value = np.mean(all_base)

    abs_shap = np.abs(aggregated_shap)
    top_idx = np.argsort(abs_shap)[-10:][::-1]

    print("\n生成集成模型SHAP Force Plot...")
    force_plot = shap.plots.force(
        base_value=base_value,
        shap_values=aggregated_shap[top_idx],
        features=X_test[sample_idx][top_idx],
        feature_names=[f"Feature_{i}" for i in top_idx],
        matplotlib=False
    )

    html_path = "ensemble_force_plot.html"
    shap.save_html(html_path, force_plot)
    print(f"集成模型Force Plot已保存至: {os.path.abspath(html_path)}")

    importance_df = pd.DataFrame({
        'Feature': [f'Feature_{i}' for i in top_idx],
        'SHAP Value': aggregated_shap[top_idx],
        'Absolute Impact': abs_shap[top_idx]
    }).sort_values('Absolute Impact', ascending=False)

    print("\n集成模型特征重要性分析:")
    print(importance_df.to_string(index=False))

    return aggregated_shap


def main(test_data_path, results_output_path, n_splits=10, sample_idx=0):
    """主函数"""
    test_drug_ids, test_disease_ids, test_dataset = load_data(test_data_path, has_label=False)
    test_feature_dim = test_dataset.tensors[0].shape[1]
    X_test = test_dataset.tensors[0].numpy()

    results = []
    model_predictions = []
    models_shap_info = []

    models_and_paths = get_models_and_paths()

    for model_name, model_info in models_and_paths.items():
        print(f"\n{'=' * 50}\nProcessing model: {model_name}")

        train_path = model_info['train_path']
        _, _, train_dataset = load_data(train_path)

        train_feature_dim = train_dataset.tensors[0].shape[1]
        X_train = train_dataset.tensors[0]
        X_test_tensor = test_dataset.tensors[0]
        X_train, X_test_tensor = align_features(X_train, X_test_tensor, train_feature_dim, test_feature_dim)

        model_class = model_info['model_class']
        model = initialize_model(model_class, input_size=train_feature_dim, output_size=1)

        y_pred_prob = model.train_and_predict(X_train, train_dataset.tensors[1], X_test_tensor)[1]

        shap_info = explain_with_shap(model, X_train, X_test_tensor, sample_idx)
        models_shap_info.append(shap_info)

        model_predictions.append({
            'model_name': model_name,
            'predictions': y_pred_prob,
            'shap_values': shap_info['shap_values'] if shap_info else None
        })

    if any(info is not None for info in models_shap_info):
        generate_ensemble_force_plot([info for info in models_shap_info if info is not None], X_test, sample_idx)

    for model_info in model_predictions:
        if model_info['shap_values'] is None:
            continue

        y_pred_prob = model_info['predictions']
        y_pred = (y_pred_prob >= 0.5).astype(int)

        try:
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
        except:
            print(f"无法评估模型 {model_info['model_name']}，测试集可能无标签")

    save_results(results, results_output_path)


if __name__ == "__main__":
    test_data_path = r'D:\a毕设\数据\未归一化\验证\验证(ksu_汉明)_top3_disease_567.csv'
    results_output_path = r'D:\a毕设\数据\未归一化\验证结果'
    output_dir = r'D:\a毕设\数据\未归一化\验证结果'
    os.makedirs(output_dir, exist_ok=True)  # 自动创建目录（如果不存在）
    main(test_data_path, results_output_path)
