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
from sklearn.inspection import PartialDependenceDisplay

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
    if y is not None:
        return drug_ids, disease_ids, TensorDataset(X, y)
    else:
        return drug_ids, disease_ids, TensorDataset(X)

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
            'train_path': r'D:\a毕设\数据\未归一化\特征选择数据\90\f_classif_切比雪夫90.csv'
        },
        'DecisionTree': {
            'model_class': DecisionTreeModel,
            'train_path': r'D:\a毕设\数据\未归一化\特征选择数据\140\f_classif_汉明140.csv'
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


def explain_with_pdp(model, X_train, feature_names, sample_idx=0):
    """使用PDP解释模型输出"""
    try:
        print("\n生成PDP图...")
        # 如果X_train是Tensor，转换为NumPy数组
        if isinstance(X_train, torch.Tensor):
            X_train = X_train.numpy()

        # 获取模型的特征重要性
        if hasattr(model, 'model') and hasattr(model.model, 'feature_importances_'):
            feature_importances = model.model.feature_importances_
        else:
            print("模型没有特征重要性属性，使用默认前5个特征")
            feature_importances = np.zeros(X_train.shape[1])

        # 获取特征重要性排名前6的特征
        important_features = np.argsort(feature_importances)[::-1][:6]
        print(f"最重要的6个特征: {', '.join([feature_names[i] for i in important_features])}")

        # 创建画布时增加标题空间 (关键修改)
        fig = plt.figure(figsize=(12, 10))
        fig.suptitle("Partial Dependence Plot", y=1, fontsize=14)  # y参数控制标题位置

        # 使用GridSpec自定义子图布局
        gs = plt.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3)  # 增加子图间距

        # 绘制每个特征的PDP
        for i, feat_idx in enumerate(important_features[:6]):
            ax = fig.add_subplot(gs[i // 3, i % 3])  # 2行3列布局
            PartialDependenceDisplay.from_estimator(
                model.model,
                X_train,
                features=[feat_idx],
                feature_names=feature_names,
                ax=ax,
                line_kw={"color": "red", "linewidth": 2},
                grid_resolution=50
            )
            ax.set_title(f"Feature: {feature_names[feat_idx]}", pad=10)  # 子图标题增加padding

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"\nPDP分析失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def aggregate_predictions(model_predictions):
    """集成模型的预测概率"""
    prob_sum = np.zeros_like(model_predictions[0]['predictions'])
    for model_info in model_predictions:
        prob_sum += model_info['predictions']
    return prob_sum / len(model_predictions)

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

        # 将每个模型的预测结果存储
        model_predictions.append({
            'model_name': model_name,
            'predictions': y_pred_prob,
        })

    # 计算集成模型的预测结果
    if len(model_predictions) == len(models_and_paths):
        # 聚合每个模型的预测结果
        aggregated_probs = aggregate_predictions(model_predictions)

        # 生成集成模型的PDP图
        feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]
        explain_with_pdp(model, X_train, feature_names, sample_idx=sample_idx)

        # 如果测试集有标签，评估集成模型
        if len(test_dataset.tensors) > 1 and test_dataset.tensors[1] is not None:
            y_true = test_dataset.tensors[1].numpy()
            y_pred = (aggregated_probs >= 0.5).astype(int)

            # 评估集成模型
            acc, auc_score, aupr, f1, mcc, sn, sp, precision_val, recall_val = evaluate_model(
                y_true, aggregated_probs, y_pred)

            results.append({
                'Model': 'Ensemble',
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
    test_data_path = r'D:\a毕设\数据\未归一化\验证\验证_top3_disease_567.csv'
    # 结果保存路径
    results_output_path = r'D:\a毕设\数据\未归一化\验证结果'
    # 调用主函数
    main(test_data_path, results_output_path)
