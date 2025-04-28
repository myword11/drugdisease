import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, f1_score, confusion_matrix, auc, \
    matthews_corrcoef, precision_score, recall_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
from decision_tree_model import DecisionTreeModel
from xgboost_model import XGBoostModel
from random_forest_model import RandomForestModel
from scipy.stats import mode
import lime
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt


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


def explain_with_lime(model, X_train, X_test, feature_names=None, num_features=20, sample_idx=0):
    """使用LIME解释模型预测（改进版：带强度分级的彩色可视化）"""
    # 转换数据格式
    if isinstance(X_train, torch.Tensor):
        X_train = X_train.numpy()
    if isinstance(X_test, torch.Tensor):
        X_test = X_test.numpy()

    # 处理特征名称
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
    elif isinstance(feature_names, torch.Tensor):
        feature_names = feature_names.numpy().tolist()
    elif not isinstance(feature_names, list):
        feature_names = list(feature_names)

    print("使用的特征名称:", feature_names)
    assert len(feature_names) == X_train.shape[1], \
        f"特征名称数量({len(feature_names)})与特征维度({X_train.shape[1]})不匹配"

    # 创建LIME解释器
    explainer = LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        class_names=['No Relation', 'Has Relation'],
        mode='classification',
        verbose=True
    )

    # 解释样本
    explanation = explainer.explain_instance(
        data_row=X_test[sample_idx],
        predict_fn=model.predict_proba,
        num_features=num_features
    )

    # 保存解释结果
    explanation.save_to_file(f'lime_explanation_sample_{sample_idx}.html')

    # 准备可视化数据
    lime_results = explanation.as_list()
    features = [x[0] for x in lime_results]
    weights = [x[1] for x in lime_results]

    # 计算颜色强度（归一化到[0.3, 1]区间）
    max_weight = max(abs(min(weights)), abs(max(weights)))
    normalized_weights = [abs(w) / max_weight for w in weights]

    # 设置颜色（正相关：绿色渐变；负相关：红色渐变）
    colors = []
    for w, nw in zip(weights, normalized_weights):
        intensity = 0.3 + 0.7 * nw  # 基础强度0.3，最大1.0
        if w > 0:
            colors.append((0, intensity, 0))  # 绿色
        else:
            colors.append((intensity, 0, 0))  # 红色

    # 创建可视化
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(weights)), weights, color=colors, align='center')

    # 添加颜色条（显示强度）
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn,
                               norm=plt.Normalize(vmin=-max_weight, vmax=max_weight))
    sm.set_array([])
    cbar = plt.colorbar(sm, pad=0.02)
    cbar.set_label('Feature Influence Strength', rotation=270, labelpad=20)

    # 设置标签和标题
    plt.yticks(range(len(features)), features)
    plt.xlabel('Influence on Prediction')
    plt.title('LIME Explanation: Feature Importance (Color indicates strength and direction)')

    # 添加自定义图例
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color='darkgreen', label='Strong Positive'),
        plt.Rectangle((0, 0), 1, 1, color='lightgreen', label='Weak Positive'),
        plt.Rectangle((0, 0), 1, 1, color='white', label='Direction & Strength'),
        plt.Rectangle((0, 0), 1, 1, color='lightcoral', label='Weak Negative'),
        plt.Rectangle((0, 0), 1, 1, color='darkred', label='Strong Negative')
    ]
    plt.legend(handles=legend_elements, loc='lower right', framealpha=1)

    plt.tight_layout()
    plt.savefig('lime_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 打印调试信息（按强度排序）
    print("\nLIME解释结果（按影响强度排序）:")
    # 按绝对权重值排序
    sorted_results = sorted(lime_results, key=lambda x: abs(x[1]), reverse=True)
    for feature, weight in sorted_results:
        strength = "强" if abs(weight) > max_weight * 0.5 else "弱"
        direction = "正" if weight > 0 else "负"
        print(f"{feature}: {weight:.4f} ({strength}{direction}相关)")


def main(test_data_path, results_output_path, n_splits=10, sample_idx=0):
    """主函数"""
    # 加载测试集
    test_drug_ids, test_disease_ids, test_dataset = load_data(test_data_path, has_label=False)
    test_feature_dim = test_dataset.tensors[0].shape[1]

    results = []
    all_model_predictions = []
    models_and_paths = get_models_and_paths()

    for model_name, model_info in models_and_paths.items():
        print(f"\n{'=' * 50}\nProcessing model: {model_name}")

        # 加载训练数据
        train_path = model_info['train_path']
        train_drug_ids, train_disease_ids, train_dataset = load_data(train_path, has_label=True)
        train_feature_dim = train_dataset.tensors[0].shape[1]

        # 生成特征名称 (示例)
        feature_names = [f'feature_{i}_x' for i in range(160)] + [f'feature_{i}_y' for i in range(64)]

        # 初始化模型
        model_class = model_info['model_class']
        model = initialize_model(model_class, input_size=train_feature_dim, output_size=1)

        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
            print(f"\nFold {fold + 1}/{n_splits}")

            # 准备训练和验证数据
            train_subset = torch.utils.data.Subset(train_dataset, train_idx)
            val_subset = torch.utils.data.Subset(train_dataset, val_idx)

            X_train_np = train_subset.dataset.tensors[0][train_idx].numpy()
            y_train_np = train_subset.dataset.tensors[1][train_idx].numpy()
            X_val_np = val_subset.dataset.tensors[0][val_idx].numpy()
            y_true = val_subset.dataset.tensors[1][val_idx].numpy()

            # 对齐特征维度
            if X_train_np.shape[1] != X_val_np.shape[1]:
                min_dim = min(X_train_np.shape[1], X_val_np.shape[1])
                X_train_np = X_train_np[:, :min_dim]
                X_val_np = X_val_np[:, :min_dim]

            # 训练和预测
            y_pred, y_pred_prob = model.train_and_predict(X_train_np, y_train_np, X_val_np)

            # 评估
            acc, auc_score, aupr, f1, mcc, sn, sp, precision_val, recall_val = evaluate_model(
                y_true, y_pred_prob, y_pred)

            # 存储结果
            fold_result = {
                'Model': model_name,
                'Fold': fold + 1,
                'TrainingData': os.path.basename(train_path),
                'FeatureDim': train_feature_dim,
                'Accuracy': acc,
                'AUC': auc_score,
                'AUPR': aupr,
                'F1 Score': f1,
                'MCC': mcc,
                'Sensitivity': sn,
                'Specificity': sp,
                'Precision': precision_val,
                'Recall': recall_val,
                'Mean Probability': np.mean(y_pred_prob),
                'Median Probability': np.median(y_pred_prob),
                'Min Probability': np.min(y_pred_prob),
                'Max Probability': np.max(y_pred_prob)
            }
            fold_results.append(fold_result)
            print(f"Fold {fold + 1} Results: {fold_result}")

        results.extend(fold_results)

        # 在整个测试集上进行预测
        print("\nMaking predictions on test set...")
        X_train_np = train_dataset.tensors[0].numpy()
        y_train_np = train_dataset.tensors[1].numpy()
        X_test_np = test_dataset.tensors[0].numpy()

        # 对齐特征维度
        if X_train_np.shape[1] != X_test_np.shape[1]:
            min_dim = min(X_train_np.shape[1], X_test_np.shape[1])
            X_train_np = X_train_np[:, :min_dim]
            X_test_np = X_test_np[:, :min_dim]
        y_pred_prob = model.train_and_predict(X_train_np, y_train_np, X_test_np)[1]
        # 存储每个模型的预测结果
        all_model_predictions.append(y_pred_prob)

    # 只为集成模型生成LIME解释
    print("\nRunning LIME explanation for Ensemble model...")
    try:
        explain_with_lime(
            model=model,
            X_train=X_train_np,
            X_test=X_test_np,
            feature_names=feature_names,
            num_features=25,
            sample_idx=sample_idx
        )
    except Exception as e:
        print(f"LIME explanation failed: {e}")

    # 保存最终结果
    save_results(results, results_output_path)


if __name__ == '__main__':
    # 配置路径参数
    test_data_path = r'D:\a毕设\数据\未归一化\验证\验证_top3_disease_567.csv'
    results_folder = r'D:\a毕设\数据\未归一化\验证结果'
    # 创建结果目录（如果不存在）
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    # 定义结果文件名
    results_output_path = os.path.join(results_folder, '567chills_multi_model.csv')
    # 运行主函数
    main(
        test_data_path=test_data_path,
        results_output_path=results_output_path,
        n_splits=10,  # 10折交叉验证
        sample_idx=1  # 对第一个样本进行分析
    )
