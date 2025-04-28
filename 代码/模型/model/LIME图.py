import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
from decision_tree_model import DecisionTreeModel
from xgboost_model import XGBoostModel
from random_forest_model import RandomForestModel
from lime.lime_tabular import LimeTabularExplainer


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


def plot_ensemble_decision_boundary(X, y, models, model_names, feature_names):
    """绘制集成模型的决策边界"""
    # 获取网格范围
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    # 初始化用于存储所有模型的预测
    model_predictions = []

    # 获取每个模型的预测结果
    for model, model_name in zip(models, model_names):
        model.train_and_predict(X, y, X)  # 使用train_and_predict方法训练并预测
        # 使用训练后的模型预测网格点的概率
        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
        Z = Z[:, 1]  # 取正类的概率（分类问题）
        model_predictions.append(Z)

    # 将所有模型的预测结果进行合成，例如通过平均
    ensemble_prediction = np.mean(model_predictions, axis=0)

    # 将预测结果重新塑形为网格的形状
    ensemble_prediction = ensemble_prediction.reshape(xx.shape)

    # 绘制决策边界
    plt.contourf(xx, yy, ensemble_prediction, alpha=0.75, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=50, cmap='coolwarm')
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.title("Ensemble Decision Boundary Visualization")
    plt.show()

def main(test_data_path, results_output_path, n_splits=10, sample_idx=0):
    """主函数"""
    # 加载测试集
    test_drug_ids, test_disease_ids, test_dataset = load_data(test_data_path, has_label=False)
    test_feature_dim = test_dataset.tensors[0].shape[1]
    results = []
    all_model_predictions = []
    models_and_paths = get_models_and_paths()

    # 初始化模型列表和模型名称
    models = []
    model_names = []

    for model_name, model_info in models_and_paths.items():
        print(f"\n{'=' * 50}\nProcessing model: {model_name}")

        # 加载训练数据
        train_path = model_info['train_path']
        train_drug_ids, train_disease_ids, train_dataset = load_data(train_path, has_label=True)
        train_feature_dim = train_dataset.tensors[0].shape[1]

        # 选择两个特征进行决策边界可视化
        X_train_np = train_dataset.tensors[0].numpy()[:, :2]  # 只选择前两个特征
        y_train_np = train_dataset.tensors[1].numpy()

        # 初始化模型
        model_class = model_info['model_class']
        model = initialize_model(model_class, input_size=X_train_np.shape[1], output_size=1)

        # 将模型和名称加入列表
        models.append(model)
        model_names.append(model_name)

    # 绘制集成模型的决策边界
    plot_ensemble_decision_boundary(X_train_np, y_train_np, models, model_names, feature_names=['feature_204_x', 'feature_111_y'])

    # 保存所有结果
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
        sample_idx=0  # 对第一个样本进行分析
    )
