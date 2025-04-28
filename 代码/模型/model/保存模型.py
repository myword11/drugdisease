import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, f1_score, confusion_matrix, auc, \
    matthews_corrcoef, precision_score, recall_score
from sklearn.model_selection import train_test_split
from scipy.stats import mode
from decision_tree_model import DecisionTreeModel
from random_forest_model import RandomForestModel
from xgboost_model import XGBoostModel
from feature_selector import FeatureSelector  # 假设你的 FeatureSelector 也在这里


def load_data(file_path):
    """加载数据，不进行任何特征处理"""
    data = pd.read_csv(file_path)
    X = data.iloc[:, 3:].values.astype('float32')  # 跳过前3列
    y = data['label'].values.astype('float32')
    return X, y


def evaluate_model(y_true, y_pred_prob, y_pred):
    """评估模型性能"""
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
    """保存评估结果"""
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)


def get_models():
    """获取模型配置（每个模型对应一个特征选择器）"""
    return {
        'DecisionTree': {'model': DecisionTreeModel, 'selector': FeatureSelector(k=140)},
        'RandomForest': {'model': RandomForestModel, 'selector': FeatureSelector(k=90)},
        'XGBoost': {'model': XGBoostModel, 'selector': FeatureSelector(k=160)}
    }


def initialize_model(model_class):
    """初始化模型"""
    return model_class()


def save_ensemble_model(model_dict, output_dir):
    """保存集成模型"""
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model_dict, os.path.join(output_dir, 'ensemble_model.pkl'))


def train_and_evaluate_models(X_train, y_train, X_test, y_test, models_to_use):
    """训练和评估模型（加入特征选择）"""
    trained_models = {}
    all_predictions = []
    results = []

    for model_name, model_info in models_to_use.items():
        model_class = model_info['model']
        feature_selector = model_info['selector']
        print(f"\nTraining {model_name}...")

        # 使用特征选择器选择特征
        X_train_selected = feature_selector.fit_transform(X_train, y_train)  # 传递y_train
        X_test_selected = feature_selector.transform(X_test)

        # 训练模型
        model = initialize_model(model_class)
        y_pred, y_pred_prob = model.train_and_predict(X_train_selected, y_train, X_test_selected)
        y_pred_binary = (np.array(y_pred_prob) >= 0.5).astype(int)

        # 保存模型和特征选择器
        trained_models[model_name] = {
            'model': model,  # 只保存模型对象
            'selector': feature_selector  # 保存特征选择器
        }

        # 评估模型性能
        acc, auc_score, aupr, f1, mcc, sn, sp, precision_val, recall_val = evaluate_model(
            y_test, y_pred_prob, y_pred_binary)

        results.append({
            'Model': model_name,
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

        # 保存所有预测值（用于硬投票集成）
        all_predictions.append(y_pred_binary)

    # 硬投票集成
    final_predictions, _ = mode(np.stack(all_predictions), axis=0)
    final_predictions = final_predictions.flatten()

    # 评估集成模型
    acc, auc_score, aupr, f1, mcc, sn, sp, precision_val, recall_val = evaluate_model(
        y_test, final_predictions, final_predictions)

    results.append({
        'Model': 'HardVoting',
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

    # 保存结果和模型
    save_results(results, results_output_path)
    model_save_dir = os.path.join(os.path.dirname(results_output_path), 'saved_models')
    save_ensemble_model(trained_models, model_save_dir)



def main(data_paths, results_output_path, models_to_use):
    """主函数"""
    trained_models = {}
    all_predictions = []
    results = []

    for model_name, model_info in models_to_use.items():
        # 加载对应的数据集
        data_path = data_paths.get(model_name)
        if data_path is None:
            print(f"Warning: No data path found for {model_name}. Skipping.")
            continue

        # 加载数据
        X, y = load_data(data_path)

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42)

        # 获取模型配置
        model_class = model_info['model']
        feature_selector = model_info['selector']

        print(f"\nTraining {model_name}...")

        # 使用特征选择器选择特征
        X_train_selected = feature_selector.fit_transform(X_train, y_train)
        X_test_selected = feature_selector.transform(X_test)

        # 训练模型
        model = initialize_model(model_class)
        y_pred, y_pred_prob = model.train_and_predict(X_train_selected, y_train, X_test_selected)
        y_pred_binary = (np.array(y_pred_prob) >= 0.5).astype(int)

        # 保存模型和特征选择器
        trained_models[model_name] = {
            'model': model,  # 只保存模型对象
            'selector': feature_selector  # 保存特征选择器
        }

        # 评估模型性能
        acc, auc_score, aupr, f1, mcc, sn, sp, precision_val, recall_val = evaluate_model(
            y_test, y_pred_prob, y_pred_binary)

        results.append({
            'Model': model_name,
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

        # 保存所有预测值（用于硬投票集成）
        all_predictions.append(y_pred_binary)

    # 硬投票集成
    final_predictions, _ = mode(np.stack(all_predictions), axis=0)
    final_predictions = final_predictions.flatten()

    # 评估集成模型
    acc, auc_score, aupr, f1, mcc, sn, sp, precision_val, recall_val = evaluate_model(
        y_test, final_predictions, final_predictions)

    results.append({
        'Model': 'HardVoting',
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

    # 保存结果和模型
    save_results(results, results_output_path)
    model_save_dir = os.path.join(os.path.dirname(results_output_path), 'saved_models')
    save_ensemble_model(trained_models, model_save_dir)



class EnsemblePredictor:
    """集成模型预测器（简化版）"""

    def __init__(self, model_path):
        self.models = joblib.load(model_path)  # 直接加载模型字典

    def predict(self, X_new):
        """预测类别"""
        all_predictions = []
        for name, comp in self.models.items():
            model = comp['model']
            selector = comp['selector']
            # 使用特征选择器选择特征
            X_new_selected = selector.transform(X_new)
            preds = (np.array(model.predict_proba(X_new_selected)) >= 0.5).astype(int)
            all_predictions.append(preds)

        final_predictions, _ = mode(np.stack(all_predictions), axis=0)
        return final_predictions.flatten()

    def predict_proba(self, X_new):
        """预测概率"""
        all_probs = []
        for comp in self.models.values():
            model = comp['model']
            selector = comp['selector']
            # 使用特征选择器选择特征
            X_new_selected = selector.transform(X_new)
            probs = model.predict_proba(X_new_selected)
            all_probs.append(probs)

        return np.mean(all_probs, axis=0)


if __name__ == '__main__':
    # 配置路径
    data_paths = {
        'DecisionTree': r'D:\a毕设\数据\未归一化\特征选择数据\140\f_classif_汉明140.csv',
        'XGBoost': r'D:\a毕设\数据\未归一化\特征选择数据\160\f_classif_汉明160.csv',
        'RandomForest': r'D:\a毕设\数据\未归一化\特征选择数据\90\f_classif_切比雪夫90.csv',
    }

    results_folder = r'D:/a毕设/数据/未归一化/集成模型(ksu)/模型'
    os.makedirs(results_folder, exist_ok=True)
    results_output_path = os.path.join(results_folder, 'ensemble_results.csv')

    models_to_use = get_models()
    main(data_paths, results_output_path, models_to_use)
