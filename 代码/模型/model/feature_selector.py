from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np


class FeatureSelector:
    def __init__(self, k=100):
        self.k = k
        self.selector = None
        self.feature_config = {}  # 用于保存每个模型对应的特征索引

    def fit(self, X, y):
        """根据输入数据X和目标y使用F检验选择前k个特征"""
        print(f"Fitting with X shape: {X.shape}, y shape: {y.shape}, k={self.k}")
        if self.k > X.shape[1]:
            self.k = X.shape[1]  # 如果k大于特征数，则自动调整k
            print(f"Adjusted k to {self.k} based on X features.")

        # 使用F检验（f_classif）选择前k个特征
        self.selector = SelectKBest(score_func=f_classif, k=self.k)

        # 计算F值并忽略NaN值
        with np.errstate(divide='ignore', invalid='ignore'):
            self.selector.fit(X, y)
            self.selector.scores_ = np.nan_to_num(self.selector.scores_)  # 处理可能的NaN值

        # 保存选中的特征索引
        selected_indices = self.selector.get_support(indices=True)
        return self

    def transform(self, X):
        """应用已选择的特征"""
        if self.selector is None:
            raise ValueError("FeatureSelector has not been fitted yet.")
        print(f"Transforming with X shape: {X.shape}")
        return self.selector.transform(X)

    def fit_transform(self, X, y):
        """结合fit和transform"""
        self.fit(X, y)
        return self.transform(X)

    def set_feature_config(self, model_name, selected_indices):
        """保存每个模型对应的选中特征索引"""
        self.feature_config[model_name] = selected_indices

    def get_selected_features(self, model_name):
        """获取模型对应的选中特征索引"""
        return self.feature_config.get(model_name, [])
