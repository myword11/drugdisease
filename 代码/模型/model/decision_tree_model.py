import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.exceptions import ConvergenceWarning
import warnings
import os

class DecisionTreeModel:
    def __init__(self, random_state=42, input_size=None, output_size=None):
        # 固定超参数设置
        self.fixed_params = {
            'max_depth': None,
            'min_samples_split': 10,
            'min_samples_leaf': 5
        }

        # 创建决策树模型，直接使用固定参数
        self.model = DecisionTreeClassifier(
            random_state=random_state,
            **self.fixed_params
        )
        self.kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        self.tree_params_ = None  # 存储训练后的树参数

    def fit(self, X_train, y_train):
        """
        训练模型并保存训练后的模型参数
        """
        try:
            # 训练模型
            self.model.fit(X_train, y_train)

            # 保存训练后的实际参数
            self.tree_params_ = {
                'actual_depth': self.model.get_depth(),
                'n_leaves': self.model.get_n_leaves(),
                'feature_importances': self.model.feature_importances_
            }

            print(f"Model trained with depth: {self.tree_params_['actual_depth']}, "
                  f"n_leaves: {self.tree_params_['n_leaves']}")
            return self  # 返回训练好的模型

        except Exception as e:
            print(f"Model training failed: {str(e)}")
            return None

    def predict(self, X_test):
        """
        使用训练好的模型进行预测
        """
        try:
            y_pred = self.model.predict(X_test)  # 预测类别标签
            y_pred_prob = self.model.predict_proba(X_test)[:, 1] if hasattr(self.model, "predict_proba") else y_pred
            return y_pred, y_pred_prob
        except Exception as e:
            print(f"Prediction failed: {str(e)}")
            return None, None

    def predict_proba(self, X_test):
        """
        获取概率预测
        """
        try:
            return self.model.predict_proba(X_test)
        except Exception as e:
            print(f"Error in predicting probabilities: {str(e)}")
            return None

    def train_and_predict(self, X_train, y_train, X_test):
        """
        训练模型并进行预测
        """
        warnings.filterwarnings('ignore', category=ConvergenceWarning)

        # 训练模型
        self.fit(X_train, y_train)

        # 使用训练好的模型进行预测
        return self.predict(X_test)
