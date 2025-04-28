import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import os


class XGBoostModel:
    def __init__(self, random_state=42, eval_metric='logloss'):
        # 固定超参数设置
        self.fixed_params = {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }

        self.model = XGBClassifier(
            random_state=random_state,
            eval_metric=eval_metric,
            **self.fixed_params  # 直接使用固定参数
        )
        self.kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)

    def train_and_predict(self, X_train, y_train, X_test):
        # 直接使用固定参数训练模型
        best_model = self.model

        # 输出使用的固定参数
        print("\n=== Fixed Hyperparameters ===")
        for param, value in self.fixed_params.items():
            print(f"{param}: {value}")

        # 训练模型
        best_model.fit(X_train, y_train)

        # 在测试集上进行预测
        y_pred = best_model.predict(X_test)
        y_pred_prob = best_model.predict_proba(X_test)[:, 1]  # 获取正类的预测概率

        return y_pred, y_pred_prob

    def predict_proba(self, X_test):
        """
        直接调用 XGBoost 模型的 predict_proba 方法。
        返回测试数据集 X_test 上每个样本的概率预测。
        """
        return self.model.predict_proba(X_test)  # 返回所有类别的预测概率


