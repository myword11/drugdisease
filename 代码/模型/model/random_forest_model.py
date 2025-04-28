import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import os

class RandomForestModel:
    def __init__(self, n_estimators=30, max_depth=5, min_samples_split=15, min_samples_leaf=10, max_features='sqrt', random_state=42, class_weight='balanced'):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
            class_weight=class_weight
        )
        self.kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    def train_and_predict(self, X_train, y_train, X_test):
        """训练模型并进行预测"""
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        y_pred_prob = self.model.predict_proba(X_test)[:, 1]  # 获取正类的预测概率
        return y_pred, y_pred_prob

    def predict_proba(self, X):
        """封装预测概率的方法"""
        return self.model.predict_proba(X)
