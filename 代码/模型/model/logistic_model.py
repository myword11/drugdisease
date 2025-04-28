# logistic_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
import os

class LogisticRegressionModel:
    def __init__(self, random_state=42, max_iter=1000, solver='liblinear'):
        self.model = LogisticRegression(random_state=random_state, max_iter=max_iter, solver=solver)
        self.kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)

    def train_and_predict(self, X_train, y_train, X_test):
        # 超参数调优
        param_grid = {'C': [0.1, 1, 10]}
        grid_search = GridSearchCV(LogisticRegression(random_state=42, max_iter=1000, solver='liblinear'), param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        # 训练模型
        best_model.fit(X_train, y_train)

        # 在测试集上进行预测
        y_pred = best_model.predict(X_test)
        y_pred_prob = best_model.predict_proba(X_test)[:, 1]

        return y_pred, y_pred_prob
