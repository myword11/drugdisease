# svm_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.svm import LinearSVC
import os

class LinearSVCModel:
    def __init__(self, random_state=42, max_iter=1000, dual=False, tol=1e-4):
        self.model = LinearSVC(random_state=random_state, max_iter=max_iter, dual=dual, tol=tol)
        self.kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)

    def train_and_predict(self, X_train, y_train, X_test):
        # 超参数调优
        param_grid = {'C': [0.1, 1, 10]}
        grid_search = GridSearchCV(LinearSVC(random_state=42, max_iter=1000, dual=False, tol=1e-4), param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        # 训练模型
        best_model.fit(X_train, y_train)

        # 在测试集上进行预测
        y_pred = best_model.predict(X_test)
        y_pred_prob = best_model.decision_function(X_test)

        return y_pred, y_pred_prob
