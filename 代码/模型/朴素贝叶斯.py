import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, matthews_corrcoef, roc_auc_score,
                             recall_score, confusion_matrix, f1_score, precision_score, precision_recall_curve, auc)
import os
from collections import Counter

# 定义数据集路径和结果文件路径
datasets = [
    {
        "name": "KSU",
        "path": r"D:/a毕设/数据/未归一化/合并/KSU_欧几里得.csv",
        "output_dir": r"D:/a毕设/数据/十折/naive_bayes/",
        "results_file": r"D:/a毕设/数据/未归一化/结果/naive_bayes/naive_bayes_KSU_欧几里得.csv"
    },
    {
        "name": "NCR",
        "path": r"D:/a毕设/数据/未归一化/合并/NCR.csv",
        "output_dir": r"D:/a毕设/数据/十折/naive_bayes/",
        "results_file": r"D:/a毕设/数据/未归一化/结果/naive_bayes/naive_bayes_NCR.csv"
    },
    {
        "name": "NearMiss",
        "path": r"D:/a毕设/数据/未归一化/合并/NearMiss.csv",
        "output_dir": r"D:/a毕设/数据/十折/naive_bayes/",
        "results_file": r"D:/a毕设/数据/未归一化/结果/naive_bayes/naive_bayes_NearMiss.csv"
    },
    {
        "name": "OSS",
        "path": r"D:/a毕设/数据/未归一化/合并/OSS.csv",
        "output_dir": r"D:/a毕设/数据/十折/naive_bayes/",
        "results_file": r"D:/a毕设/数据/未归一化/结果/naive_bayes/naive_bayes_OSS.csv"
    },
    {
        "name": "Random",
        "path": r"D:/a毕设/数据/未归一化/合并/Random.csv",
        "output_dir": r"D:/a毕设/数据/十折/naive_bayes/",
        "results_file": r"D:/a毕设/数据/未归一化/结果/naive_bayes/naive_bayes_Random.csv"
    },
    {
        "name": "ClusterCentroids",
        "path": r"D:/a毕设/数据/未归一化/合并/ClusterCentroids.csv",
        "output_dir": r"D:/a毕设/数据/十折/naive_bayes/",
        "results_file": r"D:/a毕设/数据/未归一化/结果/naive_bayes/naive_bayes_ClusterCentroids.csv"
    }
    # 可以继续添加更多的数据集
]

# 初始化十折交叉验证
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# 处理每个数据集
for dataset in datasets:
    print(f"Processing dataset: {dataset['name']}")

    # 读取数据
    df = pd.read_csv(dataset['path'])

    # 去掉列名中的前后空格，避免因空格导致匹配失败
    df.columns = df.columns.str.strip()

    # 提取药物和疾病特征的列名
    drug_columns = [col for col in df.columns if col.endswith('_x')]
    disease_columns = [col for col in df.columns if col.endswith('_y')]

    # 提取药物和疾病的序号（id）和标签
    df['drug'] = df['drug']
    df['disease'] = df['disease']
    df['label'] = df['label']  # 标签列

    # 提取药物特征矩阵和疾病特征矩阵
    X_drugs = df[drug_columns].values if drug_columns else np.zeros((df.shape[0], 0))
    X_diseases = df[disease_columns].values if disease_columns else np.zeros((df.shape[0], 0))

    # 合并药物和疾病特征
    X = np.hstack([X_drugs, X_diseases])

    # 提取标签
    y = df['label'].values

    # 检查数据集平衡性
    print("Data distribution:")
    print(Counter(y))

    # 使用朴素贝叶斯 (GaussianNB)
    model = GaussianNB()

    # 用于存储每一折的结果
    results = []

    # 创建输出目录
    os.makedirs(dataset['output_dir'], exist_ok=True)

    # 十折交叉验证
    for fold, (train_index, test_index) in enumerate(kf.split(X, y), 1):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # 获取训练集和验证集的 DataFrame
        train_df = df.iloc[train_index]
        test_df = df.iloc[test_index]

        # 保存训练集和验证集为 CSV 文件
        # train_file = f"{dataset['output_dir']}fold_train_{fold}.csv"
        # test_file = f"{dataset['output_dir']}fold_test_{fold}.csv"
        # train_df.to_csv(train_file, index=False)
        # test_df.to_csv(test_file, index=False)

        # 训练模型
        model.fit(X_train, y_train)

        # 在测试集上进行预测
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]  # 获取概率估计

        # 计算评估指标
        acc = accuracy_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_prob)
        sn = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision_val = precision_score(y_test, y_pred)
        recall_val = recall_score(y_test, y_pred)

        # 计算特异性
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sp = tn / (tn + fp)

        # 计算 AUPR
        precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
        aupr = auc(recall, precision)

        # 存储每一折的结果
        results.append({
            "Fold": fold,
            "Accuracy (ACC)": acc,
            "AUC": auc_score,
            "AUPR": aupr,
            "Precision": precision_val,
            "Recall": recall_val,
            "F1 Score": f1,
            "MCC": mcc,
            "Sensitivity (Sn)": sn,
            "Specificity (Sp)": sp
        })

        # 打印每一折的评估指标
        print(f"Fold {fold}:")
        print(f"  Accuracy (ACC): {acc:.4f}")
        print(f"  AUC: {auc_score:.4f}")
        print(f"  AUPR: {aupr:.4f}")
        print(f"  Precision: {precision_val:.4f}")
        print(f"  Recall: {recall_val:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  MCC: {mcc:.4f}")
        print(f"  Sensitivity (Sn): {sn:.4f}")
        print(f"  Specificity (Sp): {sp:.4f}")
        print("-" * 50)

    # 将结果存储到 DataFrame
    results_df = pd.DataFrame(results)

    # 计算平均值并添加到 DataFrame
    average_results = {
        "Fold": "Average",
        "Accuracy (ACC)": np.mean(results_df["Accuracy (ACC)"]),
        "AUC": np.mean(results_df["AUC"]),
        "AUPR": np.mean(results_df["AUPR"]),
        "Precision": np.mean(results_df["Precision"]),
        "Recall": np.mean(results_df["Recall"]),
        "F1 Score": np.mean(results_df["F1 Score"]),
        "MCC": np.mean(results_df["MCC"]),
        "Sensitivity (Sn)": np.mean(results_df["Sensitivity (Sn)"]),
        "Specificity (Sp)": np.mean(results_df["Specificity (Sp)"])
    }
    results_df = pd.concat([results_df, pd.DataFrame([average_results])], ignore_index=True)

    # 保存结果到 CSV 文件
    results_df.to_csv(dataset['results_file'], index=False)
    print(f"\nResults have been saved to {dataset['results_file']}")
