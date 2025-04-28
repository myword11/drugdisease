import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import shap
import time
import gc
from sklearn.model_selection import train_test_split

# 加载数据
data = pd.read_csv('D:/a毕设/数据/未归一化/合并/ClusterCentroids.csv')

# 提取药物和疾病的特征列
X = data.iloc[:, 3:303]  # 药物特征
disease_columns = [col for col in data.columns if col.endswith('_y')]  # 疾病特征列

# 提取标签列
y = data['drug']  # 目标变量列

# 对特征进行标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 打印标准化后的数据维度
print(f"Scaled data shape: {X_scaled.shape}")

# 使用随机森林进行特征选择
model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)  # 减少树的数量和深度

# 训练模型
model.fit(X_scaled, y)
print("Model training completed.")

# 使用分层采样选择数据的子集
subset_size = 20000  # 选择1000个样本
X_subset, _, y_subset, _ = train_test_split(X_scaled, y, test_size=0.9, stratify=y, random_state=42)

# 使用 SHAP 进行特征重要性计算
explainer = shap.TreeExplainer(model)
print("Starting SHAP values calculation...")

start_time = time.time()
shap_values = explainer.shap_values(X_subset)
end_time = time.time()
print(f"SHAP values calculated in {end_time - start_time:.2f} seconds.")

# 计算每个特征的 SHAP 值的绝对值的平均值
shap_importances = np.abs(shap_values).mean(axis=0).mean(axis=0)
print("SHAP importances calculated.")
print(f"SHAP importances: {shap_importances}")

# 定义 k 的范围
k_values = range(90, 190, 10)

for k in k_values:
    print(f"Selecting {k} features using SHAP...")
    important_features = shap_importances.argsort()[::-1][:k]  # 选取前k个最重要的特征

    # 获取选中的特征矩阵
    X_selected = X.iloc[:, important_features]

    # 将选中的特征与原来的药物、疾病、label列合并
    selected_data = X_selected.copy()
    selected_data['drug'] = y
    selected_data['disease'] = data['disease']
    selected_data['label'] = data['label']

    # 将疾病的特征列添加到数据框中
    selected_data = pd.concat([selected_data, data[disease_columns]], axis=1)

    # 调整列顺序，将 'drug', 'disease', 'label' 放在最前面
    selected_data = selected_data[['drug', 'disease', 'label'] + [col for col in selected_data.columns if col not in ['drug', 'disease', 'label']]]

    # 保存为新的CSV文件
    output_file = f'D:/a毕设/数据/未归一化/特征选择数据/{k}/SHAP_ClusterCentroids{k}.csv'
    selected_data.to_csv(output_file, index=False)
    print(f"已将特征选择后的数据保存为 '{output_file}'")

    # 释放内存
    del X_selected, selected_data
    gc.collect()
