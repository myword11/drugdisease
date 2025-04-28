import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 加载数据
final_reordered = pd.read_csv("D:/a毕设/24.12.26_2101花葛药物-疾病相互作用课题/数据集/B-dataset/未归一化/Association.csv")

# 获取所有特征列名，包含 'feature_x' 和 'feature_y'
features_columns = [col for col in final_reordered.columns if 'feature' in col]

# 检查数据的统计信息
print("Data Description:\n", final_reordered[features_columns].describe())

# 检查每个特征列的唯一值数量
print("\nUnique Values in Each Feature:\n", final_reordered[features_columns].nunique())

# 检查是否有缺失值
print("\nMissing Values in Each Feature:\n", final_reordered[features_columns].isnull().sum())

# 处理缺失值：可以选择填充或删除缺失值
# 这里选择填充缺失值为该列的均值
final_reordered[features_columns] = final_reordered[features_columns].fillna(final_reordered[features_columns].mean())

# 可选：去除异常值（例如，大于3个标准差的值）
# 计算每列的z-score并去除异常值
from scipy.stats import zscore

z_scores = final_reordered[features_columns].apply(zscore)
final_reordered = final_reordered[(z_scores < 3).all(axis=1)]

# 初始化 MinMaxScaler 进行归一化
scaler = MinMaxScaler(feature_range=(0, 1))


# 对特征列进行归一化操作
normalized_features = scaler.fit_transform(final_reordered[features_columns])

# 将归一化后的数据替换原来的特征列
final_reordered[features_columns] = normalized_features

# 保存最终的归一化数据
final_reordered.to_csv("D:/a毕设/数据/归一化/Association.csv", index=False)

print("Data normalization completed and saved.")
