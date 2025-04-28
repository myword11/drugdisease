import pandas as pd
from sklearn.preprocessing import StandardScaler

# 假设数据已经加载并且命名正确，这里就直接操作

# 示例数据（假设 `final_reordered` 是已经合并好的 DataFrame）
final_reordered = pd.read_csv("D:/a毕设/24.12.26_2101花葛药物-疾病相互作用课题/数据集/B-dataset/未归一化/disAssociation.csv")

# 获取所有特征列名，包含 'feature_x' 和 'feature_y'
features_columns = [col for col in final_reordered.columns if 'feature' in col]

# 初始化 StandardScaler 进行标准化
scaler = StandardScaler()

# 对特征列进行标准化操作
standardized_features = scaler.fit_transform(final_reordered[features_columns])

# 将标准化后的数据替换原来的特征列
final_reordered[features_columns] = standardized_features

# 保存最终的标准化数据
final_reordered.to_csv("D:/a毕设/数据/标准化/disAssociation.csv", index=False)

print("Data standardization completed and saved.")
