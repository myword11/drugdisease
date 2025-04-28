import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif  # 改用 f_classif

# 加载数据
data = pd.read_csv('D:/a毕设/数据/未归一化/合并/ClusterCentroids.csv')
X = data.iloc[:, 3:303]  # 药物特征
y = data['drug']  # 目标变量

# 定义 k 的范围
k_values = range(140, 150, 10)

for k in k_values:
    print(f"Selecting {k} features using SelectKBest with f_classif...")
    selector = SelectKBest(f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_features = selector.get_support(indices=True)
    print(f"Selected feature indices: {selected_features}")

    # 创建新的特征矩阵（选中的药物特征）
    X_selected_df = X.iloc[:, selected_features]

    # 合并其他列（疾病、标签等）
    selected_data = X_selected_df.copy()
    selected_data['drug'] = data['drug']
    selected_data['disease'] = data['disease']
    selected_data['label'] = data['label']

    # 添加疾病特征列
    disease_columns = [col for col in data.columns if col.endswith('_y')]
    selected_data = pd.concat([selected_data, data[disease_columns]], axis=1)

    # 调整列顺序
    selected_data = selected_data[['drug', 'disease', 'label'] + [col for col in selected_data.columns
                                                                  if col not in ['drug', 'disease', 'label']]]

    # 保存为新的CSV文件
    output_file = f'D:/a毕设/数据/未归一化/特征选择数据/{k}/f_classif_ClusterCentroids{k}.csv'
    selected_data.to_csv(output_file, index=False)
    print(f"已将特征选择后的数据保存为 '{output_file}'")
