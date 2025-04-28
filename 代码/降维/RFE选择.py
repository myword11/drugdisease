import pandas as pd
from sklearn.feature_selection import RFE, SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# 加载数据
data = pd.read_csv('D:/a毕设/数据/未归一化/合并/ClusterCentroids.csv')

# 检查数据中的缺失值
print(f"Missing values in each column:\n{data.isnull().sum()}")

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

# 使用更简单、更快的模型（如SVM或随机森林）
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)  # 使用n_jobs=-1进行并行化

# 使用递归特征消除（RFE）进行特征选择
# 定义 k 的范围
k_values = range(90, 190, 10)

for k in k_values:
    print(f"Selecting {k} features using RFE...")
    selector = RFE(model, n_features_to_select=k, step=10)  # 每次移除更多特征，减少迭代次数
    X_selected = selector.fit_transform(X_scaled, y)

    # 打印选择的特征索引
    selected_features = selector.get_support(indices=True)
    print(f"Selected feature indices: {selected_features}")

    # 创建新的特征矩阵（选中的药物特征）
    X_selected = X.iloc[:, selected_features]

    # 将选中的特征与原来的药物、疾病、label列合并
    selected_data = X_selected.copy()
    selected_data['drug'] = data['drug']
    selected_data['disease'] = data['disease']
    selected_data['label'] = data['label']

    # 将疾病的特征列添加到数据框中
    selected_data = pd.concat([selected_data, data[disease_columns]], axis=1)

    # 调整列顺序，将 'drug', 'disease', 'label' 放在最前面
    selected_data = selected_data[['drug', 'disease', 'label'] + [col for col in selected_data.columns if
                                                                  col not in ['drug', 'disease', 'label']]]

    # 保存为新的CSV文件
    output_file = f'D:/a毕设/数据/未归一化/特征选择数据/{k}/RFE_ClusterCentroids{k}.csv'
    selected_data.to_csv(output_file, index=False)
    print(f"已将特征选择后的数据保存为 '{output_file}'")