import pandas as pd
import numpy as np
from imblearn.under_sampling import NeighbourhoodCleaningRule
from collections import Counter
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler

group_sizes = [3423] * 5 + [1301]


# NeighbourhoodCleaningRule 下采样方法
def UnderNCR(req, X, y, meta):
    n_neighbors = int(req.form.get(f'NCR_n_neighbors'))
    sampling_strategy = req.form.get(f'NCR_sampling_strategy')
    # 使用 NeighbourhoodCleaningRule 进行欠采样
    ncr = NeighbourhoodCleaningRule(sampling_strategy=sampling_strategy, n_neighbors=n_neighbors)
    resampled_data, resampled_labels = ncr.fit_resample(X, y)
    # 获取下采样后的索引
    selected_indices = ncr.sample_indices_
    # 根据下采样后的索引提取原始meta数据
    meta_resampled = meta.iloc[selected_indices].reset_index(drop=True)
    return resampled_data, resampled_labels, meta_resampled


def load_data():
    try:
        print("开始读取数据...")
        final_reordered = pd.read_csv(
            "D:/a毕设/24.12.26_2101花葛药物-疾病相互作用课题/数据集/B-dataset/未归一化/disAssociation.csv")
        print("数据读取成功。")
    except Exception as e:
        print(f"读取数据时发生错误: {e}")
        raise
    return final_reordered


def assign_drug_group(df, total_drugs=269, group_sizes=[3423] * 5 + [1301]):
    drug_indices = np.arange(total_drugs)
    group_boundaries = np.linspace(0, 249, num=6, dtype=int)
    group_boundaries = np.append(group_boundaries, total_drugs)
    df['drug_group'] = pd.cut(df['drug'].astype(int),
                              bins=group_boundaries,
                              labels=[f'Group{i + 1}' for i in range(6)], right=False)
    return df


def group_downsampling_with_NCR(df, group_sizes):
    resampled_data = []
    resampled_labels = []
    for group, size in zip(df['drug_group'].unique(), group_sizes):
        print(f"正在处理分组: {group}")
        group_data = df[df['drug_group'] == group]
        feature_columns = [col for col in group_data.columns if 'feature' in col]
        group_data_features = group_data[feature_columns].values
        group_data_labels = group_data['drug'].values.astype(int)  # 确保标签是数值类型
        group_data_diseases = group_data['disease'].values
        unique_labels = np.unique(group_data_labels)

        # 使用 NeighbourhoodCleaningRule 进行欠采样
        class MockRequest:
            def __init__(self):
                self.form = {
                    'NCR_n_neighbors': '6',  # 设置邻居数
                    'NCR_sampling_strategy': 'auto',  # 使用默认采样策略
                }

        request = MockRequest()
        X_resampled, y_resampled, meta_resampled = UnderNCR(request, group_data_features, group_data_labels,
                                                            group_data[['drug', 'disease']])

        print(f"Group: {group}, 欠采样后数据集大小: {X_resampled.shape}")

        # 保留疾病序号
        group_data_resampled = pd.DataFrame(X_resampled, columns=feature_columns)
        group_data_resampled['disease'] = meta_resampled['disease']  # 保留疾病序号
        group_data_resampled['drug_group'] = group  # 保留 drug_group 列
        group_data_resampled['drug'] = y_resampled  # 保留 drug 列

        # 确保药物和疾病序号在最前面
        group_data_resampled = group_data_resampled[['drug', 'disease'] + feature_columns + ['drug_group']]

        # 确保每个组的样本数量与预期一致
        if len(group_data_resampled) > size:
            group_data_resampled = group_data_resampled.sample(n=size, random_state=42)

        resampled_data.append(group_data_resampled)
        resampled_labels.extend(y_resampled)

    final_resampled_data = pd.concat(resampled_data)
    final_resampled_labels = np.array(resampled_labels)

    # 确保每个组的样本数量与预期一致
    for group, size in zip(df['drug_group'].unique(), group_sizes):
        group_data = final_resampled_data[final_resampled_data['drug_group'] == group]
        if len(group_data) != size:
            print(f"Warning: Group '{group}' has {len(group_data)} samples, expected {size}.")
            if len(group_data) < size:
                # 从原始数据中补充样本
                original_group_data = df[df['drug_group'] == group]
                additional_samples = original_group_data.sample(n=size - len(group_data), replace=True, random_state=42)
                group_data = pd.concat([group_data, additional_samples])
                final_resampled_data = pd.concat(
                    [final_resampled_data[final_resampled_data['drug_group'] != group], group_data])

    return final_resampled_data, final_resampled_labels


def save_resampled_data(resampled_data):
    try:
        print("开始保存下采样后的数据...")
        resampled_data = resampled_data.drop(columns=['drug_group'])
        resampled_data['drug'] = resampled_data['drug'].astype(int)
        resampled_data['disease'] = resampled_data['disease'].astype(int)
        resampled_data = resampled_data.sort_values(by=['drug', 'disease'], ascending=True)
        resampled_data.to_csv("D:/a毕设/数据/未归一化/disAssociation_NCR.csv", index=False)
        print("数据下采样完成，已保存到文件。")
    except Exception as e:
        print(f"保存文件时发生错误: {e}")


if __name__ == '__main__':
    # 读取标准化后的数据
    final_reordered = load_data()
    final_reordered = assign_drug_group(final_reordered)
    try:
        print("开始分组下采样...")
        resampled_data, resampled_labels = group_downsampling_with_NCR(final_reordered, group_sizes)
        print(f"下采样完成，数据形状: {resampled_data.shape}")
        group_counts = resampled_data['drug_group'].value_counts()
        print("采样后的每组样本数量：")
        print(group_counts)
        save_resampled_data(resampled_data)
    except Exception as e:
        print(f"下采样过程中发生错误: {e}")
        raise
