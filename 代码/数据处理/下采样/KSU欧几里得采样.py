import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from collections import Counter

group_sizes = [3423] * 5 + [1301]

# 读取数据集
def load_data():
    try:
        print("开始读取数据...")
        final_reordered = pd.read_csv(
            "D:/a毕设/数据/标准化/disAssociation.csv")
        print("数据读取成功。")
    except Exception as e:
        print(f"读取数据时发生错误: {e}")
        raise
    return final_reordered

# 分配药物组
def assign_drug_group(df, total_drugs=269, group_sizes=[3423] * 5 + [1301]):
    drug_indices = np.arange(total_drugs)
    group_boundaries = np.linspace(0, 249, num=6, dtype=int)
    group_boundaries = np.append(group_boundaries, total_drugs)
    df['drug_group'] = pd.cut(df['drug'].astype(int),
                              bins=group_boundaries,
                              labels=[f'Group{i + 1}' for i in range(6)], right=False)
    return df

# KSU 下采样方法
def KSU(data, labels, sampling_strategy):
    def undersample_class(class_data, num_samples):
        if len(class_data) <= num_samples:
            return class_data
        # 计算成对欧几里得距离
        distances = squareform(pdist(class_data, 'euclidean'))
        # 将对角线设为无穷大以避免自选
        np.fill_diagonal(distances, np.inf)
        # 获取要保留的样本索引
        keep_indices = np.argsort(np.min(distances, axis=1))[:num_samples]
        # 选择要保留的样本
        sampled_data = class_data[keep_indices, :]
        return sampled_data

    resampled_data = []
    resampled_labels = []
    for label, num_samples in sampling_strategy.items():
        class_data = data[labels == label].astype(float)
        sampled_class_data = undersample_class(class_data, num_samples)
        resampled_data.append(sampled_class_data)
        resampled_labels.extend([label] * sampled_class_data.shape[0])
    resampled_data = np.vstack(resampled_data)
    resampled_labels = np.array(resampled_labels)
    return resampled_data, resampled_labels

# 分组下采样
def group_downsampling_with_KSU(df, group_sizes, feature_columns):
    resampled_data = []
    resampled_labels = []
    for group, size in zip(df['drug_group'].unique(), group_sizes):
        print(f"正在处理分组: {group}")
        group_data = df[df['drug_group'] == group]
        group_data_features = group_data[feature_columns].values
        group_data_labels = group_data['drug'].values
        group_data_diseases = group_data['disease'].values
        # 计算每个类别的目标样本数量
        label_counts = Counter(group_data_labels)
        total_samples = sum(label_counts.values())
        sampling_strategy = {label: int(round(size * (count / total_samples))) for label, count in label_counts.items()}
        # 确保总样本数量符合 group_sizes
        total_resampled = sum(sampling_strategy.values())
        if total_resampled > size:
            excess = total_resampled - size
            while excess > 0:
                for label in sorted(sampling_strategy, key=sampling_strategy.get, reverse=True):
                    if sampling_strategy[label] > 0:
                        sampling_strategy[label] -= 1
                        excess -= 1
                    if excess == 0:
                        break
        print(f"Group: {group}, 原始样本数量: {group_data.shape[0]}")
        print(f"类别标签: {label_counts}")
        X_resampled, y_resampled = KSU(group_data_features, group_data_labels, sampling_strategy)
        print(f"Group: {group}, 欠采样后数据集大小: {X_resampled.shape}")
        # 保留疾病序号
        resampled_indices = []
        for label in label_counts.keys():
            indices = np.where(group_data_labels == label)[0]
            if sampling_strategy[label] < len(indices):
                resampled_indices.extend(np.random.choice(indices, sampling_strategy[label], replace=False))
            else:
                resampled_indices.extend(indices)
        group_data_resampled = pd.DataFrame(X_resampled, columns=feature_columns)
        group_data_resampled['disease'] = group_data_diseases[resampled_indices]  # 保留疾病序号
        group_data_resampled['drug'] = group_data['drug'].iloc[resampled_indices].values  # 保留药物序号
        group_data_resampled['drug_group'] = group  # 保留 drug_group 列
        # 确保药物和疾病序号在最前面
        group_data_resampled = group_data_resampled[['drug', 'disease'] + feature_columns + ['drug_group']]
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

# 保存下采样后的数据
def save_resampled_data(resampled_data):
    try:
        print("开始保存下采样后的数据...")
        resampled_data = resampled_data.drop(columns=['drug_group'])
        resampled_data['drug'] = resampled_data['drug'].astype(int)
        resampled_data['disease'] = resampled_data['disease'].astype(int)
        resampled_data = resampled_data.sort_values(by=['drug', 'disease'], ascending=True)
        resampled_data.to_csv("D:/a毕设/数据/标准化/disKSU_欧几里得.csv", index=False)
        print("数据下采样完成，已保存到文件。")
    except Exception as e:
        print(f"保存文件时发生错误: {e}")

if __name__ == '__main__':
    final_reordered = load_data()
    final_reordered = assign_drug_group(final_reordered)
    # 指定特征列
    feature_columns = [col for col in final_reordered.columns if 'feature' in col]
    try:
        print("开始分组下采样...")
        resampled_data, resampled_labels = group_downsampling_with_KSU(final_reordered, group_sizes, feature_columns)
        print(f"下采样完成，数据形状: {resampled_data.shape}")
        group_counts = resampled_data['drug_group'].value_counts()
        print("采样后的每组样本数量：")
        print(group_counts)
        save_resampled_data(resampled_data)
    except Exception as e:
        print(f"下采样过程中发生错误: {e}")
        raise
