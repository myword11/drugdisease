import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
from collections import Counter

group_sizes = [3423] * 5 + [1301]

# OSS下采样方法
def dist_nominal(instance, D):
    d = cdist(instance.reshape(1, -1), D.T, metric='hamming')[0]
    return d

def OSS_resample(data, labels, sampling_strategy):
    unique_labels = np.unique(labels)
    for label in unique_labels:
        if label not in sampling_strategy:
            raise ValueError(f"Class label {label} in data is not present in sampling strategy.")
    data = data.T
    labels = labels.flatten()
    class_types = np.unique(labels)
    if set(sampling_strategy.keys()) != set(class_types):
        raise ValueError('Class labels in sampling strategy do not match data labels.')
    if data.shape[1] != len(labels):
        raise ValueError('Instance numbers in data and labels do not consistent.')
    class_data = {class_type: data[:, labels == class_type] for class_type in class_types}
    class_dist = {class_type: class_data[class_type].shape[1] for class_type in class_types}
    D = []
    DL = []
    for class_type in class_types:
        target_samples = sampling_strategy[class_type]
        if target_samples < class_dist[class_type]:
            K = class_data[class_type]
            indices = np.random.permutation(class_dist[class_type])
            id_to_keep = indices[:target_samples]
            D_temp = K[:, id_to_keep]
            DL_temp = np.array([class_type] * target_samples)
            D.append(D_temp)
            DL.append(DL_temp)
        else:
            D.append(class_data[class_type])
            DL.append(np.array([class_type] * class_dist[class_type]))
    D = np.hstack(D)
    DL = np.hstack(DL)
    final_class_dist = {class_label: np.sum(DL == class_label) for class_label in class_types}
    for class_label in class_types:
        if final_class_dist[class_label] != sampling_strategy[class_label]:
            print(
                f"Warning: Class '{class_label}' has {final_class_dist[class_label]} samples, expected {sampling_strategy[class_label]}.")
    return D.T, DL

# 读取标准化后的数据
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

# 分配药物组
def assign_drug_group(df, total_drugs=269, group_sizes=[3423] * 5 + [1301]):
    drug_indices = np.arange(total_drugs)
    group_boundaries = np.linspace(0, 249, num=6, dtype=int)
    group_boundaries = np.append(group_boundaries, total_drugs)
    df['drug_group'] = pd.cut(df['drug'].astype(int),
                              bins=group_boundaries,
                              labels=[f'Group{i + 1}' for i in range(6)], right=False)
    return df

# 分组下采样
def group_downsampling_with_OSS(df, group_sizes):
    resampled_data = []
    resampled_labels = []
    for group, size in zip(df['drug_group'].unique(), group_sizes):
        print(f"正在处理分组: {group}")
        group_data = df[df['drug_group'] == group]
        feature_columns = [col for col in group_data.columns if 'feature' in col]
        group_data_features = group_data[feature_columns].values
        group_data_labels = group_data['drug'].values
        group_data_diseases = group_data['disease'].values
        unique_labels = np.unique(group_data_labels)
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
        print(f"类别标签: {unique_labels}")
        X_resampled, y_resampled = OSS_resample(group_data_features, group_data_labels, sampling_strategy)
        print(f"Group: {group}, 欠采样后数据集大小: {X_resampled.shape}")
        # 保留疾病序号
        resampled_indices = []
        for label in unique_labels:
            indices = np.where(group_data_labels == label)[0]
            if sampling_strategy[label] < len(indices):
                resampled_indices.extend(np.random.choice(indices, sampling_strategy[label], replace=False))
            else:
                resampled_indices.extend(indices)
        group_data_resampled = pd.DataFrame(X_resampled, columns=feature_columns)
        group_data_resampled['disease'] = group_data_diseases[resampled_indices]  # 保留疾病序号
        group_data_resampled['drug_group'] = group  # 保留 drug_group 列
        group_data_resampled['drug'] = y_resampled  # 保留 drug 列
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
        resampled_data.to_csv("D:/a毕设/数据/未归一化/disAssociation_OSS.csv", index=False)
        print("数据下采样完成，已保存到文件。")
    except Exception as e:
        print(f"保存文件时发生错误: {e}")

if __name__ == '__main__':
    # 读取标准化后的数据
    final_reordered = load_data()
    final_reordered = assign_drug_group(final_reordered)
    try:
        print("开始分组下采样...")
        resampled_data, resampled_labels = group_downsampling_with_OSS(final_reordered, group_sizes)
        print(f"下采样完成，数据形状: {resampled_data.shape}")
        group_counts = resampled_data['drug_group'].value_counts()
        print("采样后的每组样本数量：")
        print(group_counts)
        save_resampled_data(resampled_data)
    except Exception as e:
        print(f"下采样过程中发生错误: {e}")
        raise
