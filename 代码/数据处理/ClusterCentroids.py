from imblearn.under_sampling import ClusterCentroids
from sklearn.cluster import MiniBatchKMeans
import pandas as pd

# 加载数据
final_merged = pd.read_csv("D:/a毕设/24.12.26_2101花葛药物-疾病相互作用课题/数据集/B-dataset/disAssociation归一化.csv")

# 获取药物序号的唯一值
drug_list = final_merged['drug'].unique()

# 定义目标数据量
target_data_per_group = 3425  # 每组目标大小
last_group_data = 1291  # 最后一组目标大小

final_resampled_data = []

# 分组处理药物
for i in range(0, len(drug_list), 50):
    current_drugs = drug_list[i:i + 50]

    # 获取当前组的数据
    group_data = final_merged[final_merged['drug'].isin(current_drugs)]

    # 获取药物和疾病的映射关系，确保每个药物对应多个疾病
    drug_disease_map = group_data[['drug', 'disease']].drop_duplicates().set_index('drug')['disease'].to_dict()

    # 打印映射关系，确保每个药物对应多个疾病
    print(f"Drug-Disease Mapping for group {i//50 + 1}:")
    print(drug_disease_map)

    # 如果是最后一组并且数据不足，调整为1291条数据
    if len(group_data) < last_group_data and i + 50 >= len(drug_list):
        resampled_data = group_data.sample(n=last_group_data, random_state=42)
    else:
        if len(group_data) > target_data_per_group:
            # 使用 ClusterCentroids 进行下采样
            X = group_data.drop(columns=['drug', 'disease'])  # 只保留特征数据
            y = group_data['drug']  # 将药物列作为标签

            # 初始化 ClusterCentroids（使用 MiniBatchKMeans）
            under = ClusterCentroids(
                sampling_strategy='auto',  # 尽可能平衡每个类
                random_state=1,
                voting='hard',  # 确保投票方式为 'hard'
                estimator=MiniBatchKMeans(n_init=10, random_state=1, batch_size=2048)
            )

            # 执行下采样
            x_resampled, y_resampled = under.fit_resample(X, y)

            # 创建新的 DataFrame 来存储下采样后的数据
            resampled_data = pd.DataFrame(x_resampled, columns=X.columns)

            # 合并所有列一次性
            resampled_data = pd.concat([resampled_data, pd.Series(y_resampled, name='drug')], axis=1)

            # 使用药物映射来获取疾病数据并合并
            resampled_data['disease'] = resampled_data['drug'].map(drug_disease_map)

            # 如果映射后的疾病列有缺失值，填充为默认疾病
            resampled_data['disease'].fillna('default_disease', inplace=True)

            # 限制每组数据的数量为目标大小（3425条）
            resampled_data = resampled_data.sample(n=target_data_per_group, random_state=42)
        else:
            # 如果当前组的数据不足以满足目标大小，直接使用全部数据
            resampled_data = group_data

    # 如果是最后一组，确保它只有1291条数据
    if i + 50 >= len(drug_list):
        resampled_data = resampled_data.sample(n=last_group_data, random_state=42)

    final_resampled_data.append(resampled_data)

# 合并所有组的数据
final_resampled_data = pd.concat(final_resampled_data, axis=0)

# 按照药物序号升序排列数据
final_resampled_data = final_resampled_data.sort_values(by='drug', ascending=True)

# 确保 DataFrame 没有碎片化
final_resampled_data = final_resampled_data.copy()

# 保存最终结果
final_resampled_data.to_csv("D:/a毕设/数据/dis/下采样/DisAssociation_ClusterCentroids.csv", index=False)

print("Data downsampling completed using ClusterCentroids. Final data saved.")
