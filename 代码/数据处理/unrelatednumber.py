import pandas as pd
import itertools

# 读取已有的相关药物与疾病配对表格
df_related = pd.read_csv(
    r"D:\a毕设\24.12.26_2101花葛药物-疾病相互作用课题\数据集\B-dataset\DrugDiseaseAssociationNumber.csv")  # 请替换为实际文件路径

# 获取药物和疾病的列表
all_drugs = df_related['drug'].unique()
all_diseases = df_related['disease'].unique()

# 创建药物与疾病的所有可能组合
all_combinations = set(itertools.product(all_drugs, all_diseases))

# 获取已知的相关配对
related_pairs_set = set(tuple(x) for x in df_related.values)

# 计算不相关配对
unrelated_pairs = all_combinations - related_pairs_set

# 将不相关的配对转换为DataFrame
df_unrelated = pd.DataFrame(list(unrelated_pairs), columns=['drug', 'disease'])

# 确保没有重复
df_unrelated = df_unrelated.drop_duplicates().reset_index(drop=True)

# 按照药物和疾病列排序
df_unrelated_sorted = df_unrelated.sort_values(by=['drug', 'disease']).reset_index(drop=True)

# 保存结果
df_unrelated_sorted.to_csv(
    r'D:/a毕设/24.12.26_2101花葛药物-疾病相互作用课题/数据集/B-dataset/disAssociationNumber.csv',
    index=False)

# 输出前几行以确认
print(df_unrelated_sorted.head())
