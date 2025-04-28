import pandas as pd

# 加载数据集
disease_features_df = pd.read_csv("D:/a毕设/24.12.26_2101花葛药物-疾病相互作用课题/数据集/B-dataset/NEWDiseaseFeature.csv", header=None)
drug_features_df = pd.read_csv("D:/a毕设/数据/特征提取/Drugmol2vec.csv", header=None)
association_numbers_df = pd.read_csv(r"D:\a毕设\24.12.26_2101花葛药物-疾病相互作用课题\数据集\B-dataset\DrugDiseaseAssociationNumber.csv")


# 为数据集添加列名
drug_features_df.columns = ['index'] + [f'feature_{i}' for i in range(1, drug_features_df.shape[1])]
disease_features_df.columns = ['index', 'disease'] + [f'feature_{i}' for i in range(1, disease_features_df.shape[1]-1)]
drug_features_df.columns = ['index'] + [f'feature_{i}_x' for i in range(1, drug_features_df.shape[1])]  # 给药物特征列名加上 _x 后缀
disease_features_df.columns = ['index', 'disease'] + [f'feature_{i}_y' for i in range(1, disease_features_df.shape[1]-1)]

# 检查数据内容，确保疾病名称与药物名称一致
print("Disease features head:")
print(disease_features_df.head())

print("Drug features head:")
print(drug_features_df.head())

print("Association numbers head:")
print(association_numbers_df.head())

# 清理数据：统一大小写并去除空格
association_numbers_df["drug"] = association_numbers_df["drug"].astype(str).str.strip().str.lower()
association_numbers_df["disease"] = association_numbers_df["disease"].astype(str).str.strip().str.lower()
drug_features_df["drug"] = drug_features_df["index"].astype(str).str.strip().str.lower()
disease_features_df["disease"] = disease_features_df["index"].astype(str).str.strip().str.lower()

# 检查清理后的数据
print("Cleaned Disease features:")
print(disease_features_df.head())

print("Cleaned Drug features:")
print(drug_features_df.head())

# 合并药物与疾病特征数据
merged_disease = association_numbers_df.merge(drug_features_df, on="drug", how="left")
print("After merging with disease features:")
print(merged_disease.head())  # 检查合并后的数据，确认疾病特征是否合并

# 按药物合并
final_merged = merged_disease.merge(disease_features_df, on="disease", how="left")
print("After merging with drug features:")
print(final_merged.head())  # 检查合并后的数据，确认药物特征是否合并

# 提取列并保存，确保特征列都存在
reordered_columns = ['drug', 'disease'] + [col for col in final_merged.columns if 'feature' in col]
final_reordered = final_merged[reordered_columns]

# 保存最终结果
final_reordered.to_csv("D:/a毕设/24.12.26_2101花葛药物-疾病相互作用课题/数据集/B-dataset/未归一化/Association.csv", index=False)
