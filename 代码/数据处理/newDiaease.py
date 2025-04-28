import pandas as pd

# 读取文件
mesh_file = "D:/a毕设/24.12.26_2101花葛药物-疾病相互作用课题/数据集/B-dataset/MeSHFeatureGeneratedByDeepWalk.csv"
disease_file = "D:/a毕设/24.12.26_2101花葛药物-疾病相互作用课题/数据集/B-dataset/DiseaseFeature.csv"
# 读取文件
mesh_data = pd.read_csv(mesh_file, header=None)  # 无标题，添加默认标题
disease_data = pd.read_csv(disease_file)

# 重命名 MeSH 文件列
mesh_data.rename(columns={0: 'Disease'}, inplace=True)  # 第一列重命名为 Disease
mesh_data.columns = ['Disease'] + [str(i) for i in range(1, mesh_data.shape[1])]

# 重命名 Disease 文件列
disease_data.rename(columns={disease_data.columns[1]: 'Disease'}, inplace=True)

# 统一疾病名称为小写
mesh_data['Disease'] = mesh_data['Disease'].str.lower()
disease_data['Disease'] = disease_data['Disease'].str.lower()

# 按照 DiseaseFeature.csv 中的顺序筛选 MeSH 数据
filtered_data = mesh_data[mesh_data['Disease'].isin(disease_data['Disease'])]

# 确保结果按照 DiseaseFeature.csv 的疾病顺序排列
filtered_data = filtered_data.set_index('Disease').reindex(disease_data['Disease']).reset_index()

# 添加序号列
filtered_data.insert(0, 'Index', range(1, len(filtered_data) + 1))

# 保存结果
output_file = "/mnt/data/FilteredMeSHFeatures.csv"
filtered_data.to_csv(output_file, index=False)

print(f"处理完成，结果已保存到 {output_file}")