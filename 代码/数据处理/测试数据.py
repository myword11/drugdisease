import pandas as pd

# 加载两个表格
table1 = pd.read_csv('D:/a毕设/24.12.26_2101花葛药物-疾病相互作用课题/数据集/B-dataset/未归一化/disAssociation.csv')  # 主表格
table2 = pd.read_csv('D:/a毕设/数据/未归一化/disKSU_汉明.csv')  # 需要删除的数据表格

# 根据多列进行匹配
key_columns = ['drug', 'disease']
merged = table1.merge(table2, on=key_columns, how='left', indicator=True)
table1_filtered = table1[merged['_merge'] == 'left_only']

# 按照疾病序号从小到大排列
table1_filtered = table1_filtered.sort_values(by='disease', ascending=True)

# 统计疾病序号个数最多的前三个
disease_counts = table1_filtered['disease'].value_counts().head(3)
print("疾病序号个数最多的前三个：")
print(disease_counts)

# 保存结果
output_path = 'D:/a毕设/数据/未归一化/验证/验证.csv'
table1_filtered.to_csv(output_path, index=False)
print(f"数据删除完成，结果已保存到 {output_path}")

# 将疾病序号个数最多的前三个数据单独保存为三个表格
for i, (disease, count) in enumerate(disease_counts.items(), start=1):
    disease_data = table1_filtered[table1_filtered['disease'] == disease]
    disease_output_path = f'D:/a毕设/数据/未归一化/验证/验证_top{i}_disease_{disease}.csv'
    disease_data.to_csv(disease_output_path, index=False)
    print(f"疾病序号 {disease} 的数据已保存到 {disease_output_path}")

# 将疾病序号为425和214的数据分别保存到两个单独的表中
disease_ids = [425, 214]
for disease_id in disease_ids:
    disease_data = table1_filtered[table1_filtered['disease'] == disease_id]
    disease_output_path = f'D:/a毕设/数据/未归一化/验证/验证_disease_{disease_id}.csv'
    disease_data.to_csv(disease_output_path, index=False)
    print(f"疾病序号 {disease_id} 的数据已保存到 {disease_output_path}")


