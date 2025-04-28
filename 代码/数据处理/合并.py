import pandas as pd

# 读取药物-疾病相关配对和不相关配对的表格
df_related = pd.read_csv(r"D:/a毕设/24.12.26_2101花葛药物-疾病相互作用课题/数据集/B-dataset/未归一化/Association.csv")
df_unrelated = pd.read_csv(r"D:/a毕设/24.12.26_2101花葛药物-疾病相互作用课题/数据集/B-dataset/未归一化/disAssociation.csv")

# 为相关和不相关配对设置标签
df_related['label'] = 1  # 相关配对的标签为1
df_unrelated['label'] = 0  # 不相关配对的标签为0

# 去掉列名中的前后空格，避免因空格导致匹配失败
df_related.columns = df_related.columns.str.strip()
df_unrelated.columns = df_unrelated.columns.str.strip()

# 输出列名检查是否有空格或不一致
print("Columns in df_related:", df_related.columns)
print("Columns in df_unrelated:", df_unrelated.columns)

# 提取药物和疾病特征的列名
drug_columns = [col for col in df_related.columns if col.endswith('_x')]
disease_columns = [col for col in df_related.columns if col.endswith('_y')]

# 输出检查提取的药物和疾病特征列
print("Drug columns:", drug_columns)
print("Disease columns:", disease_columns)

# 合并相关和不相关的数据集
df = pd.concat([df_related, df_unrelated], ignore_index=True)

# 包括药物和疾病特征列，确保这些列存在
df = df[['drug', 'disease', 'label'] + drug_columns + disease_columns]

# 输出合并后的数据前几行进行检查
print(df.head())

# 输出合并后的列名，确认所有药物和疾病特征列是否存在
print("Columns in merged data:", df.columns)

# 将合并后的表格保存为 CSV 文件
output_file = r"D:/a毕设/数据/未归一化/合并/未下采样.csv"
df.to_csv(output_file, index=False)

print("Merged data saved to", output_file)

