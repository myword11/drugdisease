import joblib

model_path = 'D:/a毕设/数据/未归一化/集成模型(ksu)/模型/saved_models/ensemble_model.pkl'
model_dict = joblib.load(model_path)

for model_name, content in model_dict.items():
    print(f"模型: {model_name}")
    print(f"  类型: {type(content['model'])}")
    print(f"  特征选择器类型: {type(content['selector'])}")
    print(f"  特征索引: {content['selector'].feature_config.get(model_name)}")
