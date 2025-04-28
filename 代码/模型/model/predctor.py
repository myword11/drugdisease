import sys
import os
import numpy as np
import pandas as pd
from rdkit import Chem
from gensim.models import word2vec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QLabel, QPushButton, QFileDialog,
                             QTableWidget, QTableWidgetItem, QHeaderView,
                             QMessageBox, QProgressBar)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor
import torch
from tqdm import tqdm
from features import MolSentence, mol2alt_sentence  # 假设有这些自定义模块
# 导入自定义模型类和特征选择器
from decision_tree_model import DecisionTreeModel
from random_forest_model import RandomForestModel
from xgboost_model import XGBoostModel
from feature_selector import FeatureSelector

# 配置常量
MODEL_PATH = "model_300dim.pkl"
word2vec_model = word2vec.Word2Vec.load(MODEL_PATH)
RADIUS = 1
UNCOMMON = "UNK"

class PredictionThread(QThread):
    """用于后台预测的线程"""
    finished = pyqtSignal(object, object)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)

    def __init__(self, predictor, features, drug_ids, disease_ids):
        super().__init__()
        self.predictor = predictor
        self.features = features
        self.drug_ids = drug_ids
        self.disease_ids = disease_ids

    def run(self):
        try:
            results = []
            total = len(self.drug_ids)

            # 分批处理避免内存问题
            batch_size = 100
            for i in range(0, total, batch_size):
                batch_features = self.features[i:i + batch_size]
                probs, msg = self.predictor.predict_proba(batch_features)

                if probs is None:
                    self.error.emit(f"预测失败: {msg}")
                    return

                for j in range(len(probs)):
                    idx = i + j
                    results.append({
                        'drug_id': self.drug_ids[idx],
                        'disease_id': self.disease_ids[idx],
                        'probability': probs[j],
                        'label': 'Positive' if probs[j] > 0.5 else 'Negative'
                    })

                self.progress.emit(min(i + batch_size, total))

            self.finished.emit(results, None)
        except Exception as e:
            self.error.emit(f"预测线程错误: {str(e)}")

class EnsemblePredictor:
    def __init__(self, model_path=None, feature_selector_params=None):
        self.models = {}  # 每个模型都应包含 {'model': model_obj}
        self.feature_selector_params = feature_selector_params or {}  # 存储每个模型的特征选择参数
        if model_path:
            self.load_models(model_path)

    def load_models(self, model_path):
        """加载预训练模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        try:
            model_dict = joblib.load(model_path)
            if not isinstance(model_dict, dict):
                raise ValueError("模型文件应包含字典结构")
            self.models = model_dict

            # 打印加载情况
            print("加载的模型字典: ", self.models)

            for model_name, content in self.models.items():
                print(f"正在加载模型 {model_name} ...")
                model = content['model']

                # 根据模型名称选择对应的特征选择参数
                feature_selector = self.feature_selector_params.get(model_name, None)
                if feature_selector:
                    print(f"为模型 {model_name} 设置特征选择器，选择了 {feature_selector} 个特征")

                # 确保模型类型正确
                if not isinstance(model, (DecisionTreeModel, RandomForestModel, XGBoostModel)):
                    raise ValueError(f"模型 {model_name} 类型错误: {type(model)}")

                print(f"模型 {model_name} 加载完成")

        except Exception as e:
            raise ValueError(f"加载模型失败: {str(e)}")

    def predict_proba(self, X_new):
        """应用特征选择器并返回预测概率"""
        if not self.models:
            return None, "未加载任何模型"

        all_probs = []
        error_msgs = []

        for model_name, content in self.models.items():
            model = content['model']

            # 如果没有特征选择器，则跳过
            feature_selector = self.feature_selector_params.get(model_name, None)
            if feature_selector:
                print(f"应用特征选择器 ({model_name})，选择 {feature_selector} 个特征")
                X_model = X_new.iloc[:, :feature_selector]  # 选择前 'feature_selector' 个特征
            else:
                X_model = X_new

            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_model)
                    y_prob = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
                elif hasattr(model, 'predict'):
                    y_pred = model.predict(X_model)
                    y_prob = self._normalize_predictions(y_pred)
                else:
                    raise ValueError(f"模型 {model_name} 不支持预测")

                all_probs.append(y_prob)
            except Exception as e:
                error_msgs.append(f"{model_name} 预测失败: {str(e)}")

        if not all_probs:
            return None, "所有模型预测失败:\n" + "\n".join(error_msgs)

        return np.mean(all_probs, axis=0), ("\n".join(error_msgs) if error_msgs else None)


class DrugDiseasePredictionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("药物-疾病关联预测系统")
        self.setGeometry(100, 100, 1200, 800)
        self.test_data_path = ""
        self.ensemble_model_path = ""
        self.predictor = None
        self.word2vec_model = None
        self.current_results = None

        # 定义每个模型的特征选择参数（训练时选择的特征数量）
        self.feature_selector_params = {
            'DecisionTree': 140,  # DecisionTree 使用前 140 个特征
            'RandomForest': 90,  # RandomForest 使用前 90 个特征
            'XGBoost': 160,  # XGBoost 使用前 160 个特征
        }

        self.init_ui()
        self.load_word2vec_model()


    def init_ui(self):
        """初始化用户界面"""
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # 主布局
        main_layout = QVBoxLayout()
        self.central_widget.setLayout(main_layout)

        # 控制面板
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        main_layout.addWidget(self.progress_bar)

        # 结果表格
        self.results_table = self.create_results_table()
        main_layout.addWidget(self.results_table)

        # 状态栏
        self.status_label = QLabel("准备就绪")
        self.status_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.status_label)

        # 样式设置
        self.setStyleSheet(""" 
            QPushButton { padding: 8px; font-size: 14px; min-width: 120px; }
            QLabel { font-size: 14px; }
            QProgressBar { height: 20px; }
        """)

    def create_control_panel(self):
        """创建控制面板"""
        panel = QWidget()
        layout = QHBoxLayout(panel)

        # 模型选择
        self.model_btn = QPushButton("选择模型")
        self.model_btn.clicked.connect(self.select_model)
        self.model_label = QLabel("未选择模型")
        layout.addWidget(self.model_btn)
        layout.addWidget(self.model_label)

        # 数据选择
        self.data_btn = QPushButton("选择数据")
        self.data_btn.clicked.connect(self.select_data)
        self.data_label = QLabel("未选择数据")
        layout.addWidget(self.data_btn)
        layout.addWidget(self.data_label)

        # 预测按钮
        self.predict_btn = QPushButton("开始预测")
        self.predict_btn.setStyleSheet("background-color: #4CAF50; color: white;")
        self.predict_btn.clicked.connect(self.start_prediction)
        self.predict_btn.setEnabled(False)
        layout.addWidget(self.predict_btn)

        # 添加保存按钮
        self.save_btn = QPushButton("保存结果")
        self.save_btn.setStyleSheet("background-color: #2196F3; color: white;")
        self.save_btn.clicked.connect(self.save_results)
        self.save_btn.setEnabled(False)
        layout.addWidget(self.save_btn)

        return panel

    def create_results_table(self):
        """创建结果表格"""
        table = QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(['Drug ID', 'Disease ID', 'Probability', 'Label'])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        return table

    def load_word2vec_model(self):
        """加载预训练的Word2Vec模型"""
        try:
            self.word2vec_model = word2vec.Word2Vec.load("model_300dim.pkl")
            print("Word2Vec模型加载成功")
        except Exception as e:
            print(f"加载Word2Vec模型失败: {str(e)}")

    def select_model(self):
        """选择模型文件"""
        model_file, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "Pickle Files (*.pkl)")
        if model_file:
            self.ensemble_model_path = model_file
            self.model_label.setText(os.path.basename(model_file))
            self.load_ensemble_model(model_file)

    def load_ensemble_model(self, model_path):
        """加载集成模型"""
        try:
            self.predictor = EnsemblePredictor(model_path, self.feature_selector_params)
            self.status_label.setText("模型加载成功，准备预测")
            self.predict_btn.setEnabled(True)
        except Exception as e:
            self.status_label.setText(f"模型加载失败: {str(e)}")
            self.predict_btn.setEnabled(False)

    def select_data(self):
        """选择测试数据文件"""
        data_file, _ = QFileDialog.getOpenFileName(self, "选择测试数据", "", "CSV Files (*.csv)")
        if data_file:
            self.test_data_path = data_file
            self.data_label.setText(os.path.basename(data_file))

    def start_prediction(self):
        """开始预测"""
        if not self.test_data_path or not self.predictor:
            QMessageBox.warning(self, "警告", "请选择数据和模型文件")
            return

        try:
            # 加载测试数据
            test_data = pd.read_csv(self.test_data_path)
            drug_smiles = test_data['smiles'].values  # SMILES 字符串
            disease_ids = test_data['disease'].values
            drug_ids = test_data['drug'].values

            # 特征提取
            X_new_feats = []
            invalid_count = 0
            for smiles in drug_smiles:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    try:
                        sentence = mol2alt_sentence(mol, radius=RADIUS)
                        vector = np.mean([self.word2vec_model.wv[token] if token in self.word2vec_model.wv else
                                          self.word2vec_model.wv[UNCOMMON]
                                          for token in sentence], axis=0)
                        X_new_feats.append(vector)
                    except Exception as e:
                        print(f"分子处理失败: {smiles} 错误: {str(e)}")
                        X_new_feats.append(np.zeros(self.word2vec_model.vector_size))
                        invalid_count += 1
                else:
                    print(f"无效的 SMILES: {smiles}")
                    X_new_feats.append(np.zeros(self.word2vec_model.vector_size))
                    invalid_count += 1

            print(f"无效或处理失败的分子数量: {invalid_count}")

            # 加载疾病特征列（假设是前64列）
            disease_feats = test_data.iloc[:, -64:].values  # 获取疾病特征

            # 转换药物特征为 DataFrame
            X_new_feats = pd.DataFrame(X_new_feats)

            # 选择药物特征的特征选择
            feature_selector = self.feature_selector_params.get('DecisionTree', None)  # 这里可以修改为动态选择模型
            if feature_selector:
                # 使用特征选择器对药物特征进行选择
                X_new_feats = X_new_feats.iloc[:, :feature_selector]  # 仅选择前 'feature_selector' 个药物特征

            # 拼接药物特征和疾病特征
            X_all_feats = pd.concat([X_new_feats, pd.DataFrame(disease_feats)], axis=1)
            #
        except Exception as e:
            QMessageBox.warning(self, "错误", f"加载数据失败: {str(e)}")
            return

        # 创建预测线程并启动
        self.predict_thread = PredictionThread(self.predictor, X_all_feats, drug_ids, disease_ids)
        self.predict_thread.finished.connect(self.on_prediction_finished)
        self.predict_thread.error.connect(self.on_prediction_error)
        self.predict_thread.progress.connect(self.on_prediction_progress)
        self.progress_bar.setVisible(True)
        self.predict_thread.start()

    def save_results(self):
        """保存预测结果到文件"""
        if not self.current_results:
            QMessageBox.warning(self, "警告", "没有可保存的结果")
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self, "保存结果", "", "CSV Files (*.csv);;Excel Files (*.xlsx)")

        if not save_path:
            return

        try:
            # 将结果转换为DataFrame
            df = pd.DataFrame(self.current_results)

            # 根据文件扩展名选择保存格式
            if save_path.endswith('.csv'):
                df.to_csv(save_path, index=False)
            elif save_path.endswith('.xlsx'):
                df.to_excel(save_path, index=False)
            else:
                df.to_csv(save_path + '.csv', index=False)

            QMessageBox.information(self, "成功", f"结果已保存到: {save_path}")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"保存失败: {str(e)}")

    def on_prediction_finished(self, results, error):
        """预测完成后更新结果"""
        self.progress_bar.setVisible(False)
        if error:
            QMessageBox.warning(self, "错误", error)
        else:
            self.update_results_table(results)

    def on_prediction_error(self, error_msg):
        """预测错误处理"""
        self.progress_bar.setVisible(False)
        QMessageBox.warning(self, "错误", error_msg)

    def on_prediction_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)

    def update_results_table(self, results):
        """更新结果表格"""
        self.current_results = results  # 保存结果数据
        self.results_table.setRowCount(len(results))
        for row, result in enumerate(results):
            self.results_table.setItem(row, 0, QTableWidgetItem(str(result['drug_id'])))
            self.results_table.setItem(row, 1, QTableWidgetItem(str(result['disease_id'])))
            self.results_table.setItem(row, 2, QTableWidgetItem(f"{result['probability']:.4f}"))
            self.results_table.setItem(row, 3, QTableWidgetItem(result['label']))
        self.save_btn.setEnabled(True)  # 启用保存按钮


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DrugDiseasePredictionApp()
    window.show()
    sys.exit(app.exec_())
