import pandas as pd
import os
import datetime
import joblib
from src.utils.config import GLOBAL_CONFIG
from src.utils.io import read_parquet, ensure_dir
from src.utils.logger import get_logger
from src.model.xgb_model import XGBModelWrapper

logger = get_logger()

class ModelTrainer:
    def __init__(self):
        self.config = GLOBAL_CONFIG
        self.paths = self.config["paths"]
        self.model_cfg = self.config["model"]
        
        # 自动版本号管理 (使用当前时间)
        self.version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_dir = os.path.join(self.paths["models"], self.version)
        ensure_dir(self.model_dir)

    def load_data(self):
        """加载特征工程后的全量数据"""
        data_path = os.path.join(self.paths["data_processed"], "all_stocks.parquet")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"未找到数据: {data_path}，请先运行 rebuild_features.py")
            
        logger.info(f"正在加载训练数据: {data_path} ...")
        df = read_parquet(data_path)
        
        # 1. 提取特征列和标签列
        # 自动识别 feat_ 开头的列
        feature_cols = [c for c in df.columns if c.startswith("feat_")]
        label_col = self.model_cfg["label_col"]  # 从 config 读取，例如 "label"
        
        if label_col not in df.columns:
            # 尝试找 label_5d 这种
            avail_labels = [c for c in df.columns if c.startswith("label")]
            if avail_labels:
                label_col = avail_labels[0]
                logger.warning(f"配置的 {self.model_cfg['label_col']} 不存在，自动切换为 {label_col}")
            else:
                raise ValueError("未找到标签列！")

        # 2. 剔除标签为空的行 (无法训练)
        df = df.dropna(subset=[label_col])
        
        logger.info(f"数据加载完毕: {df.shape}, 特征数: {len(feature_cols)}")
        return df, feature_cols, label_col

    def run(self):
        logger.info(f"=== 开始训练任务 (Version: {self.version}) ===")
        
        df, features, label = self.load_data()
        
        # === 核心：时间序列切分 ===
        # 读取 config 中的切分日期，例如 "2018-12-31"
        split_date = self.model_cfg.get("train_val_split_date", "2022-01-01")
        
        logger.info(f"划分数据集: 训练集 <= {split_date}, 验证集 > {split_date}")
        
        train_mask = df["date"] <= split_date
        val_mask = df["date"] > split_date
        
        X_train = df.loc[train_mask, features]
        y_train = df.loc[train_mask, label]
        
        X_val = df.loc[val_mask, features]
        y_val = df.loc[val_mask, label]
        
        logger.info(f"训练集大小: {len(X_train)}, 验证集大小: {len(X_val)}")
        
        # === 训练 ===
        model = XGBModelWrapper()
        model.train(X_train, y_train, X_val, y_val)
        
        # === 保存产物 ===
        # 1. 保存模型文件
        model_path = os.path.join(self.model_dir, "model.json") # xgboost 2.0+ 推荐 json/ubjson
        model.save(model_path)
        logger.info(f"模型已保存: {model_path}")
        
        # 2. 保存预测结果 (生成 predictions.parquet 供回测使用)
        # 这里我们对全量数据（Train+Val）都生成预测，方便看历史拟合情况
        logger.info("正在生成全量预测结果...")
        df["pred_score"] = model.predict(df[features])
        
        pred_save_path = os.path.join(self.model_dir, "predictions.parquet")
        # 只保留核心字段以节省空间
        out_cols = ["date", "symbol", "close", label, "pred_score"]
        if "label" not in out_cols and label != "label": # 确保真实标签也在
             out_cols.append(label)
             
        df[out_cols].to_parquet(pred_save_path, index=False)
        logger.info(f"预测结果已保存: {pred_save_path}")
        
        # 3. (可选) 保存特征重要性
        # ...