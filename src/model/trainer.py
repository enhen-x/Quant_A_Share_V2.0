# src/model/trainer.py

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
        
        # 自动版本号 (单次运行时使用，WF运行时会覆盖)
        self.version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_dir = os.path.join(self.paths["models"], self.version)

    def load_data(self):
        """加载特征工程后的全量数据，支持特征筛选"""
        data_path = os.path.join(self.paths["data_processed"], "all_stocks.parquet")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"未找到数据: {data_path}")
            
        logger.info(f"正在加载训练数据: {data_path} ...")
        df = read_parquet(data_path)
        
        # === 特征筛选逻辑 ===
        use_feature_selection = self.model_cfg.get("use_feature_selection", False)
        
        if use_feature_selection:
            # 从配置文件加载筛选后的特征
            selected_features_path = os.path.join(
                self.paths["data_processed"], 
                "selected_features.txt"
            )
            
            if os.path.exists(selected_features_path):
                with open(selected_features_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    feature_cols = [line.strip() for line in lines 
                                    if line.strip() and not line.startswith("#")]
                logger.info(f"使用筛选后的特征: {len(feature_cols)} 个")
            else:
                logger.warning(f"特征筛选配置文件不存在: {selected_features_path}")
                logger.warning("回退到使用全部特征")
                feature_cols = [c for c in df.columns if c.startswith("feat_")]
        else:
            # 使用全部特征
            feature_cols = [c for c in df.columns if c.startswith("feat_")]
            logger.info(f"使用全部特征: {len(feature_cols)} 个")
        
        label_col = self.model_cfg["label_col"]
        
        if label_col not in df.columns:
             # 兼容逻辑...
             avail_labels = [c for c in df.columns if c.startswith("label")]
             if avail_labels:
                 label_col = avail_labels[0]
             else:
                 raise ValueError("未找到标签列！")

        df = df.dropna(subset=[label_col])
        return df, feature_cols, label_col

    # === 新增：暴露核心训练逻辑供外部调用 ===
    def train_model(self, X_train, y_train, X_val, y_val):
        """训练单个模型并返回"""
        model = XGBModelWrapper()
        model.train(X_train, y_train, X_val, y_val)
        return model

    def run(self):
        """原有单次训练逻辑 (保持兼容)"""
        # 只有在单次运行时才创建目录
        ensure_dir(self.model_dir)
        
        logger.info(f"=== 开始训练任务 (Single Run: {self.version}) ===")
        df, features, label = self.load_data()
        
        split_date = self.model_cfg.get("train_val_split_date", "2022-01-01")
        # ... (后续原有逻辑保持不变)
        # 为了完整性，简述：
        train_mask = df["date"] <= split_date
        val_mask = df["date"] > split_date
        
        X_train = df.loc[train_mask, features]
        y_train = df.loc[train_mask, label]
        X_val = df.loc[val_mask, features]
        y_val = df.loc[val_mask, label]
        
        model = self.train_model(X_train, y_train, X_val, y_val) # 复用新方法
        
        model.save(os.path.join(self.model_dir, "model.json"))
        
        df["pred_score"] = model.predict(df[features])
        out_cols = ["date", "symbol", "close", label, "pred_score"]
        df[out_cols].to_parquet(os.path.join(self.model_dir, "predictions.parquet"), index=False)
        logger.info(f"单次训练完成。")