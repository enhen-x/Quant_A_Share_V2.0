# src/model/trainer.py

import pandas as pd
import numpy as np
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
        
        # 双头模型配置
        self.dual_head_cfg = self.model_cfg.get("dual_head", {})
        self.dual_head_enabled = self.dual_head_cfg.get("enable", False)
        
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
             # 兼容逻辑
             avail_labels = [c for c in df.columns if c.startswith("label")]
             if avail_labels:
                 label_col = avail_labels[0]
             else:
                 raise ValueError("未找到标签列！")

        df = df.dropna(subset=[label_col])
        return df, feature_cols, label_col

    # === 单模型训练 (保持向后兼容) ===
    def train_model(self, X_train, y_train, X_val, y_val):
        """训练单个 XGBoost 模型并返回"""
        model = XGBModelWrapper()
        model.train(X_train, y_train, X_val, y_val)
        return model
    
    # === 双头模型训练 ===
    def train_dual_head(self, X_train, y_train_reg, y_train_cls, X_val, y_val_reg, y_val_cls, feature_names=None):
        """
        训练双头模型 (回归 + 分类)
        
        Returns:
            (reg_model, cls_model): 回归模型和分类模型
        """
        from src.model.lgb_model import LGBModelWrapper
        
        reg_model = None
        cls_model = None
        
        # 训练回归头
        reg_cfg = self.dual_head_cfg.get("regression", {})
        if reg_cfg.get("enable", True):
            logger.info("=" * 40)
            logger.info(">>> 训练回归头 (Regression Head)")
            logger.info("=" * 40)
            reg_model = LGBModelWrapper(task_type="regression")
            reg_model.train(X_train, y_train_reg, X_val, y_val_reg, feature_names=feature_names)
        
        # 训练分类头
        cls_cfg = self.dual_head_cfg.get("classification", {})
        if cls_cfg.get("enable", True):
            logger.info("=" * 40)
            logger.info(">>> 训练分类头 (Classification Head)")
            logger.info("=" * 40)
            cls_model = LGBModelWrapper(task_type="classification")
            cls_model.train(X_train, y_train_cls, X_val, y_val_cls, feature_names=feature_names)
        
        return reg_model, cls_model
    
    def fuse_predictions(self, pred_reg: np.ndarray, pred_cls: np.ndarray) -> np.ndarray:
        """
        融合回归和分类预测
        
        Args:
            pred_reg: 回归模型预测值
            pred_cls: 分类模型预测概率
            
        Returns:
            融合后的预测分数
        """
        fusion_cfg = self.dual_head_cfg.get("fusion", {})
        method = fusion_cfg.get("method", "weighted_average")
        normalize = fusion_cfg.get("normalize", True)
        
        reg_weight = self.dual_head_cfg.get("regression", {}).get("weight", 0.6)
        cls_weight = self.dual_head_cfg.get("classification", {}).get("weight", 0.4)
        
        # 归一化
        if normalize:
            def min_max_normalize(arr):
                arr = np.array(arr)
                min_val, max_val = arr.min(), arr.max()
                if max_val - min_val > 1e-8:
                    return (arr - min_val) / (max_val - min_val)
                return np.zeros_like(arr)
            
            if pred_reg is not None:
                pred_reg = min_max_normalize(pred_reg)
            if pred_cls is not None:
                pred_cls = min_max_normalize(pred_cls)
        
        # 融合
        if method == "weighted_average":
            if pred_reg is not None and pred_cls is not None:
                fused = reg_weight * pred_reg + cls_weight * pred_cls
            elif pred_reg is not None:
                fused = pred_reg
            else:
                fused = pred_cls
        elif method == "multiplicative":
            if pred_reg is not None and pred_cls is not None:
                fused = pred_reg * pred_cls
            elif pred_reg is not None:
                fused = pred_reg
            else:
                fused = pred_cls
        else:
            fused = pred_reg if pred_reg is not None else pred_cls
        
        return fused

    def run(self):
        """主训练逻辑，支持单模型和双头模型"""
        ensure_dir(self.model_dir)
        
        logger.info(f"=== 开始训练任务 (Single Run: {self.version}) ===")
        df, features, label = self.load_data()
        
        split_date = self.model_cfg.get("train_val_split_date", "2022-01-01")
        df["date"] = pd.to_datetime(df["date"])
        train_mask = df["date"] <= split_date
        val_mask = df["date"] > split_date
        
        X_train = df.loc[train_mask, features]
        X_val = df.loc[val_mask, features]
        
        if self.dual_head_enabled:
            # ==========================================
            # 双头模型训练
            # ==========================================
            logger.info("=" * 50)
            logger.info(">>> 双头模型训练模式 (Dual-Head)")
            logger.info("=" * 50)
            
            y_train_reg = df.loc[train_mask, label]
            y_val_reg = df.loc[val_mask, label]
            
            # 检查分类标签
            label_cls = "label_cls"
            if label_cls not in df.columns:
                logger.error(f"未找到分类标签列 '{label_cls}'，请确保启用双头模型后重新运行 pipeline")
                raise ValueError(f"缺少分类标签列 '{label_cls}'")
            
            y_train_cls = df.loc[train_mask, label_cls]
            y_val_cls = df.loc[val_mask, label_cls]
            
            # 训练双头模型
            reg_model, cls_model = self.train_dual_head(
                X_train, y_train_reg, y_train_cls,
                X_val, y_val_reg, y_val_cls,
                feature_names=features
            )
            
            # 保存模型
            if reg_model is not None:
                reg_model.save(os.path.join(self.model_dir, "model_reg.joblib"))
            if cls_model is not None:
                cls_model.save(os.path.join(self.model_dir, "model_cls.joblib"))
            
            # 生成预测
            pred_reg = reg_model.predict(df[features]) if reg_model else None
            pred_cls = cls_model.predict(df[features]) if cls_model else None
            
            # 融合预测
            df["pred_reg"] = pred_reg
            df["pred_cls"] = pred_cls
            df["pred_score"] = self.fuse_predictions(pred_reg, pred_cls)
            
            # 保存结果
            out_cols = ["date", "symbol", "close", label, label_cls, "pred_reg", "pred_cls", "pred_score"]
            out_cols = [c for c in out_cols if c in df.columns]
            df[out_cols].to_parquet(os.path.join(self.model_dir, "predictions.parquet"), index=False)
            
            logger.info(f"双头模型训练完成。")
            logger.info(f"回归权重: {self.dual_head_cfg.get('regression', {}).get('weight', 0.6)}")
            logger.info(f"分类权重: {self.dual_head_cfg.get('classification', {}).get('weight', 0.4)}")
            
        else:
            # ==========================================
            # 单模型训练 (原有逻辑)
            # ==========================================
            y_train = df.loc[train_mask, label]
            y_val = df.loc[val_mask, label]
            
            model = self.train_model(X_train, y_train, X_val, y_val)
            model.save(os.path.join(self.model_dir, "model.json"))
            
            df["pred_score"] = model.predict(df[features])
            out_cols = ["date", "symbol", "close", label, "pred_score"]
            df[out_cols].to_parquet(os.path.join(self.model_dir, "predictions.parquet"), index=False)
            
            logger.info(f"单次训练完成。")
