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
    def train_dual_head(self, X_train, y_train_return, y_train_risk, X_val, y_val_return, y_val_risk, feature_names=None):
        """
        训练双头模型 (收益预测 + 风险预测)
        
        Returns:
            (return_model, risk_model): 收益模型和风险模型
        """
        from src.model.lgb_model import LGBModelWrapper
        from src.model.xgb_model import XGBModelWrapper
        
        # 获取统一的模型类型
        model_type = self.dual_head_cfg.get("model_type", "xgboost")
        
        # 获取收益头和风险头的参数
        params_return = self.dual_head_cfg.get("params_return", {})
        params_risk = self.dual_head_cfg.get("params_risk", {})
        
        return_model = None
        risk_model = None
        
        # 训练收益预测头
        return_cfg = self.dual_head_cfg.get("return_head", {})
        if return_cfg.get("enable", True):
            logger.info("=" * 40)
            logger.info(f">>> 训练收益预测头 (Return Head | {model_type.upper()})")
            logger.info(f"    参数来源: dual_head.params_return")
            logger.info("=" * 40)
            
            if model_type == "lightgbm":
                return_model = LGBModelWrapper(task_type="regression", custom_params=params_return)
                return_model.train(X_train, y_train_return, X_val, y_val_return, feature_names=feature_names)
            else:  # xgboost
                return_model = XGBModelWrapper(task_type="regression", custom_params=params_return)
                # XGBoost不需要feature_names参数
                return_model.train(X_train, y_train_return, X_val, y_val_return)
        
        # 训练风险预测头
        risk_cfg = self.dual_head_cfg.get("risk_head", {})
        if risk_cfg.get("enable", True):
            logger.info("=" * 40)
            logger.info(f">>> 训练风险预测头 (Risk Head | {model_type.upper()})")
            logger.info(f"    参数来源: dual_head.params_risk")
            logger.info("=" * 40)
            
            # 清理风险标签中的异常值（NaN, inf）
            valid_mask_train = np.isfinite(y_train_risk)
            valid_mask_val = np.isfinite(y_val_risk)
            
            n_invalid_train = (~valid_mask_train).sum()
            n_invalid_val = (~valid_mask_val).sum()
            
            if n_invalid_train > 0 or n_invalid_val > 0:
                logger.warning(f"风险标签包含异常值: 训练集 {n_invalid_train}/{len(y_train_risk)}, "
                             f"验证集 {n_invalid_val}/{len(y_val_risk)}")
                logger.info(f"过滤后: 训练集 {valid_mask_train.sum()} 样本, 验证集 {valid_mask_val.sum()} 样本")
                
                # 过滤数据
                X_train_clean = X_train[valid_mask_train]
                y_train_risk_clean = y_train_risk[valid_mask_train]
                X_val_clean = X_val[valid_mask_val]
                y_val_risk_clean = y_val_risk[valid_mask_val]
            else:
                X_train_clean = X_train
                y_train_risk_clean = y_train_risk
                X_val_clean = X_val
                y_val_risk_clean = y_val_risk
            
            if model_type == "lightgbm":
                risk_model = LGBModelWrapper(task_type="regression", custom_params=params_risk)
                # 获取风险头的early stopping
                risk_early_stop = params_risk.get("early_stopping_rounds", None)
                risk_model.train(X_train_clean, y_train_risk_clean, X_val_clean, y_val_risk_clean, 
                               feature_names=feature_names, early_stopping_rounds=risk_early_stop)
            else:  # xgboost
                risk_model = XGBModelWrapper(task_type="regression", custom_params=params_risk)
                # XGBoost不需要feature_names参数
                risk_model.train(X_train_clean, y_train_risk_clean, X_val_clean, y_val_risk_clean)
        
        return return_model, risk_model
    
    def fuse_predictions(self, pred_return: np.ndarray, pred_risk: np.ndarray) -> np.ndarray:
        """
        融合收益和风险预测
        
        Args:
            pred_return: 收益预测值
            pred_risk: 风险预测值
            
        Returns:
            融合后的预测分数（风险调整收益）
        """
        fusion_cfg = self.dual_head_cfg.get("fusion", {})
        method = fusion_cfg.get("method", "rank_ratio")
        normalize = fusion_cfg.get("normalize", True)
        
        return_weight = self.dual_head_cfg.get("return_head", {}).get("weight", 0.6)
        risk_weight = self.dual_head_cfg.get("risk_head", {}).get("weight", 0.4)
        
        # 归一化辅助函数
        def min_max_normalize(arr):
            arr = np.array(arr)
            min_val, max_val = arr.min(), arr.max()
            if max_val - min_val > 1e-8:
                return (arr - min_val) / (max_val - min_val)
            return np.zeros_like(arr)
        
        # 处理空值情况
        if pred_return is None or pred_risk is None:
            logger.warning("收益或风险预测为空，返回可用预测")
            return pred_return if pred_return is not None else pred_risk
        
        # 方案1: Sharpe-like 比值
        if method == "sharpe_like":
            epsilon = 1e-6
            fused = pred_return / (pred_risk + epsilon)
            logger.debug(f"使用 Sharpe-like 融合: return / (risk + {epsilon})")
        
        # 方案2: 分位数加权（推荐，对异常值不敏感）
        elif method == "rank_ratio":
            from scipy.stats import rankdata
            rank_return = rankdata(pred_return)
            rank_risk = rankdata(pred_risk)
            # 风险越低越好，所以用倒数
            fused = rank_return / (rank_risk + 1)
            logger.debug("使用分位数加权融合: rank(return) / rank(risk)")
        
        # 方案3: 效用函数
        elif method == "utility":
            risk_aversion = fusion_cfg.get("risk_aversion", 2.0)
            if normalize:
                pred_return = min_max_normalize(pred_return)
                pred_risk = min_max_normalize(pred_risk)
            fused = pred_return - risk_aversion * (pred_risk ** 2)
            logger.debug(f"使用效用函数融合: return - {risk_aversion} * risk^2")
        
        # 方案4: 加权平均（兼容旧版，不推荐）
        elif method == "weighted_average":
            if normalize:
                pred_return = min_max_normalize(pred_return)
                pred_risk = min_max_normalize(pred_risk)
            # 风险取反（风险越低分数越高）
            fused = return_weight * pred_return - risk_weight * pred_risk
            logger.debug(f"使用加权平均融合: {return_weight}*return - {risk_weight}*risk")
        
        else:
            logger.warning(f"未知融合方法: {method}，使用默认 rank_ratio")
            from scipy.stats import rankdata
            rank_return = rankdata(pred_return)
            rank_risk = rankdata(pred_risk)
            fused = rank_return / (rank_risk + 1)
        
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
            # 双头模型训练 (收益+风险预测)
            # ==========================================
            logger.info("=" * 50)
            logger.info(">>> 双头模型训练模式 (Return + Risk Prediction)")
            logger.info("=" * 50)
            
            # 收益标签（原有）
            y_train_return = df.loc[train_mask, label]
            y_val_return = df.loc[val_mask, label]
            
            # 风险标签（新增）
            label_risk = "label_risk"
            if label_risk not in df.columns:
                logger.error(f"未找到风险标签列 '{label_risk}'，请确保在 pipeline 中生成风险标签")
                raise ValueError(f"缺少风险标签列 '{label_risk}'")
            
            y_train_risk = df.loc[train_mask, label_risk]
            y_val_risk = df.loc[val_mask, label_risk]
            
            # 训练双头模型
            return_model, risk_model = self.train_dual_head(
                X_train, y_train_return, y_train_risk,
                X_val, y_val_return, y_val_risk,
                feature_names=features
            )
            
            # 保存模型（XGBoost使用.ubj格式，LightGBM使用.joblib）
            if return_model is not None:
                if model_type == "xgboost":
                    return_model.save(os.path.join(self.model_dir, "model_return.ubj"))
                else:
                    return_model.save(os.path.join(self.model_dir, "model_return.joblib"))
            if risk_model is not None:
                if model_type == "xgboost":
                    risk_model.save(os.path.join(self.model_dir, "model_risk.ubj"))
                else:
                    risk_model.save(os.path.join(self.model_dir, "model_risk.joblib"))
            
            # 生成预测
            pred_return = return_model.predict(df[features]) if return_model else None
            pred_risk = risk_model.predict(df[features]) if risk_model else None
            
            # 融合预测
            df["pred_return"] = pred_return
            df["pred_risk"] = pred_risk
            df["pred_score"] = self.fuse_predictions(pred_return, pred_risk)
            
            # 保存结果
            out_cols = ["date", "symbol", "close", label, label_risk, "pred_return", "pred_risk", "pred_score"]
            out_cols = [c for c in out_cols if c in df.columns]
            df[out_cols].to_parquet(os.path.join(self.model_dir, "predictions.parquet"), index=False)
            
            logger.info(f"双头模型训练完成。")
            logger.info(f"融合方法: {self.dual_head_cfg.get('fusion', {}).get('method', 'rank_ratio')}")

            
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
