# src/model/lgb_model.py
"""
LightGBM 模型包装器

支持回归和分类两种任务类型，用于双头模型系统。
"""

import lightgbm as lgb
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from src.utils.config import GLOBAL_CONFIG
from src.utils.logger import get_logger

logger = get_logger()


class LGBModelWrapper:
    """LightGBM 模型包装器，支持回归和分类任务"""
    
    def __init__(self, task_type: str = "regression", params_key: str = "lgb_params", custom_params: dict = None):
        """
        初始化模型包装器
        
        Args:
            task_type: "regression" (回归) 或 "classification" (分类)
            params_key: 参数配置key，默认 "lgb_params"，风险头可用 "lgb_params_risk"
            custom_params: 自定义参数字典，如果提供则直接使用，优先级最高
        """
        self.task_type = task_type
        self.conf = GLOBAL_CONFIG.get("model", {})
        
        # 参数优先级: custom_params > params_key > lgb_params
        if custom_params is not None:
            self.lgb_params = custom_params
        else:
            self.lgb_params = self.conf.get(params_key, self.conf.get("lgb_params", {}))
        
        self.model = None
        self.feature_names = None
        self.evals_result = {}
        
        # 构建参数
        self.params = self._build_params()
        
    def _build_params(self) -> Dict[str, Any]:
        """根据任务类型构建参数"""
        # 基础参数
        params = {
            "n_estimators": self.lgb_params.get("n_estimators", 3000),
            "max_depth": self.lgb_params.get("max_depth", 6),
            "learning_rate": self.lgb_params.get("learning_rate", 0.01),
            "num_leaves": self.lgb_params.get("num_leaves", 63),
            "subsample": self.lgb_params.get("subsample", 0.8),
            "colsample_bytree": self.lgb_params.get("colsample_bytree", 0.8),
            "reg_alpha": self.lgb_params.get("reg_alpha", 0.1),
            "reg_lambda": self.lgb_params.get("reg_lambda", 1.0),
            "min_child_samples": self.lgb_params.get("min_child_samples", 20),
            "random_state": 42,
            "n_jobs": -1,
            "verbose": self.lgb_params.get("verbose", -1),
        }
        
        # GPU 配置
        device = self.lgb_params.get("device", "cpu")
        if device == "gpu":
            params["device"] = "gpu"
            params["gpu_platform_id"] = 0
            params["gpu_device_id"] = 0
        
        # 根据任务类型设置目标函数
        if self.task_type == "classification":
            params["objective"] = "binary"
            params["metric"] = "auc"
        else:
            params["objective"] = "regression"
            params["metric"] = "rmse"
        
        return params
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              feature_names: list = None, early_stopping_rounds: int = None):
        """
        训练模型
        
        Args:
            X_train: 训练集特征
            y_train: 训练集标签
            X_val: 验证集特征（可选）
            y_val: 验证集标签（可选）
            feature_names: 特征名列表
            early_stopping_rounds: 早停轮数
        """
        logger.info(f"初始化 LightGBM ({self.task_type})")
        logger.info(f"参数: {self.params}")
        
        # 保存特征名
        if feature_names is not None:
            self.feature_names = feature_names
        elif hasattr(X_train, 'columns'):
            self.feature_names = list(X_train.columns)
        
        # 获取早停配置
        if early_stopping_rounds is None:
            early_stopping_rounds = self.conf.get("early_stopping_rounds", 100)
        
        # 提取 n_estimators（sklearn API 需要）
        n_estimators = self.params.pop("n_estimators", 3000)
        
        # 创建模型
        if self.task_type == "classification":
            self.model = lgb.LGBMClassifier(n_estimators=n_estimators, **self.params)
        else:
            self.model = lgb.LGBMRegressor(n_estimators=n_estimators, **self.params)
        
        # 准备回调
        callbacks = [
            lgb.log_evaluation(period=100),
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True)
        ]
        
        # 准备验证集
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
        
        # 训练
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            callbacks=callbacks
        )
        
        # 记录最佳迭代
        if hasattr(self.model, 'best_iteration_'):
            logger.info(f"最佳迭代轮数: {self.model.best_iteration_}")
        
        logger.info("LightGBM 训练完成")
    
    def predict(self, X) -> np.ndarray:
        """
        预测
        
        Args:
            X: 特征数据
            
        Returns:
            预测值 (回归返回连续值，分类返回概率)
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        if self.task_type == "classification":
            # 分类任务返回正类概率
            return self.model.predict_proba(X)[:, 1]
        else:
            # 回归任务返回预测值
            return self.model.predict(X)
    
    def get_feature_importance(self, importance_type: str = 'gain') -> Dict[str, float]:
        """
        获取特征重要性
        
        Args:
            importance_type: 'gain' 或 'split'
            
        Returns:
            {特征名: 重要性分数}
        """
        if self.model is None:
            return {}
        
        try:
            importance = self.model.feature_importances_
            if self.feature_names is not None:
                return dict(zip(self.feature_names, importance))
            else:
                return dict(enumerate(importance))
        except Exception as e:
            logger.warning(f"获取特征重要性失败: {e}")
            return {}
    
    def save(self, path: str):
        """保存模型到文件"""
        if self.model is None:
            raise ValueError("模型尚未训练，无法保存")
        
        # 使用 joblib 保存 sklearn 接口的模型
        import joblib
        joblib.dump({
            'model': self.model,
            'task_type': self.task_type,
            'feature_names': self.feature_names,
            'params': self.params
        }, path)
        logger.info(f"模型已保存: {path}")
    
    def load(self, path: str):
        """从文件加载模型"""
        import joblib
        data = joblib.load(path)
        self.model = data['model']
        self.task_type = data.get('task_type', 'regression')
        self.feature_names = data.get('feature_names')
        self.params = data.get('params', {})
        logger.info(f"模型已加载: {path}")
