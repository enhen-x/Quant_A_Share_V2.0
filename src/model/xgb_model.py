import os
import xgboost as xgb
from src.utils.config import GLOBAL_CONFIG
from src.utils.logger import get_logger
from typing import Optional

logger = get_logger()

class XGBModelWrapper:
    def __init__(self, task_type: str = "regression", params_key: str = "params", custom_params: dict = None):
        """
        初始化XGBoost模型包装器
        
        Args:
            task_type: 任务类型（暂未使用，保留用于统一接口）
            params_key: 参数配置key，默认 "params"，可用 "xgb_params" 或 "xgb_params_risk"
            custom_params: 自定义参数字典，如果提供则直接使用，优先级最高
        """
        self.conf = GLOBAL_CONFIG["model"]
        
        # 参数优先级: custom_params > params_key > "params"
        if custom_params is not None:
            self.params = custom_params.copy()
        else:
            self.params = self.conf.get(params_key, self.conf.get("params", {})).copy()
        
        
        # 过滤掉LightGBM专用参数（避免警告）
        # 注意: device 参数保留（XGBoost用于GPU: cuda, LightGBM用于GPU: gpu）
        lgb_only_params = ['num_leaves', 'min_child_samples', 'verbose']
        for param in lgb_only_params:
            if param in self.params:
                self.params.pop(param)
        
        self.model = None
        self.evals_result = {}  # 存储训练历史
        self.feature_names = None  # 存储特征名称

    def train(self, X_train, y_train, X_val=None, y_val=None, 
              monitor=None, experiment_name: str = None):
        """
        训练 XGBoost 模型
        
        Args:
            X_train: 训练集特征
            y_train: 训练集标签
            X_val: 验证集特征（可选）
            y_val: 验证集标签（可选）
            monitor: TrainingMonitor 实例（可选，用于 TensorBoard 监控）
            experiment_name: 实验名称（可选）
        """
        logger.info(f"初始化 XGBoost (Params: {self.params})")
        
        # 记录特征名称（用于验证预测数据）
        if hasattr(X_train, 'columns'):
            self.feature_names = list(X_train.columns)
        
        # 转换为 DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        evals = []
        if X_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals = [(dtrain, 'train'), (dval, 'eval')]
            
        # 1. 拷贝一份参数，并移除 'n_estimators'
        train_params = self.params.copy()
        num_rounds = train_params.pop('n_estimators', 1000)
        
        # 2. 准备回调函数列表
        callbacks = []
        
        # 如果启用监控，创建监控器
        if monitor is None:
            # 检查配置中是否启用监控
            enable_monitor = self.conf.get("enable_tensorboard", True)
            if enable_monitor:
                from src.model.training_monitor import TrainingMonitor
                monitor = TrainingMonitor(experiment_name=experiment_name)
        
        if monitor is not None:
            # 记录超参数
            monitor.log_hyperparams(self.params)
            
            # 记录数据信息
            label_col = self.conf.get("label_col", "unknown")
            monitor.log_data_info(
                train_size=X_train.shape[0],
                val_size=X_val.shape[0] if X_val is not None else 0,
                n_features=X_train.shape[1],
                label_name=label_col
            )
            
            # 创建回调函数
            from src.model.training_monitor import create_xgb_callback
            callbacks.append(create_xgb_callback(monitor))
        
        # 3. 获取早停轮数（从配置读取，默认 50）
        early_stopping = self.conf.get("early_stopping_rounds", 50)
        
        # 4. 开始训练
        self.evals_result = {}
        self.model = xgb.train(
            params=train_params,
            dtrain=dtrain,
            num_boost_round=num_rounds,
            evals=evals,
            early_stopping_rounds=early_stopping,
            verbose_eval=100,
            evals_result=self.evals_result,
            callbacks=callbacks if callbacks else None
        )
        
        # 4. 记录训练总结和特征重要性
        if monitor is not None:
            # 获取最终损失
            final_train_loss = None
            final_eval_loss = None
            
            if 'train' in self.evals_result:
                metric_name = list(self.evals_result['train'].keys())[0]
                final_train_loss = self.evals_result['train'][metric_name][-1]
            if 'eval' in self.evals_result:
                metric_name = list(self.evals_result['eval'].keys())[0]
                final_eval_loss = self.evals_result['eval'][metric_name][-1]
            
            # 记录特征重要性
            importance = self.get_feature_importance()
            if importance:
                monitor.log_feature_importance(importance)
            
            # 记录训练总结
            early_stopped = self.model.best_iteration < num_rounds - 1
            if final_train_loss is not None:
                monitor.log_training_summary(
                    final_train_loss=final_train_loss,
                    final_eval_loss=final_eval_loss,
                    early_stopped=early_stopped
                )
            
            monitor.close()
        
        logger.info("模型训练完成。")
    
    def get_feature_importance(self, importance_type: str = 'weight') -> dict:
        """
        获取特征重要性
        
        Args:
            importance_type: 重要性类型 ('weight', 'gain', 'cover')
            
        Returns:
            {特征名: 重要性分数} 字典
        """
        if self.model is None:
            return {}
        try:
            return self.model.get_score(importance_type=importance_type)
        except Exception as e:
            logger.warning(f"获取特征重要性失败: {e}")
            return {}

    def predict(self, X):
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        # 如果有记录的特征名称，确保输入数据列顺序一致
        if self.feature_names and hasattr(X, 'columns'):
            # 检查是否所有特征都存在
            missing = [f for f in self.feature_names if f not in X.columns]
            if missing:
                raise ValueError(f"输入数据缺少模型所需的特征: {missing}")
            
            # 按照训练时的顺序重新排列特征列
            X = X[self.feature_names]
        
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
    
    def save(self, path):
        """保存模型（默认使用UBJSON格式）"""
        if self.model is None:
            raise ValueError("模型尚未训练")
        self.model.save_model(path)
        
        # 保存特征名称到同名的 .features 文件
        if self.feature_names:
            import json
            feature_file = path.replace('.ubj', '.features').replace('.json', '.features')
            with open(feature_file, 'w') as f:
                json.dump({'feature_names': self.feature_names}, f, indent=2)

    def load(self, path):
        self.model = xgb.Booster()
        self.model.load_model(path)
        
        # 尝试从 .features 文件加载特征名称
        feature_file = path.replace('.ubj', '.features').replace('.json', '.features')
        if os.path.exists(feature_file):
            import json
            with open(feature_file, 'r') as f:
                data = json.load(f)
                self.feature_names = data.get('feature_names', [])
            logger.info(f"从 {os.path.basename(feature_file)} 加载了 {len(self.feature_names)} 个特征名称")
        else:
            # 尝试从模型本身提取（XGBoost 模型可能保存了特征名）
            try:
                self.feature_names = self.model.feature_names
                logger.info(f"从模型本身提取了 {len(self.feature_names)} 个特征名称")
            except:
                logger.warning("无法加载特征名称，预测时需要确保特征顺序一致")
                self.feature_names = None