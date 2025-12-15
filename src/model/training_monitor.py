# src/model/training_monitor.py
"""
TensorBoard 训练监控模块
用于可视化监控 XGBoost 模型训练过程
"""

import os
import datetime
from typing import Dict, List, Optional, Any
from tensorboardX import SummaryWriter
from src.utils.config import GLOBAL_CONFIG
from src.utils.logger import get_logger

logger = get_logger()


class TrainingMonitor:
    """
    XGBoost 训练监控器
    
    功能：
    1. 记录训练/验证损失曲线
    2. 记录特征重要性
    3. 记录超参数配置
    4. 支持多次实验对比
    """
    
    def __init__(self, experiment_name: str = None, log_dir: str = None):
        """
        初始化监控器
        
        Args:
            experiment_name: 实验名称，用于区分不同训练会话
            log_dir: TensorBoard 日志保存目录
        """
        # 默认日志目录
        if log_dir is None:
            base_log_dir = os.path.join(GLOBAL_CONFIG["paths"]["logs"], "tensorboard")
        else:
            base_log_dir = log_dir
            
        # 生成唯一的实验名称
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if experiment_name:
            self.run_name = f"{experiment_name}_{timestamp}"
        else:
            self.run_name = f"train_{timestamp}"
            
        self.log_dir = os.path.join(base_log_dir, self.run_name)
        
        # 确保目录存在
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 初始化 TensorBoard Writer
        self.writer = SummaryWriter(self.log_dir)
        
        # 存储训练历史
        self.train_losses: List[float] = []
        self.eval_losses: List[float] = []
        self.best_iteration: int = 0
        self.best_score: float = float('inf')
        
        logger.info(f"[监控] TensorBoard 日志目录: {self.log_dir}")
        logger.info(f"[监控] 启动命令: tensorboard --logdir={base_log_dir}")
    
    def log_hyperparams(self, params: Dict[str, Any], model_type: str = "XGBoost"):
        """
        记录模型超参数
        
        Args:
            params: 超参数字典
            model_type: 模型类型名称
        """
        # 将超参数写入 TensorBoard
        param_text = f"## {model_type} 超参数配置\n\n"
        for key, value in params.items():
            param_text += f"- **{key}**: {value}\n"
            # 数值型参数可以单独记录
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"hyperparams/{key}", value, 0)
        
        self.writer.add_text("config/hyperparams", param_text, 0)
        logger.info(f"[监控] 超参数已记录: {len(params)} 项")
    
    def log_data_info(self, train_size: int, val_size: int, 
                      n_features: int, label_name: str):
        """
        记录数据集信息
        
        Args:
            train_size: 训练集样本数
            val_size: 验证集样本数
            n_features: 特征数量
            label_name: 标签列名
        """
        info_text = f"""## 数据集信息

| 项目 | 值 |
|------|-----|
| 训练集样本数 | {train_size:,} |
| 验证集样本数 | {val_size:,} |
| 特征数量 | {n_features} |
| 标签列 | {label_name} |
"""
        self.writer.add_text("config/data_info", info_text, 0)
        
        # 也记录为标量
        self.writer.add_scalar("data/train_size", train_size, 0)
        self.writer.add_scalar("data/val_size", val_size, 0)
        self.writer.add_scalar("data/n_features", n_features, 0)
        
        logger.info(f"[监控] 数据信息已记录: 训练集 {train_size:,}, 验证集 {val_size:,}")
    
    def log_iteration(self, iteration: int, train_loss: float, 
                      eval_loss: float = None):
        """
        记录单次迭代的损失值
        
        Args:
            iteration: 当前迭代轮次
            train_loss: 训练集损失
            eval_loss: 验证集损失（可选）
        """
        self.train_losses.append(train_loss)
        self.writer.add_scalar("loss/train", train_loss, iteration)
        
        if eval_loss is not None:
            self.eval_losses.append(eval_loss)
            self.writer.add_scalar("loss/eval", eval_loss, iteration)
            
            # 记录损失差距（过拟合指标）
            gap = eval_loss - train_loss
            self.writer.add_scalar("loss/gap", gap, iteration)
            
            # 更新最佳分数
            if eval_loss < self.best_score:
                self.best_score = eval_loss
                self.best_iteration = iteration
    
    def log_feature_importance(self, importance_dict: Dict[str, float], 
                                top_k: int = 20):
        """
        记录特征重要性
        
        Args:
            importance_dict: {特征名: 重要性分数} 字典
            top_k: 记录前 K 个最重要的特征
        """
        if not importance_dict:
            return
            
        # 排序并取 Top K
        sorted_features = sorted(importance_dict.items(), 
                                  key=lambda x: x[1], reverse=True)[:top_k]
        
        # 记录为柱状图数据
        for rank, (feat, score) in enumerate(sorted_features):
            self.writer.add_scalar(f"feature_importance/{feat}", score, 0)
        
        # 创建可视化文本
        fi_text = "## Top 特征重要性\n\n"
        fi_text += "| 排名 | 特征名 | 重要性分数 |\n"
        fi_text += "|------|--------|------------|\n"
        for rank, (feat, score) in enumerate(sorted_features, 1):
            fi_text += f"| {rank} | {feat} | {score:.4f} |\n"
        
        self.writer.add_text("analysis/feature_importance", fi_text, 0)
        logger.info(f"[监控] 特征重要性已记录: Top {len(sorted_features)} 项")
    
    def log_training_summary(self, final_train_loss: float, 
                              final_eval_loss: float = None,
                              early_stopped: bool = False):
        """
        记录训练总结
        
        Args:
            final_train_loss: 最终训练损失
            final_eval_loss: 最终验证损失
            early_stopped: 是否提前停止
        """
        # 格式化验证损失
        eval_loss_str = f"{final_eval_loss:.6f}" if final_eval_loss is not None else "N/A"
        early_stop_str = "是" if early_stopped else "否"
        
        summary_text = f"""## 训练总结

| 指标 | 值 |
|------|-----|
| 最终训练损失 | {final_train_loss:.6f} |
| 最终验证损失 | {eval_loss_str} |
| 最佳迭代轮次 | {self.best_iteration} |
| 最佳验证损失 | {self.best_score:.6f} |
| 总迭代次数 | {len(self.train_losses)} |
| 提前停止 | {early_stop_str} |
"""
        self.writer.add_text("summary/training", summary_text, 0)
        
        # 记录最终指标
        self.writer.add_scalar("final/train_loss", final_train_loss, 0)
        if final_eval_loss:
            self.writer.add_scalar("final/eval_loss", final_eval_loss, 0)
        self.writer.add_scalar("final/best_iteration", self.best_iteration, 0)
        
        logger.info(f"[监控] 训练总结: 最佳轮次 {self.best_iteration}, 最佳损失 {self.best_score:.6f}")
    
    def close(self):
        """关闭监控器，刷新所有数据"""
        self.writer.close()
        logger.info(f"[监控] 监控会话已关闭")
        logger.info(f"[监控] 查看结果: tensorboard --logdir={os.path.dirname(self.log_dir)}")


class TensorBoardCallback:
    """
    XGBoost TrainingCallback 兼容的监控回调类
    支持 XGBoost 2.0+ 版本
    """
    
    def __init__(self, monitor: TrainingMonitor):
        """
        Args:
            monitor: TrainingMonitor 实例
        """
        self.monitor = monitor
    
    def __call__(self, env):
        """
        旧版 XGBoost 回调接口
        """
        self._log_metrics(env.iteration, env.evaluation_result_list)
    
    def _log_metrics(self, iteration: int, evaluation_result_list):
        """解析并记录指标"""
        train_loss = None
        eval_loss = None
        
        for item in evaluation_result_list:
            if len(item) >= 3:
                data_name = item[0]
                value = item[2]
                if data_name == "train":
                    train_loss = value
                elif data_name == "eval":
                    eval_loss = value
        
        if train_loss is not None:
            self.monitor.log_iteration(iteration, train_loss, eval_loss)


# 尝试导入新版 XGBoost 的 TrainingCallback
try:
    from xgboost.callback import TrainingCallback
    
    class XGBTensorBoardCallback(TrainingCallback):
        """
        XGBoost 2.0+ 兼容的 TrainingCallback
        """
        
        def __init__(self, monitor: TrainingMonitor):
            super().__init__()
            self.monitor = monitor
        
        def after_iteration(self, model, epoch, evals_log):
            """
            每轮迭代后调用
            
            Args:
                model: 当前模型
                epoch: 当前轮次
                evals_log: 评估日志 dict
                
            Returns:
                False 继续训练，True 停止训练
            """
            train_loss = None
            eval_loss = None
            
            # 解析 evals_log: {'train': {'rmse': [...]}, 'eval': {'rmse': [...]}}
            for data_name, metrics in evals_log.items():
                for metric_name, values in metrics.items():
                    if values:  # 确保有值
                        current_value = values[-1]  # 取最新的值
                        if data_name == "train":
                            train_loss = current_value
                        elif data_name == "eval":
                            eval_loss = current_value
            
            if train_loss is not None:
                self.monitor.log_iteration(epoch, train_loss, eval_loss)
            
            return False  # 继续训练
    
    # 使用新版回调
    _USE_NEW_CALLBACK = True
    
except ImportError:
    # 旧版 XGBoost，使用函数式回调
    _USE_NEW_CALLBACK = False
    XGBTensorBoardCallback = None


def create_xgb_callback(monitor: TrainingMonitor):
    """
    创建 XGBoost 兼容的回调
    自动检测 XGBoost 版本并返回合适的回调类型
    
    Args:
        monitor: TrainingMonitor 实例
        
    Returns:
        回调对象（TrainingCallback 或函数）
    """
    if _USE_NEW_CALLBACK and XGBTensorBoardCallback is not None:
        return XGBTensorBoardCallback(monitor)
    else:
        # 旧版回调函数
        return TensorBoardCallback(monitor)

