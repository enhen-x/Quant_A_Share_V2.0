import shap
import pandas as pd
import matplotlib.pyplot as plt
import os
from src.utils.logger import get_logger

logger = get_logger()

class ModelInterpreter:
    """
    基于 SHAP 的模型解释器
    """
    def __init__(self, model, data: pd.DataFrame = None):
        """
        初始化解释器
        
        Args:
            model: 训练好的模型对象 (xgboost.Booster 或 sklearn interface)
            data: 用于计算 SHAP 值的背景数据 (DataFrame)
        """
        self.model = model
        self.data = data
        self.explainer = None
        self.shap_values = None
        
        # 尝试使用 TreeExplainer，适用于 XGBoost/LightGBM/RandomForest
        try:
            self.explainer = shap.TreeExplainer(model)
        except Exception as e:
            logger.warning(f"无法初始化 TreeExplainer, 尝试默认 Explainer: {e}")
            self.explainer = shap.Explainer(model)

    def compute_shap_values(self, X: pd.DataFrame):
        """
        计算 SHAP 值
        
        Args:
            X: 需要解释的特征数据
            
        Returns:
            shap_values: SHAP 值对象
        """
        logger.info(f"开始计算 SHAP 值, 数据量: {X.shape}")
        
        try:
            # 尝试直接计算
            self.shap_values = self.explainer(X)
        except Exception as e1:
            logger.warning(f"SHAP计算失败 (Attempt 1 - Direct): {e1}")
            try:
                # 回退：直接传入 numpy array
                self.shap_values = self.explainer(X.values)
            except Exception as e2:
                logger.warning(f"SHAP计算失败 (Attempt 2 - Numpy): {e2}")
                try:
                    # 再次回退：使用通用的 PermutationExplainer (速度较慢但通用)
                    logger.warning("尝试切换到通用 Explainer (可能较慢)...")
                    
                    # 定义包装函数，处理 DMatrix 转换
                    def predict_wrapper(X_data):
                        import xgboost as xgb
                        import pandas as pd
                        import numpy as np
                        
                        # 如果输入是 numpy 或 pandas，转换为 DMatrix
                        if isinstance(X_data, (pd.DataFrame, np.ndarray)):
                             return self.model.predict(xgb.DMatrix(X_data))
                        return self.model.predict(X_data)

                    # 注意：这里重新初始化一个通用的 Explainer
                    generic_explainer = shap.Explainer(predict_wrapper, X)
                    self.shap_values = generic_explainer(X)
                except Exception as e3:
                     logger.error(f"SHAP计算最终失败: {e3}")
                     raise e3
                     
        logger.info("SHAP 值计算完成")
        return self.shap_values

    def plot_summary(self, save_path: str = None, max_display: int = 20, plot_type: str = 'dot'):
        """
        绘制 Summary Plot (特征重要性概览)
        
        Args:
            save_path: 图片保存路径
            max_display: 显示前 N 个特征
            plot_type: 'dot' (beeswarm) 或 'bar'
        """
        if self.shap_values is None:
            raise ValueError("请先调用 compute_shap_values() 计算 SHAP 值")
            
        plt.figure(figsize=(10, 8))
        shap.summary_plot(self.shap_values, self.data if self.data is not None else None, 
                          max_display=max_display, plot_type=plot_type, show=False)
        
        if save_path:
            self._save_plot(save_path)

    def plot_dependence(self, feature: str, interaction_feature: str = None, save_path: str = None):
        """
        绘制 Dependence Plot (特征依赖图)
        
        Args:
            feature: 目标特征名
            interaction_feature: 交互特征名 (可选，None 表示自动选择)
            save_path: 图片保存路径
        """
        if self.shap_values is None:
            raise ValueError("请先调用 compute_shap_values() 计算 SHAP 值")
            
        # dependence_plot 需要 shap_values 的 values 属性 (numpy array) 如果 shap_values 是 Explanation 对象
        # 如果是 Explanation 对象，shap.dependence_plot 的第一个参数可以是 'feature_name'，第二个参数是 shap_values, 第三个是 features (X)
        
        # 简化处理：使用 shap.plots.scatter (新版 API) 或者 shap.dependence_plot (旧版 API)
        # 这里为了兼容性，假设 self.shap_values 是 Explanation 对象
        
        plt.figure(figsize=(10, 6))
        
        try:
            # 尝试使用新版 API
            shap.plots.scatter(self.shap_values[:, feature], color=self.shap_values[:, interaction_feature] if interaction_feature else None, show=False)
        except Exception:
             # 回退旧版 API (需要 feature values)
             # 这里比较麻烦，需要 data
             if self.data is not None and feature in self.data.columns:
                 shap.dependence_plot(feature, self.shap_values.values, self.data, interaction_index=interaction_feature, show=False)
             else:
                 logger.warning(f"无法绘制 {feature} 的 Dependence Plot，可能缺少原始数据或版本兼容问题")
                 return

        if save_path:
            self._save_plot(save_path)

    def _save_plot(self, path):
        """辅助函数：保存并关闭图表"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"图表已保存: {path}")
