# scripts/run_backtest.py

import os
import sys
import pandas as pd
import numpy as np
import glob

# 路径适配
current_dir = os.path.dirname(os.path.abspath(__file__))
# 从当前文件位置 (scripts/analisis) 返回两级到项目根目录
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.config import GLOBAL_CONFIG
from src.utils.logger import get_logger
from src.utils.io import read_parquet
from src.strategy.signal import TopKSignalStrategy
from src.backtest.backtester import VectorBacktester

logger = get_logger()


def fuse_predictions_dynamic(pred_df: pd.DataFrame) -> pd.DataFrame:
    """
    动态融合预测分数 - 根据当前配置文件中的权重重新计算 pred_score
    
    如果 pred_reg 和 pred_cls 列存在，则使用配置权重融合；
    否则直接使用已有的 pred_score。
    """
    dual_head_cfg = GLOBAL_CONFIG.get("model", {}).get("dual_head", {})
    
    has_reg = "pred_reg" in pred_df.columns
    has_cls = "pred_cls" in pred_df.columns
    
    if not has_reg and not has_cls:
        # 单模型训练，直接使用原有 pred_score
        logger.info("检测到单模型预测，使用原有 pred_score")
        return pred_df
    
    # 读取配置权重
    reg_weight = dual_head_cfg.get("regression", {}).get("weight", 0.6)
    cls_weight = dual_head_cfg.get("classification", {}).get("weight", 0.4)
    normalize = dual_head_cfg.get("fusion", {}).get("normalize", True)
    method = dual_head_cfg.get("fusion", {}).get("method", "weighted_average")
    
    logger.info(f"动态融合: 回归权重={reg_weight}, 分类权重={cls_weight}, 融合方法={method}")
    
    # 归一化函数
    def min_max_normalize(arr):
        arr = np.array(arr)
        min_val, max_val = np.nanmin(arr), np.nanmax(arr)
        if max_val - min_val > 1e-8:
            return (arr - min_val) / (max_val - min_val)
        return np.zeros_like(arr)
    
    pred_reg = pred_df["pred_reg"].values if has_reg else None
    pred_cls = pred_df["pred_cls"].values if has_cls else None
    
    # 归一化
    if normalize:
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
    
    pred_df["pred_score"] = fused
    return pred_df


def get_latest_model_version():
    """自动寻找 data/models 下最新的版本目录"""
    models_dir = GLOBAL_CONFIG["paths"]["models"]
    if not os.path.exists(models_dir):
        return None
    
    # 找子目录
    subdirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    if not subdirs:
        return None
    
    # 按时间排序 (假设目录名是时间戳)
    subdirs.sort(reverse=True)
    return subdirs[0]


def main():
    # 1. 确定模型版本
    version = get_latest_model_version()
    if not version:
        logger.error("未找到任何模型版本，请先运行 train_model.py")
        return
    
    logger.info(f"=== 启动回测 (Model Version: {version}) ===")
    
    model_dir = os.path.join(GLOBAL_CONFIG["paths"]["models"], version)
    pred_path = os.path.join(model_dir, "predictions.parquet")
    
    if not os.path.exists(pred_path):
        logger.error(f"预测文件不存在: {pred_path}")
        return

    # 2. 加载预测数据
    pred_df = read_parquet(pred_path)
    # 确保 date 是 datetime 类型
    pred_df["date"] = pd.to_datetime(pred_df["date"])
    
    # 3. 动态融合预测分数 (根据当前配置权重)
    pred_df = fuse_predictions_dynamic(pred_df)
    
    # 3. 生成策略信号
    strategy = TopKSignalStrategy()
    signal_df = strategy.generate(pred_df)
    
    # 4. 运行回测
    backtester = VectorBacktester()
    # 结果保存在模型目录下的 backtest 子目录
    out_dir = os.path.join(model_dir, "backtest_result")
    
    metrics = backtester.run(signal_df, output_dir=out_dir)
    
    logger.info("=" * 40)
    logger.info(f"回测结果已保存至: {out_dir}")
    logger.info(f"查看资金曲线: {os.path.join(out_dir, 'equity_curve.png')}")
    logger.info("=" * 40)

if __name__ == "__main__":
    main()