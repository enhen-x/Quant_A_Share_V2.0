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
    动态融合预测分数 - 支持新旧双头模型架构
    
    新架构: pred_return + pred_risk
    旧架构: pred_reg + pred_cls (向后兼容)
    """
    dual_head_cfg = GLOBAL_CONFIG.get("model", {}).get("dual_head", {})
    
    # 检测列名
    has_return = "pred_return" in pred_df.columns
    has_risk = "pred_risk" in pred_df.columns
    has_reg = "pred_reg" in pred_df.columns
    has_cls = "pred_cls" in pred_df.columns
    
    is_dual_head = (has_return and has_risk) or (has_reg and has_cls)
    
    if not is_dual_head:
        logger.info("检测到单模型预测，使用原有 pred_score")
        return pred_df
    
    # 读取融合配置
    fusion_cfg = dual_head_cfg.get("fusion", {})
    method = fusion_cfg.get("method", "rank_ratio")
    risk_aversion = fusion_cfg.get("risk_aversion", 2.0)
    
    # 使用新架构或旧架构
    if has_return and has_risk:
        logger.info(f"检测到双头模型 (收益+风险预测)，融合方法={method}")
        pred_return = pred_df["pred_return"].values
        pred_risk = pred_df["pred_risk"].values
    else:
        logger.info(f"检测到旧版双头模型 (回归+分类)，融合方法={method}")
        pred_return = pred_df["pred_reg"].values
        pred_risk = pred_df["pred_cls"].values
    
    # 融合逻辑
    from scipy.stats import rankdata
    
    if method == "rank_ratio":
        # 排名比率 - 使用百分位避免数值过大
        n = len(pred_return)
        rank_return = rankdata(pred_return, nan_policy='omit') / n
        rank_risk = rankdata(pred_risk, nan_policy='omit') / n
        epsilon = 0.01
        fused = rank_return / (rank_risk + epsilon)
        # 归一化到 [0, 1]
        fused = (fused - np.nanmin(fused)) / (np.nanmax(fused) - np.nanmin(fused) + 1e-8)
        
    elif method == "sharpe_like":
        # Sharpe-like比率
        epsilon = np.nanstd(pred_risk) * 0.1 if np.nanstd(pred_risk) > 0 else 0.01
        fused = pred_return / (pred_risk + epsilon)
        
    elif method == "utility":
        # 效用函数
        fused = pred_return - risk_aversion * (pred_risk ** 2)
        
    elif method == "weighted_average":
        # 加权平均（兼容旧版）
        return_weight = dual_head_cfg.get("return_head", {}).get("weight", 0.6)
        risk_weight = dual_head_cfg.get("risk_head", {}).get("weight", 0.4)
        # 归一化
        ret_norm = (pred_return - np.nanmin(pred_return)) / (np.nanmax(pred_return) - np.nanmin(pred_return) + 1e-8)
        risk_norm = (pred_risk - np.nanmin(pred_risk)) / (np.nanmax(pred_risk) - np.nanmin(pred_risk) + 1e-8)
        fused = return_weight * ret_norm - risk_weight * risk_norm
        
    else:
        logger.warning(f"未知融合方法: {method}，使用 rank_ratio")
        rank_return = rankdata(pred_return, nan_policy='omit')
        rank_risk = rankdata(pred_risk, nan_policy='omit')
        fused = rank_return / (rank_risk + 1.0)
    
    pred_df["pred_score"] = fused
    logger.info(f"融合完成，pred_score 范围: [{np.nanmin(fused):.4f}, {np.nanmax(fused):.4f}]")
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