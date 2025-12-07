# scripts/run_backtest.py

import os
import sys
import pandas as pd
import glob

# 路径适配
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.config import GLOBAL_CONFIG
from src.utils.logger import get_logger
from src.utils.io import read_parquet
from src.strategy.signal import TopKSignalStrategy
from src.backtest.backtester import VectorBacktester

logger = get_logger()

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