# scripts/check_stress_test.py

import os
import sys
import pandas as pd
import datetime
import matplotlib.pyplot as plt

# 路径适配
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.config import GLOBAL_CONFIG
from src.utils.logger import get_logger
from src.utils.io import read_parquet, ensure_dir
from src.strategy.signal import TopKSignalStrategy
from src.backtest.backtester import VectorBacktester

logger = get_logger()

def get_latest_predictions():
    """获取最新的预测文件"""
    models_dir = GLOBAL_CONFIG["paths"]["models"]
    if not os.path.exists(models_dir):
        return None
    
    subdirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    if not subdirs:
        return None
    
    # 找最新的目录
    subdirs.sort(reverse=True)
    latest_dir = subdirs[0]
    pred_path = os.path.join(models_dir, latest_dir, "predictions.parquet")
    
    if os.path.exists(pred_path):
        logger.info(f"使用预测文件: {pred_path}")
        return read_parquet(pred_path)
    return None

def main():
    logger.info("=== 启动策略压力测试 (Stress Test) ===")
    
    # 1. 准备数据和信号
    pred_df = get_latest_predictions()
    if pred_df is None:
        logger.error("未找到预测文件，请先运行 run_walkforward.py")
        return
        
    pred_df["date"] = pd.to_datetime(pred_df["date"])
    
    # 生成基础信号
    logger.info("正在生成基础策略信号...")
    strategy = TopKSignalStrategy()
    signal_df = strategy.generate(pred_df)
    
    if signal_df.empty:
        logger.error("信号生成为空，无法进行压力测试。")
        return

    backtester = VectorBacktester()
    report_dir = os.path.join(GLOBAL_CONFIG["paths"]["reports"], "stress_test")
    ensure_dir(report_dir)

    # ==========================================
    # 场景一：交易成本敏感性测试 (Cost Sensitivity)
    # ==========================================
    logger.info("\n>>> 场景 1: 交易成本敏感性测试")
    costs = [0.001, 0.002, 0.003, 0.005] # 千1, 千2, 千3, 千5
    cost_results = []
    
    for c in costs:
        logger.info(f"Testing Cost = {c*1000:.1f}‰ ...")
        out_path = os.path.join(report_dir, f"cost_{int(c*10000)}")
        metrics = backtester.run(signal_df, output_dir=out_path, cost_rate=c)
        cost_results.append({
            "Cost Rate": f"{c*1000:.1f}‰",
            "Ann Return": metrics["annual_return"],
            "Sharpe": metrics["sharpe"]
        })
    
    df_cost = pd.DataFrame(cost_results)
    print("\n--- 成本敏感性报告 ---")
    print(df_cost.to_markdown(index=False, floatfmt=".2%") if hasattr(df_cost, "to_markdown") else df_cost)
    
    # ==========================================
    # 场景二：极端熊市生存测试 (Crisis Survival)
    # ==========================================
    logger.info("\n>>> 场景 2: 极端熊市生存测试")
    # 定义几个著名的至暗时刻
    crisis_periods = {
        "2018 Trade War": ("2018-01-01", "2018-12-31"),
        "2022 Fed Hike":  ("2022-01-01", "2022-12-31"),
        "2024 Liquidity": ("2024-01-01", "2024-02-29") # 微盘股崩盘
    }
    
    crisis_results = []
    
    for name, (start, end) in crisis_periods.items():
        # 检查数据是否覆盖该时间段
        if pred_df["date"].min() > pd.to_datetime(end):
            logger.warning(f"数据不足，跳过 {name}")
            continue
            
        logger.info(f"Testing Crisis: {name} ({start} ~ {end}) ...")
        out_path = os.path.join(report_dir, f"crisis_{name.replace(' ', '_')}")
        
        # 使用默认成本进行测试
        metrics = backtester.run(signal_df, output_dir=out_path, 
                               start_date=start, end_date=end)
        
        crisis_results.append({
            "Crisis Name": name,
            "Period": f"{start}~{end}",
            "Strategy Ret": metrics["annual_return"], # 注意这里是年化，短期可能不准，看回撤更重要
            "Max Drawdown": metrics["max_drawdown"]
        })

    df_crisis = pd.DataFrame(crisis_results)
    print("\n--- 熊市生存报告 ---")
    print(df_crisis.to_markdown(index=False, floatfmt=".2%") if hasattr(df_crisis, "to_markdown") else df_crisis)
    
    logger.info(f"\n压力测试完成！详细图表已保存至: {report_dir}")

if __name__ == "__main__":
    main()