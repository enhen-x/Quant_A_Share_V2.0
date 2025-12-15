# scripts/check_stress_test.py

import os
import sys
import pandas as pd
import datetime
import matplotlib.pyplot as plt

# 路径适配
current_dir = os.path.dirname(os.path.abspath(__file__))
# 从当前文件位置 (scripts/analisis) 返回两级到项目根目录
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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
    
    # [优化] 如果未开启仓位管理，强制关闭最低分过滤，确保满仓
    if not GLOBAL_CONFIG["strategy"].get("position_control", {}).get("enable", False):
        logger.info("[Full Load] 检测到未启用动态仓位管理，强制 min_score = -999 以确保满仓测试")
        strategy.min_score = -999.0
        
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
    equity_curves = {} # 用于保存所有曲线以便画汇总图
    
    for c in costs:
        logger.info(f"Testing Cost = {c*1000:.1f}‰ ...")
        out_path = os.path.join(report_dir, f"cost_{int(c*10000)}")
    for c in costs:
        logger.info(f"Testing Cost = {c*1000:.1f}‰ ...")
        out_path = os.path.join(report_dir, f"cost_{int(c*10000)}")
        
        # 调用回测 (Backtester 已更新，会在 metrics 中返回 equity_curve)
        metrics = backtester.run(signal_df, output_dir=out_path, cost_rate=c)
        
        # 收集曲线数据
        if "equity_curve" in metrics:
            equity_curves[f"Cost {c*1000:.1f}‰"] = metrics["equity_curve"]
            
        cost_results.append({
            "Cost Rate": f"{c*1000:.1f}‰",
            "Ann Return": metrics["annual_return"],
            "Sharpe": metrics["sharpe"]
        })

    # --- 绘制成本敏感性汇总对比图 ---
    if equity_curves:
        plt.figure(figsize=(12, 7))
        
        # 绘制基准 (Benchmark) - 只画一次
        # 从最后一条曲线中尝试提取基准时间段（通常所有曲线时间轴一致）
        # 这里为了简单，我们直接画各个 Cost 的曲线，基准可画可不画，重点是看 Cost 的衰减
        
        for label, curve in equity_curves.items():
            curve.plot(label=label, linewidth=2, alpha=0.8)
            
        plt.title("Cost Sensitivity Analysis: Strategy Equity Curves")
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(loc="upper left")
        
        # 使用 Log Scale 小图嵌入或直接画 Log 轴？
        # 这里直接画线性轴为主，用户主要看净值差异
        plt.savefig(os.path.join(report_dir, "cost_sensitivity_comparison.png"), dpi=120)
        plt.close()

    df_cost = pd.DataFrame(cost_results)
    print("\n--- 成本敏感性报告 ---")
    print(df_cost.to_markdown(index=False, floatfmt=".2%") if hasattr(df_cost, "to_markdown") else df_cost)
    
    # 保存 csv
    df_cost.to_csv(os.path.join(report_dir, "cost_sensitivity.csv"), index=False, encoding='utf-8-sig')

    # ==========================================
    # 场景二：极端熊市生存测试 (Crisis Survival)
    # ==========================================
    logger.info("\n>>> 场景 2: 极端熊市生存测试")
    crisis_periods = {
        "2015 Crash":     ("2015-06-01", "2016-02-29"), # 股灾+熔断
        "2018 Trade War": ("2018-01-01", "2018-12-31"),
        "2022 Fed Hike":  ("2022-01-01", "2022-12-31"),
        "2024 Liquidity": ("2024-01-01", "2024-04-30")  # 微盘股崩盘(延长至4月)
    }
    
    crisis_results = []
    
    for name, (start, end) in crisis_periods.items():
        if pred_df["date"].min() > pd.to_datetime(end):
            logger.warning(f"数据不足，跳过 {name}")
            continue
            
        logger.info(f"Testing Crisis: {name} ({start} ~ {end}) ...")
        out_path = os.path.join(report_dir, f"crisis_{name.replace(' ', '_')}")
        
        metrics = backtester.run(signal_df, output_dir=out_path, 
                               start_date=start, end_date=end)
        
        crisis_results.append({
            "Crisis Name": name,
            "Period": f"{start}~{end}",
            "Strategy Ret": metrics["annual_return"], 
            "Max Drawdown": metrics["max_drawdown"]
        })

    df_crisis = pd.DataFrame(crisis_results)
    print("\n--- 熊市生存报告 ---")
    print(df_crisis.to_markdown(index=False, floatfmt=".2%") if hasattr(df_crisis, "to_markdown") else df_crisis)
    df_crisis.to_csv(os.path.join(report_dir, "crisis_test.csv"), index=False, encoding='utf-8-sig')
    
    logger.info(f"\n压力测试完成！详细图表已保存至: {report_dir}")

if __name__ == "__main__":
    main()