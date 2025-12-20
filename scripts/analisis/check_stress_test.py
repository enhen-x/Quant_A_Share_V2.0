# scripts/check_stress_test.py

import os
import sys
import pandas as pd
import numpy as np
import datetime

# Matplotlib 字体配置（必须在 import pyplot 之前设置）
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False  # 使用 ASCII 减号代替 Unicode 减号
matplotlib.rcParams['mathtext.fontset'] = 'stix'   # 数学字体集
matplotlib.rcParams['font.family'] = 'sans-serif'

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
    
    # [Restored] 预先加载并切割好 benchmark
    idx_code = GLOBAL_CONFIG.get("preprocessing", {}).get("labels", {}).get("index_code", "000300.SH")
    idx_file = os.path.join(GLOBAL_CONFIG["paths"]["data_raw"], f"index_{idx_code.replace('.', '')}.parquet")
    benchmark_series = None
    if os.path.exists(idx_file):
        idx_df = read_parquet(idx_file)
        idx_df["date"] = pd.to_datetime(idx_df["date"])
        idx_df = idx_df.set_index("date")["close"].sort_index()
        benchmark_series = idx_df
    
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

    # --- 绘制成本敏感性汇总对比图 (Log Scale + Benchmark) ---
    if equity_curves:
        plt.figure(figsize=(12, 8))
        
        # 1. 确定统一的时间范围
        # 取第一个非空曲线的时间索引
        first_curve = next(iter(equity_curves.values()))
        start_date = first_curve.index[0]
        end_date = first_curve.index[-1]
        
        # 2. 绘制基准 (Benchmark)
        if benchmark_series is not None:
            # 截取对应时间段
            bench_slice = benchmark_series.truncate(before=start_date, after=end_date)
            if not bench_slice.empty:
                # 归一化：从 1.0 开始
                bench_norm = bench_slice / bench_slice.iloc[0]
                bench_norm.plot(label=f"Benchmark ({idx_code})", color="black", linestyle="--", linewidth=2, alpha=0.6)

        # 3. 绘制不同 Cost 的策略曲线
        # 使用不同颜色的渐变或区别度高的颜色
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(equity_curves)))
        
        for i, (label, curve) in enumerate(equity_curves.items()):
            curve.plot(label=label, linewidth=2, alpha=0.9, color=colors[i])
            
        plt.title("Cost Sensitivity Analysis: Strategy Equity Curves (Log Scale)")
        plt.xlabel("Date")
        plt.ylabel("Equity (Log Scale)")
        plt.grid(True, linestyle="--", alpha=0.5, which='both') # box grid
        plt.legend(loc="upper left")
        plt.yscale('log') # 设置对数坐标
        
        # 调整 y 轴显示的 format，避免纯科学计数法
        from matplotlib.ticker import ScalarFormatter
        plt.gca().yaxis.set_major_formatter(ScalarFormatter())
        
        plt.tight_layout()
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
    # [优化] 仅选择数据覆盖范围内 (2019-2025) 的重大市场风险事件
    crisis_periods = {
        "2020 Covid-19":  ("2020-01-20", "2020-03-31"), # 疫情爆发
        "2021 Bubble Burst": ("2021-02-18", "2021-12-31"), # 核心资产泡沫破裂
        "2022 Fed Hike":  ("2022-01-01", "2022-12-31"), # 美联储加息+俄乌战争
        "2023-2024 Bear": ("2023-01-01", "2024-02-05")  # 长期阴跌+微盘股流动性危机
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