# scripts/check_stress_test.py

import os
import sys
import pandas as pd
import numpy as np
import datetime
import warnings
import logging

# 抑制 matplotlib 字体警告
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', message='.*glyph.*')
warnings.filterwarnings('ignore', message='.*Glyph.*')
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
logging.getLogger('matplotlib.backends').setLevel(logging.ERROR)

# Matplotlib 字体配置（必须在 import pyplot 之前设置）
import matplotlib
matplotlib.rcParams.update({
    'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif'],
    'font.family': 'sans-serif',
    'axes.unicode_minus': False,      # 使用 ASCII 减号代替 Unicode 减号 (\u2212)
    'mathtext.fontset': 'dejavusans', # 使用 DejaVu Sans 数学字体（支持更多符号）
    'text.usetex': False,             # 不使用 LaTeX 渲染
})

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


def ensure_min_duration(start_str, end_str, min_days=30):
    """
    确保时间段至少有 min_days 天（约1个月）
    如果不足，向前后各扩展一半的差额
    """
    start_dt = pd.to_datetime(start_str)
    end_dt = pd.to_datetime(end_str)
    duration = (end_dt - start_dt).days
    
    if duration < min_days:
        gap = min_days - duration
        extend_before = gap // 2
        extend_after = gap - extend_before
        start_dt = start_dt - pd.Timedelta(days=extend_before)
        end_dt = end_dt + pd.Timedelta(days=extend_after)
        logger.info(f"  时间段不足{min_days}天，已扩展至: {start_dt.strftime('%Y-%m-%d')} ~ {end_dt.strftime('%Y-%m-%d')}")
    
    return start_dt.strftime('%Y-%m-%d'), end_dt.strftime('%Y-%m-%d')


def _plot_crisis_summary(crisis_results, report_dir):
    """
    绘制所有危机时段的汇总对比图
    """
    if not crisis_results:
        return
    
    # 抑制绘图时的字体警告
    import io
    old_stderr = sys.stderr
    sys.stderr = io.StringIO()
    
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        df = pd.DataFrame(crisis_results)
        names = df["Crisis Name"].tolist()
        x = np.arange(len(names))
        
        # 子图1: 年化收益率
        ax1 = axes[0]
        colors1 = ['green' if v > 0 else 'red' for v in df["Strategy Ret"]]
        bars1 = ax1.bar(x, df["Strategy Ret"] * 100, color=colors1, alpha=0.7, edgecolor='black')
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax1.set_ylabel('Annual Return (%)', fontsize=11)
        ax1.set_title('Crisis Period: Strategy Annual Return', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
        ax1.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for bar, val in zip(bars1, df["Strategy Ret"]):
            height = bar.get_height()
            ax1.annotate(f'{val*100:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height >= 0 else -12),
                        textcoords="offset points",
                        ha='center', va='bottom' if height >= 0 else 'top',
                        fontsize=9, fontweight='bold')
        
        # 子图2: 最大回撤
        ax2 = axes[1]
        bars2 = ax2.bar(x, df["Max Drawdown"] * 100, color='darkred', alpha=0.7, edgecolor='black')
        ax2.axhline(y=-20, color='orange', linestyle='--', linewidth=1, label='Warning Line (-20%)')
        ax2.set_ylabel('Max Drawdown (%)', fontsize=11)
        ax2.set_title('Crisis Period: Max Drawdown', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
        ax2.grid(axis='y', alpha=0.3)
        ax2.legend(loc='lower right', fontsize=9)
        
        # 添加数值标签
        for bar, val in zip(bars2, df["Max Drawdown"]):
            height = bar.get_height()
            ax2.annotate(f'{val*100:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, -12),
                        textcoords="offset points",
                        ha='center', va='top',
                        fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(report_dir, "crisis_summary.png"), dpi=120)
        plt.close()
        
    finally:
        sys.stderr = old_stderr


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
    costs = [0.001, 0.002, 0.003, 0.005]  # 千1, 千2, 千3, 千5
    cost_results = []
    equity_curves = {}  # 用于保存所有曲线以便画汇总图
    
    # 预先加载并切割好 benchmark
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
        # 抑制字体警告
        import io
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        
        try:
            plt.figure(figsize=(12, 8))
            
            # 1. 确定统一的时间范围
            first_curve = next(iter(equity_curves.values()))
            start_date = first_curve.index[0]
            end_date = first_curve.index[-1]
            
            # 2. 绘制基准 (Benchmark)
            if benchmark_series is not None:
                bench_slice = benchmark_series.truncate(before=start_date, after=end_date)
                if not bench_slice.empty:
                    bench_norm = bench_slice / bench_slice.iloc[0]
                    bench_norm.plot(label=f"Benchmark ({idx_code})", color="black", linestyle="--", linewidth=2, alpha=0.6)

            # 3. 绘制不同 Cost 的策略曲线
            colors = plt.cm.viridis(np.linspace(0, 0.9, len(equity_curves)))
            
            for i, (label, curve) in enumerate(equity_curves.items()):
                curve.plot(label=label, linewidth=2, alpha=0.9, color=colors[i])
                
            plt.title("Cost Sensitivity Analysis: Strategy Equity Curves (Log Scale)")
            plt.xlabel("Date")
            plt.ylabel("Equity (Log Scale)")
            plt.grid(True, linestyle="--", alpha=0.5, which='both')
            plt.legend(loc="upper left")
            plt.yscale('log')
            
            from matplotlib.ticker import ScalarFormatter
            plt.gca().yaxis.set_major_formatter(ScalarFormatter())
            
            plt.tight_layout()
            plt.savefig(os.path.join(report_dir, "cost_sensitivity_comparison.png"), dpi=120)
            plt.close()
        finally:
            sys.stderr = old_stderr

    df_cost = pd.DataFrame(cost_results)
    print("\n--- 成本敏感性报告 ---")
    print(df_cost.to_markdown(index=False, floatfmt=".2%") if hasattr(df_cost, "to_markdown") else df_cost)
    
    # 保存 csv
    df_cost.to_csv(os.path.join(report_dir, "cost_sensitivity.csv"), index=False, encoding='utf-8-sig')

    # ==========================================
    # 场景二：极端熊市生存测试 (Crisis Survival)
    # ==========================================
    logger.info("\n>>> 场景 2: 极端熊市生存测试")
    # [优化] 仅选择数据覆盖范围内 (2017-2025) 的重大市场风险事件
    # [要求] 每个危机时间段至少1个月（约22个交易日）
    crisis_periods = {
        "2018 Trade War":    ("2018-01-29", "2018-12-31"),  # 中美贸易战+去杠杆
        "2020 Covid-19":     ("2020-01-02", "2020-03-31"),  # 疫情爆发（扩展到1月初）
        "2021 Bubble Burst": ("2021-02-18", "2021-05-31"),  # 核心资产泡沫破裂（缩短到3个月）
        "2022 Fed Hike":     ("2022-01-01", "2022-10-31"),  # 美联储加息+俄乌战争
        "2023-2024 Bear":    ("2023-01-01", "2024-02-05"),  # 长期阴跌
        "2024 Liquidity":    ("2024-01-02", "2024-02-29"),  # 微盘股流动性危机（扩展到1个月）
    }
    
    crisis_results = []
    
    for name, (start, end) in crisis_periods.items():
        # 检查数据覆盖范围
        if pred_df["date"].min() > pd.to_datetime(end):
            logger.warning(f"数据不足，跳过 {name}")
            continue
        if pred_df["date"].max() < pd.to_datetime(start):
            logger.warning(f"数据不足，跳过 {name}")
            continue
        
        # 确保时间段至少1个月
        start_adj, end_adj = ensure_min_duration(start, end, min_days=30)
        
        logger.info(f"Testing Crisis: {name} ({start_adj} ~ {end_adj}) ...")
        out_path = os.path.join(report_dir, f"crisis_{name.replace(' ', '_')}")
        ensure_dir(out_path)
        
        # 运行回测（backtester.run 内部会自动生成 daily_comparison.png）
        metrics = backtester.run(signal_df, output_dir=out_path, 
                               start_date=start_adj, end_date=end_adj)
        
        # 如果回测返回为空，跳过
        if not metrics or "annual_return" not in metrics:
            logger.warning(f"  {name} 回测结果为空，跳过")
            continue
        
        crisis_results.append({
            "Crisis Name": name,
            "Period": f"{start_adj}~{end_adj}",
            "Strategy Ret": metrics["annual_return"], 
            "Max Drawdown": metrics["max_drawdown"],
            "Sharpe": metrics.get("sharpe", 0)
        })

    df_crisis = pd.DataFrame(crisis_results)
    print("\n--- 熊市生存报告 ---")
    print(df_crisis.to_markdown(index=False, floatfmt=".2%") if hasattr(df_crisis, "to_markdown") else df_crisis)
    df_crisis.to_csv(os.path.join(report_dir, "crisis_test.csv"), index=False, encoding='utf-8-sig')
    
    # 绘制危机期间汇总对比图
    _plot_crisis_summary(crisis_results, report_dir)
    
    logger.info(f"\n压力测试完成！详细图表已保存至: {report_dir}")
    logger.info(f"每个危机时段的 daily_comparison.png 已保存在各自子目录中")


if __name__ == "__main__":
    main()