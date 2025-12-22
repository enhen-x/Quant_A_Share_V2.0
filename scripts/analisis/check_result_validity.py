# scripts/analisis/check_result_validity.py
# ============================================================================
# 回测结果有效性验证 (Result Validity Checker)
# ============================================================================
#
# 【功能】
# 检查回测结果是否存在潜在问题：
#   1. 市值/流动性偏差 - 是否只选小盘股
#   2. 成本敏感性 - 高成本下是否还盈利
#   3. 分年度收益 - 收益是否集中在某些年份
#   4. 特征泄露检查 - 是否存在未来函数
#
# 【使用方法】
# python scripts/analisis/check_result_validity.py
# ============================================================================

import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict

# Matplotlib 字体配置
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt

# 路径适配
current_dir = os.path.dirname(os.path.abspath(__file__))
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
        return None, None
    
    subdirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    if not subdirs:
        return None, None
    
    subdirs.sort(reverse=True)
    latest_dir = subdirs[0]
    pred_path = os.path.join(models_dir, latest_dir, "predictions.parquet")
    
    if os.path.exists(pred_path):
        logger.info(f"使用预测文件: {pred_path}")
        return read_parquet(pred_path), latest_dir
    return None, None


def load_stock_data():
    """加载股票行情数据"""
    data_path = os.path.join(GLOBAL_CONFIG["paths"]["data_processed"], "all_stocks.parquet")
    if os.path.exists(data_path):
        return read_parquet(data_path)
    return None


def analyze_market_cap_distribution(pred_df, signal_df, stock_df, report_dir):
    """分析选股组合的市值分布"""
    print("\n" + "=" * 60)
    print("📊 市值分布分析 (Market Cap Distribution)")
    print("=" * 60)
    
    # 获取被选中的股票
    selected_stocks = signal_df[["date", "symbol"]].copy()
    selected_stocks["date"] = pd.to_datetime(selected_stocks["date"])
    
    # 合并市值数据
    if "mcap" in stock_df.columns or "feat_mcap" in stock_df.columns:
        mcap_col = "mcap" if "mcap" in stock_df.columns else "feat_mcap"
        stock_df["date"] = pd.to_datetime(stock_df["date"])
        
        merged = selected_stocks.merge(
            stock_df[["date", "symbol", mcap_col]], 
            on=["date", "symbol"], 
            how="left"
        )
        
        # 市值分位数
        mcap_values = merged[mcap_col].dropna() / 1e8  # 转换为亿元
        
        print(f"\n  选股组合市值统计 (单位: 亿元):")
        print(f"  ┌─────────────────────────────┐")
        print(f"  │ 最小市值:    {mcap_values.min():>10.2f} 亿 │")
        print(f"  │ 25%分位:     {mcap_values.quantile(0.25):>10.2f} 亿 │")
        print(f"  │ 中位数:      {mcap_values.median():>10.2f} 亿 │")
        print(f"  │ 平均值:      {mcap_values.mean():>10.2f} 亿 │")
        print(f"  │ 75%分位:     {mcap_values.quantile(0.75):>10.2f} 亿 │")
        print(f"  │ 最大市值:    {mcap_values.max():>10.2f} 亿 │")
        print(f"  └─────────────────────────────┘")
        
        # 市值分层统计
        small_cap = (mcap_values < 50).sum()
        mid_cap = ((mcap_values >= 50) & (mcap_values < 200)).sum()
        large_cap = (mcap_values >= 200).sum()
        total = len(mcap_values)
        
        print(f"\n  市值分层:")
        print(f"    小盘股 (<50亿):   {small_cap:>5} 次  ({small_cap/total*100:.1f}%)")
        print(f"    中盘股 (50-200亿): {mid_cap:>5} 次  ({mid_cap/total*100:.1f}%)")
        print(f"    大盘股 (>200亿):  {large_cap:>5} 次  ({large_cap/total*100:.1f}%)")
        
        # 警告
        if small_cap / total > 0.7:
            print(f"\n  ⚠️  [警告] 选股组合过度偏向小盘股 ({small_cap/total*100:.1f}%)")
            print(f"      小盘股流动性差，实盘可能无法按预期价格成交")
        
        # 绘图
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(mcap_values, bins=50, edgecolor='black', alpha=0.7, color='#3498db')
        ax.axvline(mcap_values.median(), color='red', linestyle='--', linewidth=2, 
                   label=f'中位数: {mcap_values.median():.1f}亿')
        ax.set_xlabel("流通市值 (亿元)")
        ax.set_ylabel("频次")
        ax.set_title("选股组合市值分布")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        chart_path = os.path.join(report_dir, "market_cap_distribution.png")
        plt.savefig(chart_path, dpi=150)
        plt.close()
        logger.info(f"市值分布图已保存: {chart_path}")
        
        return mcap_values.median()
    else:
        print("  [跳过] 数据中未找到市值列")
        return None


def analyze_cost_sensitivity(pred_df, report_dir):
    """分析不同成本下的回测表现"""
    print("\n" + "=" * 60)
    print("💰 成本敏感性分析 (Cost Sensitivity)")
    print("=" * 60)
    
    pred_df = pred_df.copy()
    pred_df["date"] = pd.to_datetime(pred_df["date"])
    
    strategy = TopKSignalStrategy()
    if not GLOBAL_CONFIG["strategy"].get("position_control", {}).get("enable", False):
        strategy.min_score = -999.0
    
    signal_df = strategy.generate(pred_df)
    
    cost_rates = [0.001, 0.002, 0.003, 0.005, 0.008, 0.01]
    results = []
    
    backtester = VectorBacktester()
    
    for cost in cost_rates:
        out_path = os.path.join(report_dir, f"cost_{int(cost*1000)}bps")
        metrics = backtester.run(signal_df, output_dir=out_path, cost_rate=cost)
        
        results.append({
            "cost_bps": cost * 10000,
            "annual_return": metrics["annual_return"],
            "sharpe": metrics["sharpe"],
            "max_drawdown": metrics["max_drawdown"]
        })
        print(f"  成本 {cost*100:.1f}%: 年化={metrics['annual_return']*100:.1f}%, 夏普={metrics['sharpe']:.2f}")
    
    df_results = pd.DataFrame(results)
    
    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1 = axes[0]
    ax1.plot(df_results["cost_bps"], df_results["annual_return"] * 100, 'b-o', linewidth=2, markersize=8)
    ax1.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax1.set_xlabel("交易成本 (基点 bps)")
    ax1.set_ylabel("年化收益率 (%)")
    ax1.set_title("年化收益 vs 交易成本")
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[1]
    ax2.plot(df_results["cost_bps"], df_results["sharpe"], 'g-o', linewidth=2, markersize=8)
    ax2.axhline(1, color='red', linestyle='--', alpha=0.5, label="夏普=1 门槛")
    ax2.set_xlabel("交易成本 (基点 bps)")
    ax2.set_ylabel("夏普比率")
    ax2.set_title("夏普比率 vs 交易成本")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle("成本敏感性分析", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    chart_path = os.path.join(report_dir, "cost_sensitivity.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    logger.info(f"成本敏感性图已保存: {chart_path}")
    
    # 找到盈亏平衡点
    for r in results:
        if r["annual_return"] <= 0:
            print(f"\n  ⚠️  盈亏平衡点: 成本约 {r['cost_bps']:.0f} 基点 ({r['cost_bps']/100:.2f}%)")
            break
    else:
        print(f"\n  ✅  即使成本达到 100 基点 (1%)，策略仍然盈利")
    
    return df_results


def analyze_yearly_returns(pred_df, report_dir):
    """分年度收益分析"""
    print("\n" + "=" * 60)
    print("📅 分年度收益分析 (Yearly Returns)")
    print("=" * 60)
    
    pred_df = pred_df.copy()
    pred_df["date"] = pd.to_datetime(pred_df["date"])
    pred_df["year"] = pred_df["date"].dt.year
    
    years = sorted(pred_df["year"].unique())
    
    strategy = TopKSignalStrategy()
    if not GLOBAL_CONFIG["strategy"].get("position_control", {}).get("enable", False):
        strategy.min_score = -999.0
    
    backtester = VectorBacktester()
    results = []
    
    for year in years:
        year_df = pred_df[pred_df["year"] == year].copy()
        if len(year_df) < 50:
            continue
        
        signal_df = strategy.generate(year_df)
        if signal_df.empty:
            continue
        
        out_path = os.path.join(report_dir, f"year_{year}")
        try:
            metrics = backtester.run(signal_df, output_dir=out_path)
            results.append({
                "year": year,
                "annual_return": metrics["annual_return"],
                "sharpe": metrics["sharpe"],
                "max_drawdown": metrics["max_drawdown"],
                "trades": len(signal_df)
            })
            print(f"  {year}: 年化={metrics['annual_return']*100:>7.1f}%, 夏普={metrics['sharpe']:>5.2f}, 回撤={metrics['max_drawdown']*100:>6.1f}%")
        except Exception as e:
            print(f"  {year}: [跳过] {e}")
    
    if not results:
        return None
    
    df_results = pd.DataFrame(results)
    
    # 绘图
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#2ecc71' if r > 0 else '#e74c3c' for r in df_results["annual_return"]]
    bars = ax.bar(df_results["year"].astype(str), df_results["annual_return"] * 100, color=colors, edgecolor='black')
    
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axhline(df_results["annual_return"].mean() * 100, color='blue', linestyle='--', 
               label=f'平均: {df_results["annual_return"].mean()*100:.1f}%')
    
    ax.set_xlabel("年份")
    ax.set_ylabel("年化收益率 (%)")
    ax.set_title("分年度收益表现")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, val in zip(bars, df_results["annual_return"] * 100):
        height = bar.get_height()
        ax.annotate(f'{val:.0f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    chart_path = os.path.join(report_dir, "yearly_returns.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    logger.info(f"分年度收益图已保存: {chart_path}")
    
    # 分析
    best_year = df_results.loc[df_results["annual_return"].idxmax()]
    worst_year = df_results.loc[df_results["annual_return"].idxmin()]
    
    print(f"\n  最佳年份: {int(best_year['year'])} ({best_year['annual_return']*100:.1f}%)")
    print(f"  最差年份: {int(worst_year['year'])} ({worst_year['annual_return']*100:.1f}%)")
    print(f"  平均年化: {df_results['annual_return'].mean()*100:.1f}%")
    print(f"  年化标准差: {df_results['annual_return'].std()*100:.1f}%")
    
    # 检查是否收益过于集中
    if best_year['annual_return'] > df_results['annual_return'].sum() * 0.5:
        print(f"\n  ⚠️  [警告] 收益过度集中在 {int(best_year['year'])} 年")
        print(f"      该年贡献了超过50%的累计收益，策略可能不够稳健")
    
    return df_results


def check_feature_leakage(stock_df):
    """检查特征是否存在未来数据泄露"""
    print("\n" + "=" * 60)
    print("🔍 特征泄露检查 (Feature Leakage Check)")
    print("=" * 60)
    
    if stock_df is None:
        print("  [跳过] 无法加载股票数据")
        return
    
    # 检查特征列
    feat_cols = [c for c in stock_df.columns if c.startswith("feat_")]
    print(f"\n  特征总数: {len(feat_cols)}")
    
    # 检查特征与标签的相关性
    if "label" in stock_df.columns:
        correlations = []
        for col in feat_cols:
            if stock_df[col].notna().sum() > 100:
                corr = stock_df[col].corr(stock_df["label"])
                correlations.append((col, abs(corr)))
        
        correlations.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n  特征与标签相关性 Top 10:")
        print(f"  {'特征名':<30} {'相关系数':>10}")
        print(f"  {'-'*40}")
        
        suspicious = []
        for col, corr in correlations[:10]:
            flag = " ⚠️" if corr > 0.5 else ""
            print(f"  {col:<30} {corr:>10.4f}{flag}")
            if corr > 0.5:
                suspicious.append(col)
        
        if suspicious:
            print(f"\n  ⚠️  [警告] 以下特征与标签相关性过高 (>0.5):")
            for col in suspicious:
                print(f"      - {col}")
            print(f"      这可能表示存在未来数据泄露，请检查特征计算逻辑")
        else:
            print(f"\n  ✅  未发现明显的特征泄露迹象")


def generate_summary_report(results, report_dir):
    """生成综合诊断报告"""
    print("\n" + "=" * 60)
    print("📋 综合诊断报告 (Summary Report)")
    print("=" * 60)
    
    warnings = []
    positives = []
    
    # 市值分析结论
    if results.get("median_mcap"):
        if results["median_mcap"] < 50:
            warnings.append(f"选股组合中位市值仅 {results['median_mcap']:.1f} 亿，偏向小盘股")
        else:
            positives.append(f"选股组合中位市值 {results['median_mcap']:.1f} 亿，流动性可接受")
    
    # 成本敏感性结论
    if results.get("cost_df") is not None:
        high_cost = results["cost_df"][results["cost_df"]["cost_bps"] == 50]
        if not high_cost.empty:
            ret_at_50bps = high_cost.iloc[0]["annual_return"]
            if ret_at_50bps > 0.2:
                positives.append(f"0.5% 成本下仍有 {ret_at_50bps*100:.1f}% 年化收益")
            elif ret_at_50bps > 0:
                warnings.append(f"0.5% 成本下收益降至 {ret_at_50bps*100:.1f}%，空间有限")
            else:
                warnings.append(f"0.5% 成本下已亏损，策略对成本极其敏感")
    
    # 年度收益结论
    if results.get("yearly_df") is not None:
        yearly = results["yearly_df"]
        if yearly["annual_return"].std() > 0.5:
            warnings.append(f"年度收益波动大 (标准差 {yearly['annual_return'].std()*100:.1f}%)，稳定性存疑")
        neg_years = (yearly["annual_return"] < 0).sum()
        if neg_years > 0:
            warnings.append(f"有 {neg_years} 年录得负收益")
        else:
            positives.append("历史上未出现年度亏损")
    
    # 输出结论
    if positives:
        print("\n  ✅ 积极信号:")
        for p in positives:
            print(f"     • {p}")
    
    if warnings:
        print("\n  ⚠️ 需关注的问题:")
        for w in warnings:
            print(f"     • {w}")
    
    # 最终建议
    print("\n  💡 建议:")
    if len(warnings) > len(positives):
        print("     模型结果存在多个可疑点，建议谨慎对待回测收益")
        print("     在实盘前应进行小资金测试，验证实际表现")
    else:
        print("     模型通过了基本验证，但仍需注意:")
        print("     1. 实盘资金量不宜过大 (建议 <50万)")
        print("     2. 关注小盘股流动性风险")
        print("     3. 定期监控策略表现衰减")
    
    print("=" * 60)


def main():
    logger.info("=" * 60)
    logger.info("=== 回测结果有效性验证 (Result Validity Checker) ===")
    logger.info("=" * 60)
    
    # 1. 加载数据
    pred_df, version = get_latest_predictions()
    if pred_df is None:
        logger.error("未找到预测文件，请先运行 run_walkforward.py")
        return
    
    pred_df["date"] = pd.to_datetime(pred_df["date"])
    logger.info(f"预测数据: {len(pred_df)} 行, 日期范围: {pred_df['date'].min()} ~ {pred_df['date'].max()}")
    
    stock_df = load_stock_data()
    
    # 生成信号
    strategy = TopKSignalStrategy()
    if not GLOBAL_CONFIG["strategy"].get("position_control", {}).get("enable", False):
        strategy.min_score = -999.0
    signal_df = strategy.generate(pred_df)
    
    # 输出目录
    report_dir = os.path.join(GLOBAL_CONFIG["paths"]["reports"], "validity_check")
    ensure_dir(report_dir)
    
    results = {}
    
    # 2. 市值分布分析
    results["median_mcap"] = analyze_market_cap_distribution(pred_df, signal_df, stock_df, report_dir)
    
    # 3. 成本敏感性分析
    results["cost_df"] = analyze_cost_sensitivity(pred_df, report_dir)
    
    # 4. 分年度收益分析
    results["yearly_df"] = analyze_yearly_returns(pred_df, report_dir)
    
    # 5. 特征泄露检查
    check_feature_leakage(stock_df)
    
    # 6. 生成综合报告
    generate_summary_report(results, report_dir)
    
    logger.info(f"\n验证完成！报告已保存至: {report_dir}")


if __name__ == "__main__":
    main()
