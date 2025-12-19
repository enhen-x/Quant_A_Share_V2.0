# scripts/analisis/check_return_distribution.py
# ============================================================================
# 每日收益分布诊断脚本 (Daily Return Distribution Diagnostic)
# ============================================================================
#
# 【功能】
# 对回测产生的信号进行全面诊断，检测潜在问题：
# - 收益分布的正态性
# - 极端收益（尾部风险）
# - 日历效应（周效应、月效应）
# - 连续亏损/盈利分析
# - 信号质量诊断
#
# 【使用方法】
# python scripts/analisis/check_return_distribution.py
# ============================================================================

import os
import sys
import pandas as pd
import numpy as np
from scipy import stats

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
        return None
    
    subdirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    if not subdirs:
        return None
    
    subdirs.sort(reverse=True)
    latest_dir = subdirs[0]
    pred_path = os.path.join(models_dir, latest_dir, "predictions.parquet")
    
    if os.path.exists(pred_path):
        logger.info(f"使用预测文件: {pred_path}")
        return read_parquet(pred_path), latest_dir
    return None, None


def analyze_return_distribution(daily_returns, report_dir):
    """分析收益分布特征"""
    
    returns = daily_returns.dropna()
    
    # 基础统计
    mean_ret = returns.mean()
    std_ret = returns.std()
    skewness = stats.skew(returns)
    kurtosis = stats.kurtosis(returns)
    
    # 正态性检验 (Jarque-Bera)
    jb_stat, jb_pvalue = stats.jarque_bera(returns)
    
    # 收益分位数
    quantiles = returns.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    
    # 极端收益统计
    extreme_up = (returns > 0.05).sum()  # 单日涨幅超5%
    extreme_down = (returns < -0.05).sum()  # 单日跌幅超5%
    
    print("\n" + "=" * 60)
    print("📈 收益分布统计 (Return Distribution)")
    print("=" * 60)
    print(f"  样本数量: {len(returns)}")
    print(f"  日均收益: {mean_ret:.4%}")
    print(f"  收益标准差: {std_ret:.4%}")
    print(f"  偏度 (Skewness): {skewness:.4f}  {'[右偏]' if skewness > 0.5 else '[左偏]' if skewness < -0.5 else '[正常]'}")
    print(f"  峰度 (Kurtosis): {kurtosis:.4f}  {'[肥尾]' if kurtosis > 3 else '[瘦尾]' if kurtosis < -1 else '[正常]'}")
    print(f"\n  ┌─ 分位数 ─────────────────┐")
    print(f"  │ 1%分位:  {quantiles[0.01]:>8.2%}       │")
    print(f"  │ 5%分位:  {quantiles[0.05]:>8.2%}       │")
    print(f"  │ 25%分位: {quantiles[0.25]:>8.2%}       │")
    print(f"  │ 中位数:  {quantiles[0.5]:>8.2%}       │")
    print(f"  │ 75%分位: {quantiles[0.75]:>8.2%}       │")
    print(f"  │ 95%分位: {quantiles[0.95]:>8.2%}       │")
    print(f"  │ 99%分位: {quantiles[0.99]:>8.2%}       │")
    print(f"  └────────────────────────┘")
    print(f"\n  极端收益统计:")
    print(f"    单日涨幅 > 5%: {extreme_up} 次 ({extreme_up/len(returns)*100:.2f}%)")
    print(f"    单日跌幅 > 5%: {extreme_down} 次 ({extreme_down/len(returns)*100:.2f}%)")
    
    # 正态性诊断
    print(f"\n  正态性检验 (Jarque-Bera):")
    print(f"    JB统计量: {jb_stat:.2f}")
    print(f"    P值: {jb_pvalue:.4f}")
    if jb_pvalue < 0.05:
        print("    ⚠️ 收益分布显著偏离正态分布")
    else:
        print("    ✅ 收益分布近似正态")
    
    # 绘制分布图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 收益直方图 + KDE
    ax1 = axes[0, 0]
    returns.hist(bins=50, density=True, alpha=0.7, ax=ax1, color='steelblue', edgecolor='white')
    x = np.linspace(returns.min(), returns.max(), 100)
    ax1.plot(x, stats.norm.pdf(x, mean_ret, std_ret), 'r-', linewidth=2, label='正态分布拟合')
    ax1.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax1.set_title('每日收益分布 (Histogram + Normal Fit)')
    ax1.set_xlabel('日收益率')
    ax1.set_ylabel('密度')
    ax1.legend()
    
    # 2. Q-Q 图
    ax2 = axes[0, 1]
    stats.probplot(returns, dist="norm", plot=ax2)
    ax2.set_title('Q-Q 图 (正态性检验)')
    
    # 3. 累计收益曲线
    ax3 = axes[1, 0]
    cumulative = (1 + returns).cumprod()
    cumulative.plot(ax=ax3, linewidth=1.5, color='green')
    ax3.set_title('累计收益曲线')
    ax3.set_xlabel('日期')
    ax3.set_ylabel('累计净值')
    ax3.grid(True, alpha=0.3)
    
    # 4. 滚动波动率
    ax4 = axes[1, 1]
    rolling_vol = returns.rolling(20).std() * np.sqrt(252)
    rolling_vol.plot(ax=ax4, linewidth=1.5, color='orange')
    ax4.axhline(rolling_vol.mean(), color='red', linestyle='--', label=f'平均年化波动率: {rolling_vol.mean():.1%}')
    ax4.set_title('20日滚动年化波动率')
    ax4.set_xlabel('日期')
    ax4.set_ylabel('年化波动率')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, "return_distribution.png"), dpi=150)
    plt.close()
    logger.info(f"收益分布图已保存")
    
    return {
        "mean": mean_ret,
        "std": std_ret,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "jb_pvalue": jb_pvalue
    }


def analyze_calendar_effects(daily_returns, report_dir):
    """分析日历效应（周效应、月效应）"""
    
    returns = daily_returns.dropna()
    returns_df = pd.DataFrame({'return': returns})
    returns_df['weekday'] = returns_df.index.dayofweek
    returns_df['month'] = returns_df.index.month
    
    # 周效应
    weekday_names = ['周一', '周二', '周三', '周四', '周五']
    weekday_stats = returns_df.groupby('weekday')['return'].agg(['mean', 'std', 'count'])
    weekday_stats.index = weekday_names[:len(weekday_stats)]
    
    # 月效应
    month_stats = returns_df.groupby('month')['return'].agg(['mean', 'std', 'count'])
    month_stats.index = [f'{i}月' for i in month_stats.index]
    
    print("\n" + "=" * 60)
    print("📅 日历效应分析 (Calendar Effects)")
    print("=" * 60)
    
    print("\n  周效应 (Weekday Effect):")
    print("  " + "-" * 40)
    for day, row in weekday_stats.iterrows():
        bar = "█" * int(abs(row['mean']) * 500)
        sign = "+" if row['mean'] > 0 else ""
        print(f"  {day}: {sign}{row['mean']:.3%} ± {row['std']:.3%}  {bar}")
    
    # 检测显著的周效应
    best_day = weekday_stats['mean'].idxmax()
    worst_day = weekday_stats['mean'].idxmin()
    print(f"\n    最佳交易日: {best_day} ({weekday_stats.loc[best_day, 'mean']:.3%})")
    print(f"    最差交易日: {worst_day} ({weekday_stats.loc[worst_day, 'mean']:.3%})")
    
    print("\n  月效应 (Monthly Effect):")
    print("  " + "-" * 40)
    best_month = month_stats['mean'].idxmax()
    worst_month = month_stats['mean'].idxmin()
    print(f"    最佳月份: {best_month} ({month_stats.loc[best_month, 'mean']:.3%})")
    print(f"    最差月份: {worst_month} ({month_stats.loc[worst_month, 'mean']:.3%})")
    
    # 绘制日历效应图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 周效应柱状图
    colors = ['green' if x > 0 else 'red' for x in weekday_stats['mean']]
    weekday_stats['mean'].plot(kind='bar', ax=axes[0], color=colors, edgecolor='white')
    axes[0].set_title('周效应: 各交易日平均收益')
    axes[0].set_xlabel('')
    axes[0].set_ylabel('平均日收益')
    axes[0].axhline(0, color='black', linewidth=0.5)
    axes[0].tick_params(axis='x', rotation=0)
    
    # 月效应柱状图
    colors = ['green' if x > 0 else 'red' for x in month_stats['mean']]
    month_stats['mean'].plot(kind='bar', ax=axes[1], color=colors, edgecolor='white')
    axes[1].set_title('月效应: 各月平均收益')
    axes[1].set_xlabel('')
    axes[1].set_ylabel('平均日收益')
    axes[1].axhline(0, color='black', linewidth=0.5)
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, "calendar_effects.png"), dpi=150)
    plt.close()
    logger.info(f"日历效应图已保存")
    
    return weekday_stats, month_stats


def analyze_streak_patterns(daily_returns, report_dir):
    """分析连续盈亏模式"""
    
    returns = daily_returns.dropna()
    
    # 计算连续盈亏
    is_positive = (returns > 0).astype(int)
    
    # 找出所有连续序列
    win_streaks = []
    loss_streaks = []
    
    current_streak = 1
    for i in range(1, len(is_positive)):
        if is_positive.iloc[i] == is_positive.iloc[i-1]:
            current_streak += 1
        else:
            if is_positive.iloc[i-1] == 1:
                win_streaks.append(current_streak)
            else:
                loss_streaks.append(current_streak)
            current_streak = 1
    
    # 最后一个序列
    if is_positive.iloc[-1] == 1:
        win_streaks.append(current_streak)
    else:
        loss_streaks.append(current_streak)
    
    print("\n" + "=" * 60)
    print("📊 连续盈亏分析 (Streak Analysis)")
    print("=" * 60)
    
    win_rate = (returns > 0).mean()
    avg_win = returns[returns > 0].mean()
    avg_loss = returns[returns < 0].mean()
    
    print(f"  胜率: {win_rate:.2%}")
    print(f"  平均盈利: {avg_win:.3%}")
    print(f"  平均亏损: {avg_loss:.3%}")
    print(f"  盈亏比: {abs(avg_win/avg_loss):.2f}")
    
    if win_streaks:
        print(f"\n  连续盈利:")
        print(f"    最长连胜: {max(win_streaks)} 天")
        print(f"    平均连胜: {np.mean(win_streaks):.1f} 天")
    
    if loss_streaks:
        print(f"\n  连续亏损:")
        print(f"    最长连亏: {max(loss_streaks)} 天")
        print(f"    平均连亏: {np.mean(loss_streaks):.1f} 天")
    
    # 最大单日涨跌
    print(f"\n  极值统计:")
    print(f"    最大单日涨幅: {returns.max():.2%} ({returns.idxmax().strftime('%Y-%m-%d')})")
    print(f"    最大单日跌幅: {returns.min():.2%} ({returns.idxmin().strftime('%Y-%m-%d')})")
    
    # 绘制连续盈亏分布
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    if win_streaks:
        axes[0].hist(win_streaks, bins=range(1, max(win_streaks)+2), color='green', 
                     alpha=0.7, edgecolor='white', align='left')
        axes[0].set_title('连续盈利分布')
        axes[0].set_xlabel('连续盈利天数')
        axes[0].set_ylabel('频次')
    
    if loss_streaks:
        axes[1].hist(loss_streaks, bins=range(1, max(loss_streaks)+2), color='red', 
                     alpha=0.7, edgecolor='white', align='left')
        axes[1].set_title('连续亏损分布')
        axes[1].set_xlabel('连续亏损天数')
        axes[1].set_ylabel('频次')
    
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, "streak_patterns.png"), dpi=150)
    plt.close()
    logger.info(f"连续盈亏图已保存")
    
    return {
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "max_win_streak": max(win_streaks) if win_streaks else 0,
        "max_loss_streak": max(loss_streaks) if loss_streaks else 0
    }


def diagnose_issues(dist_stats, streak_stats):
    """综合诊断并给出建议"""
    
    issues = []
    warnings = []
    
    # 1. 检查偏度
    if dist_stats['skewness'] < -0.5:
        issues.append("收益分布左偏严重，存在较大的负向尾部风险")
    elif dist_stats['skewness'] > 1.0:
        warnings.append("收益分布右偏明显，可能存在少数极端盈利主导业绩")
    
    # 2. 检查峰度
    if dist_stats['kurtosis'] > 5:
        issues.append("收益分布呈现肥尾特征，极端事件风险较高")
    
    # 3. 检查胜率和盈亏比
    if streak_stats['win_rate'] < 0.45:
        issues.append(f"胜率偏低 ({streak_stats['win_rate']:.1%})，需要较高盈亏比来弥补")
    
    profit_loss_ratio = abs(streak_stats['avg_win'] / streak_stats['avg_loss'])
    if profit_loss_ratio < 1.2:
        warnings.append(f"盈亏比较低 ({profit_loss_ratio:.2f})，风险调整后收益可能不稳定")
    
    # 4. 检查连续亏损
    if streak_stats['max_loss_streak'] > 10:
        issues.append(f"最大连亏 {streak_stats['max_loss_streak']} 天，需关注资金管理")
    
    print("\n" + "=" * 60)
    print("🔍 综合诊断报告 (Diagnostic Summary)")
    print("=" * 60)
    
    if not issues and not warnings:
        print("\n✅ 恭喜！未发现明显问题，策略表现健康。")
    else:
        if issues:
            print("\n⚠️ 需要关注的问题:")
            for i, issue in enumerate(issues, 1):
                print(f"   {i}. {issue}")
        
        if warnings:
            print("\n📋 温和建议:")
            for i, warning in enumerate(warnings, 1):
                print(f"   {i}. {warning}")
    
    print("=" * 60)


def main():
    logger.info("=" * 60)
    logger.info("=== 每日收益分布诊断 (Return Distribution Diagnostic) ===")
    logger.info("=" * 60)
    
    # 1. 加载预测数据
    pred_df, version = get_latest_predictions()
    if pred_df is None:
        logger.error("未找到预测文件，请先运行 run_walkforward.py")
        return
    
    pred_df["date"] = pd.to_datetime(pred_df["date"])
    
    # 2. 生成信号并回测
    strategy = TopKSignalStrategy()
    if not GLOBAL_CONFIG["strategy"].get("position_control", {}).get("enable", False):
        strategy.min_score = -999.0
    
    signal_df = strategy.generate(pred_df)
    
    if signal_df.empty:
        logger.error("信号生成为空")
        return
    
    # 3. 运行回测获取日收益
    report_dir = os.path.join(GLOBAL_CONFIG["paths"]["reports"], "return_diagnostic")
    ensure_dir(report_dir)
    
    backtester = VectorBacktester()
    metrics = backtester.run(signal_df, output_dir=report_dir)
    
    # 4. 获取日收益序列
    daily_returns = metrics.get("equity_curve")
    if daily_returns is None:
        logger.error("回测未返回净值曲线")
        return
    
    # 转换为日收益率
    daily_returns = daily_returns.pct_change().dropna()
    
    # 5. 执行各项分析
    dist_stats = analyze_return_distribution(daily_returns, report_dir)
    analyze_calendar_effects(daily_returns, report_dir)
    streak_stats = analyze_streak_patterns(daily_returns, report_dir)
    
    # 6. 综合诊断
    diagnose_issues(dist_stats, streak_stats)
    
    logger.info(f"\n诊断完成！报告已保存至: {report_dir}")


if __name__ == "__main__":
    main()
