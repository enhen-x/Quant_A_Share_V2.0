# scripts/analisis/analyze_threshold.py
"""
分类阈值分析脚本

分析不同分类阈值下的样本分布，为双头模型的分类标签选择最佳阈值。
同时分析绝对涨幅和相对涨幅（超额收益）两种模式。

输出:
1. 收益率分布直方图（绝对/相对）
2. 不同阈值下的正负样本比例
3. 建议阈值（使正负样本接近平衡）
"""

import os
import sys
import pandas as pd
import numpy as np

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

logger = get_logger()


def load_label_data():
    """加载已计算好的标签数据"""
    data_path = os.path.join(GLOBAL_CONFIG["paths"]["data_processed"], "all_stocks.parquet")
    
    if not os.path.exists(data_path):
        logger.error(f"未找到数据文件: {data_path}")
        logger.info("请先运行 python scripts/run_pipeline.py 生成特征和标签")
        return None
    
    df = read_parquet(data_path)
    df["date"] = pd.to_datetime(df["date"])
    
    logger.info(f"已加载数据: {len(df):,} 条, 时间范围: {df['date'].min()} ~ {df['date'].max()}")
    return df


def calculate_absolute_return(df: pd.DataFrame, horizon: int = 4) -> pd.Series:
    """计算绝对收益率"""
    # 使用 VWAP 作为价格基准
    if "amount" in df.columns and "volume" in df.columns:
        vwap = df["amount"] / (df["volume"] + 1e-8)
        price = vwap.where(df["volume"] > 0, df["close"])
    else:
        price = df["close"]
    
    # 按股票分组计算收益
    df = df.sort_values(["symbol", "date"])
    
    # Entry: T+1, Exit: T+1+horizon
    entry_price = price.shift(-1)
    exit_price = price.shift(-(1 + horizon))
    
    raw_return = (exit_price / entry_price) - 1.0
    return raw_return


def analyze_distribution(returns: pd.Series, mode_name: str, output_dir: str):
    """分析收益率分布并绘图"""
    
    # 去除 NaN
    returns = returns.dropna()
    
    if len(returns) == 0:
        logger.warning(f"{mode_name} 数据为空")
        return None
    
    # 基础统计
    stats = {
        "mode": mode_name,
        "count": len(returns),
        "mean": returns.mean(),
        "std": returns.std(),
        "median": returns.median(),
        "q25": returns.quantile(0.25),
        "q75": returns.quantile(0.75),
        "min": returns.min(),
        "max": returns.max(),
    }
    
    logger.info(f"\n=== {mode_name} 分布统计 ===")
    logger.info(f"样本数: {stats['count']:,}")
    logger.info(f"均值: {stats['mean']:.4f} ({stats['mean']*100:.2f}%)")
    logger.info(f"中位数: {stats['median']:.4f} ({stats['median']*100:.2f}%)")
    logger.info(f"标准差: {stats['std']:.4f}")
    
    # 绘制分布直方图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图: 完整分布
    ax1 = axes[0]
    # 限制范围以便观察主体分布
    plot_range = (-0.3, 0.3)
    returns_clipped = returns.clip(lower=plot_range[0], upper=plot_range[1])
    
    ax1.hist(returns_clipped, bins=100, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='阈值=0')
    ax1.axvline(x=returns.median(), color='green', linestyle='-', linewidth=2, label=f'中位数={returns.median():.4f}')
    ax1.set_xlabel(f'{mode_name} (%)')
    ax1.set_ylabel('样本数')
    ax1.set_title(f'{mode_name} 分布 (截断至 ±30%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 右图: 累积分布函数 (CDF)
    ax2 = axes[1]
    sorted_returns = np.sort(returns)
    cdf = np.arange(1, len(sorted_returns) + 1) / len(sorted_returns)
    
    # 只绘制核心范围
    mask = (sorted_returns >= -0.2) & (sorted_returns <= 0.2)
    ax2.plot(sorted_returns[mask], cdf[mask], linewidth=2, color='steelblue')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='阈值=0')
    ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel(f'{mode_name}')
    ax2.set_ylabel('累积概率')
    ax2.set_title(f'{mode_name} 累积分布函数 (CDF)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"distribution_{mode_name.replace(' ', '_')}.png")
    plt.savefig(fig_path, dpi=120)
    plt.close()
    logger.info(f"分布图已保存: {fig_path}")
    
    return stats


def analyze_thresholds(returns: pd.Series, mode_name: str, output_dir: str):
    """分析不同阈值下的样本分布"""
    
    returns = returns.dropna()
    total = len(returns)
    
    if total == 0:
        return None, None
    
    # 测试的阈值范围
    thresholds = np.arange(-0.05, 0.051, 0.005)
    
    results = []
    for th in thresholds:
        positive = (returns > th).sum()
        negative = (returns <= th).sum()
        pos_ratio = positive / total
        neg_ratio = negative / total
        balance_score = 1 - abs(pos_ratio - 0.5) * 2  # 越接近0.5越好
        
        results.append({
            "threshold": th,
            "positive": positive,
            "negative": negative,
            "pos_ratio": pos_ratio,
            "neg_ratio": neg_ratio,
            "balance_score": balance_score
        })
    
    df_results = pd.DataFrame(results)
    
    # 找到最佳阈值（使正负样本最接近平衡）
    best_idx = df_results["balance_score"].idxmax()
    best_threshold = df_results.loc[best_idx, "threshold"]
    best_pos_ratio = df_results.loc[best_idx, "pos_ratio"]
    
    logger.info(f"\n=== {mode_name} 阈值分析 ===")
    logger.info(f"建议阈值: {best_threshold:.4f} ({best_threshold*100:.2f}%)")
    logger.info(f"该阈值下: 正样本 {best_pos_ratio:.2%}, 负样本 {1-best_pos_ratio:.2%}")
    
    # 绘制阈值分析图
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(df_results["threshold"] * 100, df_results["pos_ratio"] * 100, 
            'b-o', linewidth=2, markersize=4, label='正样本 (涨) 比例')
    ax.plot(df_results["threshold"] * 100, df_results["neg_ratio"] * 100, 
            'r-s', linewidth=2, markersize=4, label='负样本 (跌) 比例')
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% 平衡线')
    ax.axvline(x=best_threshold * 100, color='green', linestyle='-', linewidth=2, 
               label=f'建议阈值 = {best_threshold*100:.2f}%')
    
    ax.set_xlabel('分类阈值 (%)')
    ax.set_ylabel('样本比例 (%)')
    ax.set_title(f'{mode_name} - 不同阈值下的样本分布')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"threshold_analysis_{mode_name.replace(' ', '_')}.png")
    plt.savefig(fig_path, dpi=120)
    plt.close()
    logger.info(f"阈值分析图已保存: {fig_path}")
    
    # 保存详细数据
    csv_path = os.path.join(output_dir, f"threshold_details_{mode_name.replace(' ', '_')}.csv")
    df_results.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    return best_threshold, df_results


def generate_report(abs_stats, excess_stats, abs_threshold, excess_threshold, output_dir):
    """生成最终分析报告"""
    
    report_lines = [
        "# 分类阈值分析报告",
        "",
        f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## 1. 分析目的",
        "",
        "为双头模型的分类标签选择最佳阈值，使正负样本分布接近平衡（约50:50），",
        "避免类别不平衡导致模型偏向预测某一类。",
        "",
        "---",
        "",
        "## 2. 分析结果",
        "",
        "### 2.1 绝对涨幅模式 (absolute)",
        "",
        "| 指标 | 数值 |",
        "|------|------|",
    ]
    
    if abs_stats:
        report_lines.extend([
            f"| 样本数 | {abs_stats['count']:,} |",
            f"| 均值 | {abs_stats['mean']*100:.2f}% |",
            f"| 中位数 | {abs_stats['median']*100:.2f}% |",
            f"| 标准差 | {abs_stats['std']*100:.2f}% |",
            "",
            f"**建议阈值: `{abs_threshold:.4f}` ({abs_threshold*100:.2f}%)**",
        ])
    else:
        report_lines.append("| (无数据) | - |")
    
    report_lines.extend([
        "",
        "### 2.2 超额涨幅模式 (excess_index)",
        "",
        "| 指标 | 数值 |",
        "|------|------|",
    ])
    
    if excess_stats:
        report_lines.extend([
            f"| 样本数 | {excess_stats['count']:,} |",
            f"| 均值 | {excess_stats['mean']*100:.2f}% |",
            f"| 中位数 | {excess_stats['median']*100:.2f}% |",
            f"| 标准差 | {excess_stats['std']*100:.2f}% |",
            "",
            f"**建议阈值: `{excess_threshold:.4f}` ({excess_threshold*100:.2f}%)**",
        ])
    else:
        report_lines.append("| (无数据) | - |")
    
    report_lines.extend([
        "",
        "---",
        "",
        "## 3. 配置建议",
        "",
        "根据分析结果，建议在 `config/main.yaml` 中配置：",
        "",
        "```yaml",
        "model:",
        "  dual_head:",
        "    classification:",
        f"      # 绝对涨跌模式 (推荐)",
        f"      label_mode: \"absolute\"",
        f"      threshold: {abs_threshold if abs_threshold is not None else 0.0:.4f}",
        "",
        f"      # 或者使用超额涨跌模式",
        f"      # label_mode: \"excess_index\"",
        f"      # threshold: {excess_threshold if excess_threshold is not None else 0.0:.4f}",
        "```",
        "",
        "---",
        "",
        "## 4. 相关图表",
        "",
        "- `distribution_绝对涨幅.png` - 绝对收益分布图",
        "- `distribution_超额涨幅.png` - 超额收益分布图",
        "- `threshold_analysis_绝对涨幅.png` - 绝对涨幅阈值分析",
        "- `threshold_analysis_超额涨幅.png` - 超额涨幅阈值分析",
    ])
    
    report_path = os.path.join(output_dir, "threshold_analysis_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    logger.info(f"\n报告已保存: {report_path}")
    return report_path


def main():
    logger.info("=" * 60)
    logger.info("=== 分类阈值分析脚本 ===")
    logger.info("=" * 60)
    
    # 准备输出目录
    output_dir = os.path.join(GLOBAL_CONFIG["paths"]["reports"], "threshold_analysis")
    ensure_dir(output_dir)
    
    # 加载数据
    df = load_label_data()
    if df is None:
        return
    
    # 获取配置
    horizon = GLOBAL_CONFIG.get("preprocessing", {}).get("labels", {}).get("horizon", 4)
    logger.info(f"使用持有期: {horizon} 天")
    
    # ========================================
    # 1. 分析绝对涨幅
    # ========================================
    logger.info("\n" + "=" * 40)
    logger.info(">>> 分析绝对涨幅 (absolute)")
    logger.info("=" * 40)
    
    # 按股票分组计算绝对收益
    abs_returns_list = []
    for symbol, group in df.groupby("symbol"):
        group = group.sort_values("date")
        abs_ret = calculate_absolute_return(group, horizon)
        abs_returns_list.append(abs_ret)
    
    abs_returns = pd.concat(abs_returns_list).dropna()
    abs_stats = analyze_distribution(abs_returns, "绝对涨幅", output_dir)
    abs_threshold, _ = analyze_thresholds(abs_returns, "绝对涨幅", output_dir)
    
    # ========================================
    # 2. 分析超额涨幅 (使用已有的 label 列)
    # ========================================
    logger.info("\n" + "=" * 40)
    logger.info(">>> 分析超额涨幅 (excess_index)")
    logger.info("=" * 40)
    
    if "label" in df.columns:
        excess_returns = df["label"].dropna()
        excess_stats = analyze_distribution(excess_returns, "超额涨幅", output_dir)
        excess_threshold, _ = analyze_thresholds(excess_returns, "超额涨幅", output_dir)
    else:
        logger.warning("未找到 label 列，跳过超额涨幅分析")
        excess_stats = None
        excess_threshold = None
    
    # ========================================
    # 3. 生成报告
    # ========================================
    logger.info("\n" + "=" * 40)
    logger.info(">>> 生成分析报告")
    logger.info("=" * 40)
    
    report_path = generate_report(abs_stats, excess_stats, abs_threshold, excess_threshold, output_dir)
    
    # 打印最终结论
    print("\n" + "=" * 60)
    print(">>> 分析完成! 结论汇总:")
    print("=" * 60)
    
    if abs_threshold is not None:
        print(f"\n[绝对涨幅模式] 建议阈值: {abs_threshold:.4f} ({abs_threshold*100:.2f}%)")
    
    if excess_threshold is not None:
        print(f"[超额涨幅模式] 建议阈值: {excess_threshold:.4f} ({excess_threshold*100:.2f}%)")
    
    print(f"\n详细报告: {report_path}")
    print(f"图表目录: {output_dir}")


if __name__ == "__main__":
    main()
