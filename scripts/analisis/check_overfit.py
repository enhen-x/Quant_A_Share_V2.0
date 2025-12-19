# scripts/analisis/check_overfit.py
# ============================================================================
# 过拟合检测脚本 (Shuffle Test / Noise Injection Test)
# ============================================================================
#
# 【功能】
# 通过向预测结果添加噪音来检测模型是否存在"过拟合/背答案"现象。
# 健康的模型：加噪后收益平缓下降
# 过拟合模型：加噪后收益急剧崩溃
#
# 【使用方法】
# python scripts/analisis/check_overfit.py
# ============================================================================

import os
import sys
import pandas as pd
import numpy as np

# Matplotlib 字体配置（必须在 import pyplot 之前设置）
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

# ============================================================================
# 辅助函数
# ============================================================================

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
        return read_parquet(pred_path)
    return None


def add_noise_to_predictions(pred_df, noise_ratio):
    """
    向预测分数添加噪音
    
    :param pred_df: 包含 pred_score 的 DataFrame
    :param noise_ratio: 噪音比例 (0.05 = 5%)
    :return: 添加噪音后的 DataFrame 副本
    """
    noisy_df = pred_df.copy()
    
    # 计算噪音：噪音 = score * noise_ratio * random(-1, 1)
    noise = noisy_df["pred_score"].std() * noise_ratio * np.random.randn(len(noisy_df))
    noisy_df["pred_score"] = noisy_df["pred_score"] + noise
    
    return noisy_df


def generate_random_signals(pred_df):
    """
    生成完全随机的预测分数
    
    :param pred_df: 原始 DataFrame（用于保持结构）
    :return: 随机分数的 DataFrame 副本
    """
    random_df = pred_df.copy()
    random_df["pred_score"] = np.random.randn(len(random_df))
    return random_df


def run_single_test(pred_df, backtester, strategy, test_name, output_dir):
    """
    执行单项回测测试
    
    :return: dict 包含指标和净值曲线
    """
    logger.info(f">>> 执行测试: {test_name}")
    
    # 生成信号
    signal_df = strategy.generate(pred_df)
    
    if signal_df.empty:
        logger.warning(f"  {test_name}: 信号为空，跳过")
        return None
    
    # 回测
    out_path = os.path.join(output_dir, test_name.replace(" ", "_").replace("%", "pct"))
    metrics = backtester.run(signal_df, output_dir=out_path)
    
    logger.info(f"  {test_name}: 年化收益={metrics['annual_return']:.2%}, 夏普={metrics['sharpe']:.2f}, 最大回撤={metrics['max_drawdown']:.2%}")
    
    return {
        "name": test_name,
        "annual_return": metrics["annual_return"],
        "sharpe": metrics["sharpe"],
        "max_drawdown": metrics["max_drawdown"],
        "equity_curve": metrics.get("equity_curve")
    }


# ============================================================================
# 主程序
# ============================================================================

def main():
    logger.info("=" * 60)
    logger.info("=== 过拟合检测 (Shuffle Test / Noise Injection) ===")
    logger.info("=" * 60)
    
    # 1. 加载预测数据
    pred_df = get_latest_predictions()
    if pred_df is None:
        logger.error("未找到预测文件，请先运行 run_walkforward.py")
        return
    
    pred_df["date"] = pd.to_datetime(pred_df["date"])
    logger.info(f"预测数据: {len(pred_df)} 行, 日期范围: {pred_df['date'].min()} ~ {pred_df['date'].max()}")
    
    # 2. 初始化
    backtester = VectorBacktester()
    strategy = TopKSignalStrategy()
    
    # 如果未开启仓位管理，强制满仓测试
    if not GLOBAL_CONFIG["strategy"].get("position_control", {}).get("enable", False):
        strategy.min_score = -999.0
    
    report_dir = os.path.join(GLOBAL_CONFIG["paths"]["reports"], "overfit_test")
    ensure_dir(report_dir)
    
    # 设置随机种子以保证可复现
    np.random.seed(42)
    
    # 3. 执行测试
    test_results = []
    equity_curves = {}
    
    # 3.1 Baseline (原始预测)
    result = run_single_test(pred_df, backtester, strategy, "Baseline", report_dir)
    if result:
        test_results.append(result)
        equity_curves["Baseline"] = result["equity_curve"]
    
    # 3.2 Noise 5%
    noisy_5 = add_noise_to_predictions(pred_df, 0.05)
    result = run_single_test(noisy_5, backtester, strategy, "Noise 5%", report_dir)
    if result:
        test_results.append(result)
        equity_curves["Noise 5%"] = result["equity_curve"]
    
    # 3.3 Noise 10%
    noisy_10 = add_noise_to_predictions(pred_df, 0.10)
    result = run_single_test(noisy_10, backtester, strategy, "Noise 10%", report_dir)
    if result:
        test_results.append(result)
        equity_curves["Noise 10%"] = result["equity_curve"]
    
    # 3.4 Noise 20%
    noisy_20 = add_noise_to_predictions(pred_df, 0.20)
    result = run_single_test(noisy_20, backtester, strategy, "Noise 20%", report_dir)
    if result:
        test_results.append(result)
        equity_curves["Noise 20%"] = result["equity_curve"]
    
    # 3.5 Random Signal (完全随机)
    random_df = generate_random_signals(pred_df)
    result = run_single_test(random_df, backtester, strategy, "Random", report_dir)
    if result:
        test_results.append(result)
        equity_curves["Random"] = result["equity_curve"]
    
    # 4. 汇总对比图
    if equity_curves:
        plt.figure(figsize=(14, 8))
        
        colors = {
            "Baseline": "#2ecc71",   # 绿色（基准）
            "Noise 5%": "#3498db",   # 蓝色
            "Noise 10%": "#f39c12",  # 橙色
            "Noise 20%": "#e74c3c",  # 红色
            "Random": "#95a5a6"      # 灰色
        }
        
        for label, curve in equity_curves.items():
            if curve is not None:
                curve.plot(label=label, linewidth=2, alpha=0.85, color=colors.get(label, "black"))
        
        plt.title("过拟合检测: 噪音敏感性对比 (Overfit Detection: Noise Sensitivity)", fontsize=14)
        plt.xlabel("日期")
        plt.ylabel("净值 (对数坐标)")
        plt.yscale("log")
        plt.grid(True, linestyle="--", alpha=0.5, which="both")
        plt.legend(loc="upper left", fontsize=10)
        
        from matplotlib.ticker import ScalarFormatter
        plt.gca().yaxis.set_major_formatter(ScalarFormatter())
        
        plt.tight_layout()
        chart_path = os.path.join(report_dir, "overfit_test_comparison.png")
        plt.savefig(chart_path, dpi=150)
        plt.close()
        logger.info(f"对比图已保存: {chart_path}")
    
    # 5. 生成报告
    if test_results:
        df_results = pd.DataFrame([{
            "Test": r["name"],
            "Annual Return": r["annual_return"],
            "Sharpe": r["sharpe"],
            "Max Drawdown": r["max_drawdown"]
        } for r in test_results])
        
        print("\n" + "=" * 60)
        print("📊 过拟合检测报告 (Overfit Detection Report)")
        print("=" * 60)
        print(df_results.to_markdown(index=False, floatfmt=".2%") if hasattr(df_results, "to_markdown") else df_results)
        
        csv_path = os.path.join(report_dir, "overfit_test_results.csv")
        df_results.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.info(f"报告已保存: {csv_path}")
        
        # 6. 结论分析
        print("\n" + "-" * 60)
        print("🔍 诊断结论:")
        print("-" * 60)
        
        baseline_ret = test_results[0]["annual_return"]
        noise_5_ret = test_results[1]["annual_return"] if len(test_results) > 1 else 0
        noise_10_ret = test_results[2]["annual_return"] if len(test_results) > 2 else 0
        
        # 计算衰减率
        decay_5 = (baseline_ret - noise_5_ret) / abs(baseline_ret) if baseline_ret != 0 else 0
        decay_10 = (baseline_ret - noise_10_ret) / abs(baseline_ret) if baseline_ret != 0 else 0
        
        print(f"  - Baseline 年化收益: {baseline_ret:.2%}")
        print(f"  - 5% 噪音后收益: {noise_5_ret:.2%} (衰减 {decay_5:.1%})")
        print(f"  - 10% 噪音后收益: {noise_10_ret:.2%} (衰减 {decay_10:.1%})")
        
        # 判断是否过拟合
        if decay_5 > 0.5:  # 5%噪音导致超过50%收益衰减
            print("\n⚠️  [警告] 模型可能存在严重过拟合风险！")
            print("    5% 噪音就导致收益衰减超过 50%，说明模型对预测分数的微小变化极度敏感。")
            print("    建议：增加正则化、减少特征数量、使用更长的验证窗口。")
        elif decay_10 > 0.6:  # 10%噪音导致超过60%收益衰减
            print("\n⚠️  [注意] 模型存在一定过拟合倾向。")
            print("    10% 噪音导致收益衰减超过 60%，建议检查特征工程和模型复杂度。")
        else:
            print("\n✅  [良好] 模型对噪音的抵抗力较强，过拟合风险较低。")
        
        print("-" * 60)
    
    logger.info(f"\n过拟合检测完成！报告已保存至: {report_dir}")


if __name__ == "__main__":
    main()
