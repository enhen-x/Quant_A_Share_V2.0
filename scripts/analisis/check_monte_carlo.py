# scripts/analisis/check_monte_carlo.py
# ============================================================================
# 蒙特卡洛模拟分析 (Monte Carlo Simulation Analysis)
# ============================================================================
#
# 【功能】
# 对双头模型的预测结果进行蒙特卡洛模拟，评估策略的稳健性和置信区间。
# 包含4种模拟方法：
#   1. Bootstrap 重采样 - 评估收益分布置信区间
#   2. 权重扰动 - 评估融合权重敏感性
#   3. 噪音注入 - 评估模型抗干扰能力
#   4. 时间窗口采样 - 评估策略时间稳定性
#
# 【使用方法】
# python scripts/analisis/check_monte_carlo.py
# ============================================================================

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

# Matplotlib 字体配置（必须在 import pyplot 之前设置）
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

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
# 蒙特卡洛分析器
# ============================================================================

class MonteCarloAnalyzer:
    """
    蒙特卡洛模拟分析器
    
    对双头模型预测结果进行多种模拟分析，评估策略稳健性。
    """
    
    def __init__(self, n_simulations: int = 500, random_seed: int = 42):
        """
        初始化分析器
        
        Args:
            n_simulations: 模拟次数
            random_seed: 随机种子（保证可复现）
        """
        self.n_simulations = n_simulations
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        self.config = GLOBAL_CONFIG
        self.dual_head_cfg = self.config["model"].get("dual_head", {})
        
        # 获取原始融合权重
        self.base_reg_weight = self.dual_head_cfg.get("regression", {}).get("weight", 0.6)
        self.base_cls_weight = self.dual_head_cfg.get("classification", {}).get("weight", 0.4)
        
        # 回测器和策略
        self.backtester = VectorBacktester()
        self.strategy = TopKSignalStrategy()
        
        # 如果未开启仓位管理，强制满仓测试
        if not self.config["strategy"].get("position_control", {}).get("enable", False):
            self.strategy.min_score = -999.0
        
        # 输出目录
        self.report_dir = os.path.join(self.config["paths"]["reports"], "monte_carlo")
        ensure_dir(self.report_dir)
    
    def load_predictions(self) -> Optional[pd.DataFrame]:
        """加载最新的预测文件"""
        models_dir = self.config["paths"]["models"]
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
            df = read_parquet(pred_path)
            df["date"] = pd.to_datetime(df["date"])
            return df
        return None
    
    def _run_single_backtest(self, pred_df: pd.DataFrame, silent: bool = True) -> Optional[Dict]:
        """
        执行单次回测
        
        Returns:
            包含绩效指标的字典，失败返回 None
        """
        try:
            signal_df = self.strategy.generate(pred_df)
            if signal_df.empty:
                return None
            
            # 使用临时目录避免覆盖
            temp_dir = os.path.join(self.report_dir, "_temp")
            metrics = self.backtester.run(signal_df, output_dir=temp_dir)
            
            return {
                "annual_return": metrics["annual_return"],
                "sharpe": metrics["sharpe"],
                "max_drawdown": metrics["max_drawdown"],
                "total_return": metrics.get("total_return", 0),
                "volatility": metrics.get("volatility", 0),
                "equity_curve": metrics.get("equity_curve")
            }
        except Exception as e:
            if not silent:
                logger.warning(f"回测失败: {e}")
            return None
    
    # ========================================================================
    # 模拟方法 1: Bootstrap 重采样
    # ========================================================================
    def run_bootstrap_simulation(self, pred_df: pd.DataFrame) -> List[Dict]:
        """
        Bootstrap 重采样模拟
        
        对每日的股票信号进行有放回抽样，重新计算收益。
        
        Returns:
            模拟结果列表
        """
        logger.info(f">>> 执行 Bootstrap 重采样模拟 ({self.n_simulations} 次)...")
        results = []
        
        # 按日期分组
        dates = pred_df["date"].unique()
        
        for i in range(self.n_simulations):
            # 随机抽样 80% 的日期（有放回）
            sample_dates = np.random.choice(dates, size=int(len(dates) * 0.8), replace=True)
            sample_df = pred_df[pred_df["date"].isin(sample_dates)].copy()
            
            if len(sample_df) < 100:
                continue
            
            metrics = self._run_single_backtest(sample_df)
            if metrics:
                metrics["simulation_id"] = i
                metrics["method"] = "bootstrap"
                results.append(metrics)
            
            if (i + 1) % 100 == 0:
                logger.info(f"  Bootstrap 进度: {i + 1}/{self.n_simulations}")
        
        logger.info(f"  Bootstrap 完成: {len(results)} 次有效模拟")
        return results
    
    # ========================================================================
    # 模拟方法 2: 权重扰动
    # ========================================================================
    def run_weight_perturbation(self, pred_df: pd.DataFrame) -> List[Dict]:
        """
        权重扰动模拟
        
        随机扰动回归/分类融合权重，评估权重敏感性。
        
        Returns:
            模拟结果列表
        """
        logger.info(f">>> 执行权重扰动模拟 ({self.n_simulations} 次)...")
        results = []
        
        # 检查是否有 pred_reg 和 pred_cls 列
        has_dual_head = "pred_reg" in pred_df.columns and "pred_cls" in pred_df.columns
        
        if not has_dual_head:
            logger.warning("  预测文件不包含双头模型输出 (pred_reg, pred_cls)，跳过权重扰动模拟")
            return results
        
        for i in range(self.n_simulations):
            # 随机生成权重 (保证和为1)
            reg_weight = np.random.uniform(0.2, 0.8)
            cls_weight = 1.0 - reg_weight
            
            # 重新计算融合分数
            perturbed_df = pred_df.copy()
            
            # 归一化
            def min_max_normalize(arr):
                arr = np.array(arr)
                min_val, max_val = arr.min(), arr.max()
                if max_val - min_val < 1e-9:
                    return np.zeros_like(arr)
                return (arr - min_val) / (max_val - min_val)
            
            pred_reg_norm = min_max_normalize(perturbed_df["pred_reg"].values)
            pred_cls_norm = min_max_normalize(perturbed_df["pred_cls"].values)
            
            perturbed_df["pred_score"] = reg_weight * pred_reg_norm + cls_weight * pred_cls_norm
            
            metrics = self._run_single_backtest(perturbed_df)
            if metrics:
                metrics["simulation_id"] = i
                metrics["method"] = "weight_perturbation"
                metrics["reg_weight"] = reg_weight
                metrics["cls_weight"] = cls_weight
                results.append(metrics)
            
            if (i + 1) % 100 == 0:
                logger.info(f"  权重扰动进度: {i + 1}/{self.n_simulations}")
        
        logger.info(f"  权重扰动完成: {len(results)} 次有效模拟")
        return results
    
    # ========================================================================
    # 模拟方法 3: 噪音注入
    # ========================================================================
    def run_noise_injection(self, pred_df: pd.DataFrame) -> List[Dict]:
        """
        噪音注入模拟
        
        向预测分数添加随机噪音，评估模型抗干扰能力。
        
        Returns:
            模拟结果列表
        """
        logger.info(f">>> 执行噪音注入模拟 ({self.n_simulations} 次)...")
        results = []
        
        # 噪音比例范围
        noise_levels = np.linspace(0.0, 0.3, 20)  # 0% ~ 30%
        repeats_per_level = max(1, self.n_simulations // len(noise_levels))
        
        for noise_ratio in noise_levels:
            for j in range(repeats_per_level):
                noisy_df = pred_df.copy()
                
                # 添加噪音
                noise = noisy_df["pred_score"].std() * noise_ratio * np.random.randn(len(noisy_df))
                noisy_df["pred_score"] = noisy_df["pred_score"] + noise
                
                metrics = self._run_single_backtest(noisy_df)
                if metrics:
                    metrics["simulation_id"] = len(results)
                    metrics["method"] = "noise_injection"
                    metrics["noise_ratio"] = noise_ratio
                    results.append(metrics)
        
        logger.info(f"  噪音注入完成: {len(results)} 次有效模拟")
        return results
    
    # ========================================================================
    # 模拟方法 4: 时间窗口采样
    # ========================================================================
    def run_time_window_sampling(self, pred_df: pd.DataFrame) -> List[Dict]:
        """
        时间窗口采样模拟
        
        随机采样不同时间区间进行回测，评估策略时间稳定性。
        
        Returns:
            模拟结果列表
        """
        logger.info(f">>> 执行时间窗口采样模拟 ({self.n_simulations} 次)...")
        results = []
        
        dates = sorted(pred_df["date"].unique())
        total_days = len(dates)
        min_window = max(60, total_days // 4)  # 最少 60 天或总天数的 1/4
        
        for i in range(self.n_simulations):
            # 随机选择窗口大小和起始位置
            window_size = np.random.randint(min_window, total_days)
            start_idx = np.random.randint(0, total_days - window_size)
            
            sample_dates = dates[start_idx:start_idx + window_size]
            sample_df = pred_df[pred_df["date"].isin(sample_dates)].copy()
            
            if len(sample_df) < 100:
                continue
            
            metrics = self._run_single_backtest(sample_df)
            if metrics:
                metrics["simulation_id"] = i
                metrics["method"] = "time_window"
                metrics["start_date"] = str(sample_dates[0])[:10]
                metrics["end_date"] = str(sample_dates[-1])[:10]
                metrics["window_days"] = window_size
                results.append(metrics)
            
            if (i + 1) % 100 == 0:
                logger.info(f"  时间窗口采样进度: {i + 1}/{self.n_simulations}")
        
        logger.info(f"  时间窗口采样完成: {len(results)} 次有效模拟")
        return results
    
    # ========================================================================
    # 汇总和可视化
    # ========================================================================
    def aggregate_results(self, all_results: List[Dict]) -> pd.DataFrame:
        """汇总所有模拟结果"""
        if not all_results:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_results)
        # 移除 equity_curve 列（太大）
        if "equity_curve" in df.columns:
            df = df.drop(columns=["equity_curve"])
        return df
    
    def compute_statistics(self, results_df: pd.DataFrame) -> Dict:
        """
        计算统计汇总
        
        Returns:
            包含各项指标统计的字典
        """
        if results_df.empty:
            return {}
        
        stats = {
            "total_simulations": len(results_df),
            "annual_return": {
                "mean": results_df["annual_return"].mean(),
                "median": results_df["annual_return"].median(),
                "std": results_df["annual_return"].std(),
                "p5": results_df["annual_return"].quantile(0.05),
                "p25": results_df["annual_return"].quantile(0.25),
                "p75": results_df["annual_return"].quantile(0.75),
                "p95": results_df["annual_return"].quantile(0.95),
                "min": results_df["annual_return"].min(),
                "max": results_df["annual_return"].max(),
            },
            "sharpe": {
                "mean": results_df["sharpe"].mean(),
                "median": results_df["sharpe"].median(),
                "std": results_df["sharpe"].std(),
                "p5": results_df["sharpe"].quantile(0.05),
                "p95": results_df["sharpe"].quantile(0.95),
            },
            "max_drawdown": {
                "mean": results_df["max_drawdown"].mean(),
                "median": results_df["max_drawdown"].median(),
                "std": results_df["max_drawdown"].std(),
                "p5": results_df["max_drawdown"].quantile(0.05),
                "p95": results_df["max_drawdown"].quantile(0.95),
            }
        }
        
        return stats
    
    def plot_return_distribution(self, results_df: pd.DataFrame, stats: Dict):
        """绘制收益分布图"""
        if results_df.empty:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. 年化收益分布
        ax1 = axes[0, 0]
        returns = results_df["annual_return"] * 100  # 转为百分比
        ax1.hist(returns, bins=50, edgecolor='black', alpha=0.7, color='#3498db')
        ax1.axvline(stats["annual_return"]["median"] * 100, color='red', linestyle='--', 
                   linewidth=2, label=f'中位数: {stats["annual_return"]["median"]*100:.1f}%')
        ax1.axvline(stats["annual_return"]["p5"] * 100, color='orange', linestyle=':', 
                   linewidth=2, label=f'5%分位: {stats["annual_return"]["p5"]*100:.1f}%')
        ax1.axvline(stats["annual_return"]["p95"] * 100, color='green', linestyle=':', 
                   linewidth=2, label=f'95%分位: {stats["annual_return"]["p95"]*100:.1f}%')
        ax1.set_xlabel("年化收益率 (%)")
        ax1.set_ylabel("频次")
        ax1.set_title("年化收益分布")
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # 2. 夏普比率分布
        ax2 = axes[0, 1]
        sharpes = results_df["sharpe"]
        ax2.hist(sharpes, bins=50, edgecolor='black', alpha=0.7, color='#2ecc71')
        ax2.axvline(stats["sharpe"]["median"], color='red', linestyle='--', 
                   linewidth=2, label=f'中位数: {stats["sharpe"]["median"]:.2f}')
        ax2.set_xlabel("夏普比率")
        ax2.set_ylabel("频次")
        ax2.set_title("夏普比率分布")
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # 3. 最大回撤分布
        ax3 = axes[1, 0]
        drawdowns = results_df["max_drawdown"] * 100  # 转为百分比
        ax3.hist(drawdowns, bins=50, edgecolor='black', alpha=0.7, color='#e74c3c')
        ax3.axvline(stats["max_drawdown"]["median"] * 100, color='blue', linestyle='--', 
                   linewidth=2, label=f'中位数: {stats["max_drawdown"]["median"]*100:.1f}%')
        ax3.set_xlabel("最大回撤 (%)")
        ax3.set_ylabel("频次")
        ax3.set_title("最大回撤分布")
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        
        # 4. 收益-风险散点图
        ax4 = axes[1, 1]
        scatter = ax4.scatter(
            results_df["max_drawdown"] * 100, 
            results_df["annual_return"] * 100,
            c=results_df["sharpe"], 
            cmap='RdYlGn', 
            alpha=0.6,
            s=30
        )
        ax4.set_xlabel("最大回撤 (%)")
        ax4.set_ylabel("年化收益率 (%)")
        ax4.set_title("收益-风险散点图 (颜色=夏普比率)")
        plt.colorbar(scatter, ax=ax4, label="夏普比率")
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle("蒙特卡洛模拟结果分析", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        chart_path = os.path.join(self.report_dir, "monte_carlo_distribution.png")
        plt.savefig(chart_path, dpi=150)
        plt.close()
        logger.info(f"分布图已保存: {chart_path}")
    
    def plot_noise_sensitivity(self, results_df: pd.DataFrame):
        """绘制噪音敏感性图"""
        noise_results = results_df[results_df["method"] == "noise_injection"]
        if noise_results.empty:
            return
        
        # 按噪音比例分组
        grouped = noise_results.groupby("noise_ratio").agg({
            "annual_return": ["mean", "std"],
            "sharpe": ["mean", "std"]
        }).reset_index()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 1. 年化收益 vs 噪音比例
        ax1 = axes[0]
        noise_levels = grouped["noise_ratio"] * 100
        returns_mean = grouped[("annual_return", "mean")] * 100
        returns_std = grouped[("annual_return", "std")] * 100
        
        ax1.plot(noise_levels, returns_mean, 'b-o', linewidth=2, markersize=6, label='均值')
        ax1.fill_between(noise_levels, returns_mean - returns_std, returns_mean + returns_std, 
                        alpha=0.3, color='blue')
        ax1.set_xlabel("噪音比例 (%)")
        ax1.set_ylabel("年化收益率 (%)")
        ax1.set_title("年化收益 vs 噪音强度")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. 夏普比率 vs 噪音比例
        ax2 = axes[1]
        sharpe_mean = grouped[("sharpe", "mean")]
        sharpe_std = grouped[("sharpe", "std")]
        
        ax2.plot(noise_levels, sharpe_mean, 'g-o', linewidth=2, markersize=6, label='均值')
        ax2.fill_between(noise_levels, sharpe_mean - sharpe_std, sharpe_mean + sharpe_std, 
                        alpha=0.3, color='green')
        ax2.set_xlabel("噪音比例 (%)")
        ax2.set_ylabel("夏普比率")
        ax2.set_title("夏普比率 vs 噪音强度")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.suptitle("噪音敏感性分析", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        chart_path = os.path.join(self.report_dir, "noise_sensitivity.png")
        plt.savefig(chart_path, dpi=150)
        plt.close()
        logger.info(f"噪音敏感性图已保存: {chart_path}")
    
    def plot_weight_sensitivity(self, results_df: pd.DataFrame):
        """绘制权重敏感性图"""
        weight_results = results_df[results_df["method"] == "weight_perturbation"]
        if weight_results.empty or "reg_weight" not in weight_results.columns:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        scatter = ax.scatter(
            weight_results["reg_weight"], 
            weight_results["annual_return"] * 100,
            c=weight_results["sharpe"], 
            cmap='RdYlGn', 
            alpha=0.6,
            s=50
        )
        
        # 标记最佳点
        best_idx = weight_results["sharpe"].idxmax()
        best_row = weight_results.loc[best_idx]
        ax.scatter(best_row["reg_weight"], best_row["annual_return"] * 100, 
                  s=200, c='red', marker='*', edgecolors='black', linewidths=2,
                  label=f'最佳: α={best_row["reg_weight"]:.2f}, 夏普={best_row["sharpe"]:.2f}')
        
        # 标记原始权重
        ax.axvline(self.base_reg_weight, color='orange', linestyle='--', 
                  linewidth=2, label=f'配置权重: α={self.base_reg_weight:.2f}')
        
        ax.set_xlabel("回归权重 (α)")
        ax.set_ylabel("年化收益率 (%)")
        ax.set_title("融合权重敏感性分析")
        plt.colorbar(scatter, ax=ax, label="夏普比率")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        chart_path = os.path.join(self.report_dir, "weight_sensitivity.png")
        plt.savefig(chart_path, dpi=150)
        plt.close()
        logger.info(f"权重敏感性图已保存: {chart_path}")
    
    def generate_report(self, results_df: pd.DataFrame, stats: Dict):
        """生成文本报告"""
        if results_df.empty:
            return
        
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("📊 蒙特卡洛模拟分析报告 (Monte Carlo Simulation Report)")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        report_lines.append(f"模拟总次数: {stats['total_simulations']}")
        report_lines.append("")
        
        report_lines.append("-" * 60)
        report_lines.append("【年化收益率统计】")
        report_lines.append("-" * 60)
        ar = stats["annual_return"]
        report_lines.append(f"  均值:     {ar['mean']*100:>8.2f}%")
        report_lines.append(f"  中位数:   {ar['median']*100:>8.2f}%")
        report_lines.append(f"  标准差:   {ar['std']*100:>8.2f}%")
        report_lines.append(f"  5%分位:   {ar['p5']*100:>8.2f}%")
        report_lines.append(f"  25%分位:  {ar['p25']*100:>8.2f}%")
        report_lines.append(f"  75%分位:  {ar['p75']*100:>8.2f}%")
        report_lines.append(f"  95%分位:  {ar['p95']*100:>8.2f}%")
        report_lines.append(f"  最小值:   {ar['min']*100:>8.2f}%")
        report_lines.append(f"  最大值:   {ar['max']*100:>8.2f}%")
        report_lines.append("")
        
        report_lines.append("-" * 60)
        report_lines.append("【夏普比率统计】")
        report_lines.append("-" * 60)
        sr = stats["sharpe"]
        report_lines.append(f"  均值:     {sr['mean']:>8.2f}")
        report_lines.append(f"  中位数:   {sr['median']:>8.2f}")
        report_lines.append(f"  标准差:   {sr['std']:>8.2f}")
        report_lines.append(f"  5%分位:   {sr['p5']:>8.2f}")
        report_lines.append(f"  95%分位:  {sr['p95']:>8.2f}")
        report_lines.append("")
        
        report_lines.append("-" * 60)
        report_lines.append("【最大回撤统计】")
        report_lines.append("-" * 60)
        md = stats["max_drawdown"]
        report_lines.append(f"  均值:     {md['mean']*100:>8.2f}%")
        report_lines.append(f"  中位数:   {md['median']*100:>8.2f}%")
        report_lines.append(f"  标准差:   {md['std']*100:>8.2f}%")
        report_lines.append(f"  5%分位:   {md['p5']*100:>8.2f}%")
        report_lines.append(f"  95%分位:  {md['p95']*100:>8.2f}%")
        report_lines.append("")
        
        report_lines.append("-" * 60)
        report_lines.append("【置信区间解读】")
        report_lines.append("-" * 60)
        report_lines.append(f"  • 90% 置信区间下，年化收益预期在 {ar['p5']*100:.1f}% ~ {ar['p95']*100:.1f}% 之间")
        report_lines.append(f"  • 90% 置信区间下，夏普比率预期在 {sr['p5']:.2f} ~ {sr['p95']:.2f} 之间")
        report_lines.append(f"  • 90% 置信区间下，最大回撤预期在 {md['p5']*100:.1f}% ~ {md['p95']*100:.1f}% 之间")
        
        # 稳健性评估
        report_lines.append("")
        report_lines.append("-" * 60)
        report_lines.append("【稳健性评估】")
        report_lines.append("-" * 60)
        
        # 收益波动系数
        cv = abs(ar['std'] / ar['mean']) if ar['mean'] != 0 else float('inf')
        if cv < 0.3:
            stability = "✅ 高度稳健 (变异系数 < 0.3)"
        elif cv < 0.6:
            stability = "⚠️ 中等稳健 (变异系数 0.3 ~ 0.6)"
        else:
            stability = "❌ 波动较大 (变异系数 > 0.6)"
        report_lines.append(f"  收益稳定性: {stability}")
        report_lines.append(f"  变异系数: {cv:.2f}")
        
        # 最差情况分析
        if ar['p5'] > 0:
            report_lines.append(f"  最差 5% 情况仍盈利: ✅ 是 ({ar['p5']*100:.1f}%)")
        else:
            report_lines.append(f"  最差 5% 情况仍盈利: ❌ 否 ({ar['p5']*100:.1f}%)")
        
        report_lines.append("=" * 60)
        
        # 打印报告
        report_text = "\n".join(report_lines)
        print("\n" + report_text)
        
        # 保存报告
        report_path = os.path.join(self.report_dir, "monte_carlo_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        logger.info(f"报告已保存: {report_path}")


# ============================================================================
# 主程序
# ============================================================================

def main():
    logger.info("=" * 60)
    logger.info("=== 蒙特卡洛模拟分析 (Monte Carlo Simulation) ===")
    logger.info("=" * 60)
    
    # 初始化分析器
    analyzer = MonteCarloAnalyzer(n_simulations = 50, random_seed=42)
    
    # 1. 加载预测数据
    pred_df = analyzer.load_predictions()
    if pred_df is None:
        logger.error("未找到预测文件，请先运行 run_walkforward.py")
        return
    
    logger.info(f"预测数据: {len(pred_df)} 行, 日期范围: {pred_df['date'].min()} ~ {pred_df['date'].max()}")
    
    # 检查双头模型列
    has_dual_head = "pred_reg" in pred_df.columns and "pred_cls" in pred_df.columns
    logger.info(f"双头模型预测列: {'存在' if has_dual_head else '不存在'}")
    
    # 2. 执行各种模拟
    all_results = []
    
    # 2.1 Bootstrap 重采样
    bootstrap_results = analyzer.run_bootstrap_simulation(pred_df)
    all_results.extend(bootstrap_results)
    
    # 2.2 权重扰动（仅双头模型）
    if has_dual_head:
        weight_results = analyzer.run_weight_perturbation(pred_df)
        all_results.extend(weight_results)
    
    # 2.3 噪音注入
    noise_results = analyzer.run_noise_injection(pred_df)
    all_results.extend(noise_results)
    
    # 2.4 时间窗口采样
    time_results = analyzer.run_time_window_sampling(pred_df)
    all_results.extend(time_results)
    
    # 3. 汇总结果
    results_df = analyzer.aggregate_results(all_results)
    
    if results_df.empty:
        logger.error("所有模拟均失败，无法生成报告")
        return
    
    # 保存详细结果
    results_path = os.path.join(analyzer.report_dir, "monte_carlo_results.csv")
    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
    logger.info(f"详细结果已保存: {results_path}")
    
    # 4. 计算统计汇总
    stats = analyzer.compute_statistics(results_df)
    
    # 5. 生成可视化
    analyzer.plot_return_distribution(results_df, stats)
    analyzer.plot_noise_sensitivity(results_df)
    if has_dual_head:
        analyzer.plot_weight_sensitivity(results_df)
    
    # 6. 生成报告
    analyzer.generate_report(results_df, stats)
    
    logger.info(f"\n蒙特卡洛分析完成！报告已保存至: {analyzer.report_dir}")


if __name__ == "__main__":
    main()
