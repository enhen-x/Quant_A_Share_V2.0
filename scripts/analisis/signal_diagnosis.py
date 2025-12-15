# scripts/signal_diagnosis.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
# 从当前文件位置 (scripts/analisis) 返回两级到项目根目录
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.config import GLOBAL_CONFIG
from src.utils.io import read_parquet, ensure_dir
from src.strategy.signal import TopKSignalStrategy
from src.backtest.backtester import VectorBacktester

# ==============================================================================
# 模块级绘图配置 (强制覆盖默认设置) - 确保在任何 plt.figure() 调用之前运行
# ==============================================================================
try:
    # 1. 设置样式 (如果样式冲突，可以尝试注释掉这一行，以验证冲突是否是根源)
    plt.style.use('ggplot')
except:
    pass

# 2. 确保中文字体可用，并包含一个兼容性强的字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
# 3. 强制使用标准的 ASCII 减号 ('-') 代替 Unicode 减号 ('\u2212')，解决警告
plt.rcParams['axes.unicode_minus'] = False 
plt.rcParams['figure.figsize'] = (12, 6)
# ==============================================================================

class SignalDiagnosis:
    def __init__(self):
        self.config = GLOBAL_CONFIG
        self.paths = self.config["paths"]
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(self.paths["reports"], "signal_diagnosis", timestamp)
        self.figure_dir = os.path.join(self.paths["figures"], "signals", timestamp)
        self.report_path = os.path.join(self.output_dir, "diagnosis_summary.md")
        ensure_dir(self.output_dir)
        ensure_dir(self.figure_dir)
        # 注意：此处已删除 self._setup_plotting() 的调用
        self.report_lines = []

    # ！！！已移除原有的 _setup_plotting 方法 ！！！

    def load_data(self):
        model_dir = os.path.join(self.paths["models"])
        latest_model = sorted(os.listdir(model_dir))[-1]
        pred_path = os.path.join(model_dir, latest_model, "predictions.parquet")
        self.pred_df = read_parquet(pred_path)
        self.pred_df["date"] = pd.to_datetime(self.pred_df["date"])
        data_path = os.path.join(self.paths["data_processed"], "all_stocks.parquet")
        self.all_df = read_parquet(data_path)
        strategy = TopKSignalStrategy()
        self.signal_df = strategy.generate(self.pred_df)

    def log(self, text):
        print(text)
        self.report_lines.append(text)

    def analyze_risk_exposure(self):
        """风险暴露分析 - 修复单位问题"""
        merged = self.signal_df.merge(self.all_df, on=["date", "symbol"], how="left")

        self.log("## 风险暴露分析\n")

        # 换手率分析 (注意：turnover 数据是百分比形式，如 1.5 表示 1.5%)
        liquidity = merged["turnover"]
        median_turnover = liquidity.median()
        # 直接使用数值，不用 % 格式化（因为已经是百分比）
        self.log(f"- 中位换手率：{median_turnover:.2f}%")
        if median_turnover < 1.0:  # 小于 1% 表示流动性风险
            self.log("  - ⚠️ 警告：换手率偏低，存在流动性风险")
        
        plt.figure()
        sns.histplot(liquidity.dropna(), bins=50, color="blue")
        plt.axvline(x=1.0, color='r', linestyle='--', label='1% 阈值')
        plt.xlabel("换手率 (%)")
        plt.title("换手率分布（选股股票）")
        plt.legend()
        plt.savefig(os.path.join(self.figure_dir, "turnover_distribution.png"))
        plt.close()

        # 价格分析
        prices = merged["close"]
        median_price = prices.median()
        low_price_ratio = (prices < 5).mean()
        self.log(f"- 中位价格：{median_price:.2f} 元，低于5元占比：{low_price_ratio:.1%}")
        if low_price_ratio > 0.3:
            self.log("  - ⚠️ 警告：低价股比例偏高")

        plt.figure()
        # 裁剪极端值以便更好显示
        plot_prices = prices[(prices > 0) & (prices < prices.quantile(0.99))]
        sns.histplot(plot_prices.dropna(), bins=50, color="purple")
        plt.axvline(x=5, color='r', linestyle='--', label='5元阈值')
        plt.xlabel("价格 (元)")
        plt.title("价格分布（选股股票）")
        plt.legend()
        plt.savefig(os.path.join(self.figure_dir, "price_distribution.png"))
        plt.close()

        # 波动率分析 - 与全市场对比
        merged = merged.sort_values(by=["symbol", "date"])
        merged["volatility"] = merged.groupby("symbol")["close"].transform(
            lambda x: x.pct_change().rolling(60).std() * 100  # 转换为百分比
        )
        
        # 同时计算全市场波动率作为对比
        self.all_df_sorted = self.all_df.sort_values(by=["symbol", "date"])
        self.all_df_sorted["volatility"] = self.all_df_sorted.groupby("symbol")["close"].transform(
            lambda x: x.pct_change().rolling(60).std() * 100
        )
        
        vol_selected = merged["volatility"].dropna()
        vol_all = self.all_df_sorted["volatility"].dropna()
        median_vol = vol_selected.median()
        median_vol_all = vol_all.median()
        
        vol_diff = median_vol - median_vol_all
        self.log(f"- 波动率中位数（60日）：{median_vol:.2f}% (全市场: {median_vol_all:.2f}%, 差异: {vol_diff:+.2f}%)")
        if median_vol > median_vol_all * 1.2:
            self.log("  - ⚠️ 警告：选股组合波动率显著高于市场平均")

        # 绘制对比图
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 左图：分布对比
        vol_sel_clip = vol_selected[(vol_selected > 0) & (vol_selected < vol_selected.quantile(0.95))]
        vol_all_clip = vol_all[(vol_all > 0) & (vol_all < vol_all.quantile(0.95))]
        
        axes[0].hist(vol_all_clip, bins=50, alpha=0.5, label=f'全市场 (中位数:{median_vol_all:.2f}%)', color='gray', density=True)
        axes[0].hist(vol_sel_clip, bins=50, alpha=0.7, label=f'选股组合 (中位数:{median_vol:.2f}%)', color='orange', density=True)
        axes[0].axvline(median_vol, color='orange', linestyle='--', linewidth=2)
        axes[0].axvline(median_vol_all, color='gray', linestyle='--', linewidth=2)
        axes[0].set_xlabel("日波动率 (%)")
        axes[0].set_ylabel("密度")
        axes[0].set_title("波动率分布对比：选股组合 vs 全市场")
        axes[0].legend()
        
        # 右图：箱线图对比
        box_data = pd.DataFrame({
            '选股组合': vol_sel_clip.sample(min(5000, len(vol_sel_clip)), random_state=42),
            '全市场': vol_all_clip.sample(min(5000, len(vol_all_clip)), random_state=42)
        })
        box_data.boxplot(ax=axes[1])
        axes[1].set_ylabel("日波动率 (%)")
        axes[1].set_title("波动率箱线图对比")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_dir, "volatility_distribution.png"), dpi=120)
        plt.close()

        # 动量分析 - 与全市场对比
        merged["momentum_1m"] = merged.groupby("symbol")["close"].transform(
            lambda x: x.pct_change(20) * 100
        )
        merged["momentum_3m"] = merged.groupby("symbol")["close"].transform(
            lambda x: x.pct_change(60) * 100
        )
        
        self.all_df_sorted["momentum_1m"] = self.all_df_sorted.groupby("symbol")["close"].transform(
            lambda x: x.pct_change(20) * 100
        )
        self.all_df_sorted["momentum_3m"] = self.all_df_sorted.groupby("symbol")["close"].transform(
            lambda x: x.pct_change(60) * 100
        )
        
        mom1_sel = merged["momentum_1m"].dropna()
        mom3_sel = merged["momentum_3m"].dropna()
        mom1_all = self.all_df_sorted["momentum_1m"].dropna()
        mom3_all = self.all_df_sorted["momentum_3m"].dropna()
        
        mom1 = mom1_sel.median()
        mom3 = mom3_sel.median()
        mom1_all_med = mom1_all.median()
        mom3_all_med = mom3_all.median()
        
        self.log(f"- 动量中位数：1月={mom1:.2f}% (市场:{mom1_all_med:.2f}%)，3月={mom3:.2f}% (市场:{mom3_all_med:.2f}%)")
        
        # 判断动量风格
        if mom1 > mom1_all_med + 5:
            self.log("  - 📈 选股组合呈现**强动量**风格")
        elif mom1 < mom1_all_med - 5:
            self.log("  - 📉 选股组合呈现**反转/弱势**风格")

        # 近1月动量对比图
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        mom1_sel_clip = mom1_sel[(mom1_sel > mom1_sel.quantile(0.02)) & (mom1_sel < mom1_sel.quantile(0.98))]
        mom1_all_clip = mom1_all[(mom1_all > mom1_all.quantile(0.02)) & (mom1_all < mom1_all.quantile(0.98))]
        
        axes[0].hist(mom1_all_clip, bins=50, alpha=0.5, label=f'全市场 (中位数:{mom1_all_med:.1f}%)', color='gray', density=True)
        axes[0].hist(mom1_sel_clip, bins=50, alpha=0.7, label=f'选股组合 (中位数:{mom1:.1f}%)', color='green', density=True)
        axes[0].axvline(0, color='k', linestyle='--', linewidth=1)
        axes[0].axvline(mom1, color='green', linestyle='--', linewidth=2)
        axes[0].set_xlabel("近1月收益率 (%)")
        axes[0].set_ylabel("密度")
        axes[0].set_title("近1月动量分布对比")
        axes[0].legend()
        
        # 近3月动量对比图
        mom3_sel_clip = mom3_sel[(mom3_sel > mom3_sel.quantile(0.02)) & (mom3_sel < mom3_sel.quantile(0.98))]
        mom3_all_clip = mom3_all[(mom3_all > mom3_all.quantile(0.02)) & (mom3_all < mom3_all.quantile(0.98))]
        
        axes[1].hist(mom3_all_clip, bins=50, alpha=0.5, label=f'全市场 (中位数:{mom3_all_med:.1f}%)', color='gray', density=True)
        axes[1].hist(mom3_sel_clip, bins=50, alpha=0.7, label=f'选股组合 (中位数:{mom3:.1f}%)', color='teal', density=True)
        axes[1].axvline(0, color='k', linestyle='--', linewidth=1)
        axes[1].axvline(mom3, color='teal', linestyle='--', linewidth=2)
        axes[1].set_xlabel("近3月收益率 (%)")
        axes[1].set_ylabel("密度")
        axes[1].set_title("近3月动量分布对比")
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_dir, "momentum_distribution.png"), dpi=120)
        plt.close()
        
        # 兼容性：保留原有文件名
        plt.figure(figsize=(10, 6))
        plt.hist(mom1_sel_clip, bins=50, alpha=0.7, color='green', density=True)
        plt.axvline(x=0, color='k', linestyle='--')
        plt.axvline(x=mom1, color='green', linestyle='--', label=f'中位数:{mom1:.1f}%')
        plt.xlabel("收益率 (%)")
        plt.title("近1月收益分布（选股组合）")
        plt.legend()
        plt.savefig(os.path.join(self.figure_dir, "momentum_1m_distribution.png"), dpi=120)
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.hist(mom3_sel_clip, bins=50, alpha=0.7, color='teal', density=True)
        plt.axvline(x=0, color='k', linestyle='--')
        plt.axvline(x=mom3, color='teal', linestyle='--', label=f'中位数:{mom3:.1f}%')
        plt.xlabel("收益率 (%)")
        plt.title("近3月收益分布（选股组合）")
        plt.legend()
        plt.savefig(os.path.join(self.figure_dir, "momentum_3m_distribution.png"), dpi=120)
        plt.close()

    def analyze_signal_quality(self):
        self.log("\n## 信号质量与稳定性分析\n")

        selected = self.pred_df[self.pred_df["symbol"].isin(self.signal_df["symbol"])]
        scores = selected["pred_score"]
        score_std = scores.std()
        self.log(f"- 预测分数标准差：{score_std:.4f}")
        if score_std < 0.01:
            self.log("  - ⚠️ 警告：预测分数过于集中，可能缺乏区分力")

        plt.figure()
        sns.histplot(scores, bins=50, kde=True, color="green")
        plt.title("模型预测得分分布（选股股票）")
        plt.savefig(os.path.join(self.figure_dir, "score_distribution.png"))
        plt.close()

        turnover_rates = []
        signal_by_date = self.signal_df.groupby("date")["symbol"].apply(list)
        dates = sorted(signal_by_date.index)
        for i in range(1, len(dates)):
            prev = set(signal_by_date[dates[i - 1]])
            curr = set(signal_by_date[dates[i]])
            # 计算换手率：(调仓数) / (当前持仓数) = (新增 + 卖出) / (持仓)
            # 简化为： 1 - (不变持仓数) / (新持仓数)
            turnover = 1 - len(prev & curr) / len(curr)
            turnover_rates.append(turnover)
        avg_turnover = np.mean(turnover_rates)
        self.log(f"- 平均换仓率：{avg_turnover:.1%}")

        plt.figure()
        # x 轴需要是日期对象
        plt.plot(dates[1:], turnover_rates) 
        plt.title("换仓率变化曲线")
        plt.savefig(os.path.join(self.figure_dir, "turnover_rate.png"))
        plt.close()

        self.pred_df["month"] = self.pred_df["date"].dt.to_period("M")
        ic_list = []
        for _, group in self.pred_df.groupby("month"):
            if group["label"].nunique() > 1:
                ic = group["pred_score"].corr(group["label"], method="spearman")
                ic_list.append(ic)
        ic_mean = np.mean(ic_list)
        ic_ir = ic_mean / np.std(ic_list) if np.std(ic_list) != 0 else np.nan
        self.log(f"- 月度IC均值：{ic_mean:.4f}，IR={ic_ir:.3f}")

        plt.figure()
        sns.barplot(x=list(range(len(ic_list))), y=ic_list, hue=list(range(len(ic_list))), palette="viridis", legend=False)
        plt.axhline(0, color="black", linestyle="--")
        plt.title("每月IC值")
        plt.savefig(os.path.join(self.figure_dir, "ic_by_month.png"))
        plt.close()

        backtester = VectorBacktester()
        cost_rates = [0.001, 0.002, 0.003, 0.005]
        cost_results = []
        # 注意：这里 run 方法会在内部调用 _plot_result，并再次创建 Figure，因此模块级配置至关重要
        for cost in cost_rates:
            # 传递 output_dir 是为了让 backtester 知道把图表放在哪里
            result = backtester.run(self.signal_df, cost_rate=cost, output_dir=self.output_dir)
            cost_results.append([cost, result.get("annual_return", 0), result.get("sharpe", 0)])
        df_cost = pd.DataFrame(cost_results, columns=["Cost", "AnnualReturn", "Sharpe"])
        df_cost.to_csv(os.path.join(self.output_dir, "cost_sensitivity.csv"), index=False)
        self.log("\n- 交易成本敏感性测试已保存为 `cost_sensitivity.csv`")

        crisis_periods = {
            "2018_TradeWar": ("2018-01-01", "2018-12-31"),
            "2022_FedHike": ("2022-01-01", "2022-12-31"),
            "2024_Liquidity": ("2024-01-01", "2024-02-29")
        }
        crisis_results = []
        for name, (start, end) in crisis_periods.items():
            result = backtester.run(self.signal_df, start_date=start, end_date=end, output_dir=self.output_dir)
            crisis_results.append([name, result.get("annual_return", 0), result.get("max_drawdown", 0)])
        df_crisis = pd.DataFrame(crisis_results, columns=["Scenario", "AnnReturn", "MaxDrawdown"])
        df_crisis.to_csv(os.path.join(self.output_dir, "crisis_test.csv"), index=False)
        self.log("- 历史危机时期压力测试结果已保存为 `crisis_test.csv`")

    def run(self):
        self.load_data()
        if self.signal_df.empty:
            print("未生成信号，终止诊断。")
            return
        self.analyze_risk_exposure()
        self.analyze_signal_quality()
        with open(self.report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.report_lines))
        print(f"诊断报告已完成：\n- Markdown报告: {self.report_path}\n- 图表目录: {self.figure_dir}")

if __name__ == "__main__":
    diag = SignalDiagnosis()
    diag.run()