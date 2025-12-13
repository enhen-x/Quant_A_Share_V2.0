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
        merged = self.signal_df.merge(self.all_df, on=["date", "symbol"], how="left")

        self.log("## 风险暴露分析\n")

        liquidity = merged["turnover"]
        median_turnover = liquidity.median()
        self.log(f"- 中位换手率：{median_turnover:.2%}")
        if median_turnover < 0.01:
            self.log("  - ⚠️ 警告：换手率偏低，存在流动性风险")
        
        plt.figure()
        sns.histplot(liquidity.dropna(), bins=50, color="blue")
        plt.axvline(x=0.01, color='r', linestyle='--')
        plt.title("换手率分布（选股股票）")
        plt.savefig(os.path.join(self.figure_dir, "turnover_distribution.png"))
        plt.close()

        prices = merged["close"]
        median_price = prices.median()
        low_price_ratio = (prices < 5).mean()
        self.log(f"- 中位价格：{median_price:.2f} 元，低于5元占比：{low_price_ratio:.1%}")
        if low_price_ratio > 0.3:
            self.log("  - ⚠️ 警告：低价股比例偏高")

        plt.figure()
        sns.histplot(prices.dropna(), bins=50, color="purple")
        plt.axvline(x=5, color='r', linestyle='--')
        plt.title("价格分布（选股股票）")
        plt.savefig(os.path.join(self.figure_dir, "price_distribution.png"))
        plt.close()

        merged = merged.sort_values(by=["symbol", "date"])
        merged["volatility"] = merged.groupby("symbol")["close"].transform(lambda x: x.pct_change().rolling(60).std())
        vol = merged["volatility"]
        median_vol = vol.median()
        self.log(f"- 波动率中位数（60日）：{median_vol:.2%}")

        plt.figure()
        sns.histplot(vol.dropna(), bins=50, color="orange")
        plt.title("历史收益波动率分布（选股股票）")
        plt.savefig(os.path.join(self.figure_dir, "volatility_distribution.png"))
        plt.close()

        merged["momentum_1m"] = merged.groupby("symbol")["close"].transform(lambda x: x.pct_change(20))
        merged["momentum_3m"] = merged.groupby("symbol")["close"].transform(lambda x: x.pct_change(60))
        mom1 = merged["momentum_1m"].median()
        mom3 = merged["momentum_3m"].median()
        self.log(f"- 动量中位数：1月={mom1:.2%}，3月={mom3:.2%}")

        plt.figure()
        sns.histplot(merged["momentum_1m"].dropna(), bins=50, color="green")
        plt.title("近1个月收益分布")
        plt.savefig(os.path.join(self.figure_dir, "momentum_1m_distribution.png"))
        plt.close()

        plt.figure()
        sns.histplot(merged["momentum_3m"].dropna(), bins=50, color="teal")
        plt.title("近3个月收益分布")
        plt.savefig(os.path.join(self.figure_dir, "momentum_3m_distribution.png"))
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