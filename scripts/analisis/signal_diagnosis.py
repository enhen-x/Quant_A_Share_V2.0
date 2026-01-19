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
import warnings
import logging
import matplotlib as mpl

# 抑制字体相关的警告
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', message='.*glyph.*')

# 禁用 matplotlib 的字体警告日志
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
logging.getLogger('matplotlib.backends').setLevel(logging.ERROR)

try:
    plt.style.use('ggplot')
except:
    pass

# 字体配置 - 确保中文显示和减号正确
plt.rcParams.update({
    'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'sans-serif'],
    'font.family': 'sans-serif',
    'axes.unicode_minus': False,  # 使用 ASCII 减号
    'mathtext.fontset': 'dejavusans',  # 使用 DejaVu Sans 数学字体
    'figure.figsize': (12, 6),
})
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

        # 波动率分析 - 修正：先在全市场数据上计算波动率，再筛选信号股票
        self.all_df_sorted = self.all_df.sort_values(by=["symbol", "date"])
        self.all_df_sorted["volatility"] = self.all_df_sorted.groupby("symbol")["close"].transform(
            lambda x: x.pct_change().rolling(60).std() * 100  # 转换为百分比
        )
        
        # 通过 merge 筛选信号股票的波动率数据
        signal_subset = self.signal_df[["date", "symbol"]].copy()
        selected_vol = self.all_df_sorted.merge(signal_subset, on=["date", "symbol"], how="inner")
        
        vol_selected = selected_vol["volatility"].dropna()
        vol_all = self.all_df_sorted["volatility"].dropna()
        median_vol = vol_selected.median()
        median_vol_all = vol_all.median()
        
        print(f"[调试] 波动率 - 信号数据量: {len(signal_subset)}, 匹配数据: {len(selected_vol)}, 非空: {len(vol_selected)}")
        
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

        # 短期收益分析 - 适配4天轮换策略，分析1d/2d/4d/7d/10d的短期动量
        # 定义分析周期（天数）
        momentum_periods = [1, 2, 4, 7, 10]
        period_labels = {1: '1d', 2: '2d', 4: '4d', 7: '7d', 10: '10d'}
        
        # 全市场动量计算
        for period in momentum_periods:
            col_name = f"momentum_{period}d"
            self.all_df_sorted[col_name] = self.all_df_sorted.groupby("symbol")["close"].transform(
                lambda x: x.pct_change(period) * 100
            )
        
        # 通过 merge 筛选信号股票的动量数据
        signal_subset = self.signal_df[["date", "symbol"]].copy()
        selected_momentum = self.all_df_sorted.merge(signal_subset, on=["date", "symbol"], how="inner")
        
        print(f"[调试] 信号数据量: {len(signal_subset)}, 匹配动量数据: {len(selected_momentum)}")
        
        self.log(f"- 短期收益分析（适配4天轮换策略）：")
        
        # 收集统计结果
        momentum_stats = []
        for period in momentum_periods:
            col_name = f"momentum_{period}d"
            mom_sel = selected_momentum[col_name].dropna()
            mom_all = self.all_df_sorted[col_name].dropna()
            
            med_sel = mom_sel.median()
            med_all = mom_all.median()
            diff = med_sel - med_all
            
            momentum_stats.append({
                'period': period,
                'label': period_labels[period],
                'med_sel': med_sel,
                'med_all': med_all,
                'diff': diff,
                'data_sel': mom_sel,
                'data_all': mom_all
            })
            
            self.log(f"  - {period_labels[period]}: 选股={med_sel:.2f}% | 市场={med_all:.2f}% | 差异={diff:+.2f}%")
        
        # 判断短期动量风格（基于4日动量，与策略周期一致）
        mom_4d_stats = next(s for s in momentum_stats if s['period'] == 4)
        if mom_4d_stats['diff'] > 1.0:
            self.log("  - 📈 选股组合呈现**短期强势**风格")
        elif mom_4d_stats['diff'] < -1.0:
            self.log("  - 📉 选股组合呈现**短期弱势/反转**风格")
        else:
            self.log("  - ➡️ 选股组合短期动量与市场接近")

        # 绘制多周期对比图
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()
        
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12']
        
        for idx, stats in enumerate(momentum_stats):
            ax = axes[idx]
            mom_sel = stats['data_sel']
            mom_all = stats['data_all']
            
            # 裁剪极端值
            mom_sel_clip = mom_sel[(mom_sel > mom_sel.quantile(0.02)) & (mom_sel < mom_sel.quantile(0.98))]
            mom_all_clip = mom_all[(mom_all > mom_all.quantile(0.02)) & (mom_all < mom_all.quantile(0.98))]
            
            ax.hist(mom_all_clip, bins=50, alpha=0.5, label=f'全市场 ({stats["med_all"]:.2f}%)', color='gray', density=True)
            ax.hist(mom_sel_clip, bins=50, alpha=0.7, label=f'选股 ({stats["med_sel"]:.2f}%)', color=colors[idx], density=True)
            ax.axvline(0, color='k', linestyle='--', linewidth=1)
            ax.axvline(stats['med_sel'], color=colors[idx], linestyle='--', linewidth=2)
            ax.set_xlabel(f"收益率 (%)")
            ax.set_ylabel("密度")
            ax.set_title(f"{stats['label']} 动量分布对比")
            ax.legend(fontsize=8)
        
        # 最后一个子图：汇总柱状图
        ax = axes[5]
        periods = [s['label'] for s in momentum_stats]
        diffs = [s['diff'] for s in momentum_stats]
        bar_colors = ['green' if d > 0 else 'red' for d in diffs]
        ax.bar(periods, diffs, color=bar_colors, alpha=0.7, edgecolor='black')
        ax.axhline(0, color='k', linestyle='-', linewidth=1)
        ax.set_xlabel("周期")
        ax.set_ylabel("选股 vs 市场 差异 (%)")
        ax.set_title("各周期动量差异汇总")
        for i, (p, d) in enumerate(zip(periods, diffs)):
            ax.annotate(f'{d:+.2f}%', (i, d), ha='center', va='bottom' if d > 0 else 'top', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_dir, "short_term_momentum.png"), dpi=120)
        plt.close()
        
        # 保存4日动量分布图（与策略周期一致）
        mom_4d_sel = momentum_stats[2]['data_sel']  # index 2 = 4d
        mom_4d_clip = mom_4d_sel[(mom_4d_sel > mom_4d_sel.quantile(0.02)) & (mom_4d_sel < mom_4d_sel.quantile(0.98))]
        
        plt.figure(figsize=(10, 6))
        plt.hist(mom_4d_clip, bins=50, alpha=0.7, color='#e74c3c', density=True)
        plt.axvline(x=0, color='k', linestyle='--')
        plt.axvline(x=momentum_stats[2]['med_sel'], color='#e74c3c', linestyle='--', label=f'中位数:{momentum_stats[2]["med_sel"]:.2f}%')
        plt.xlabel("收益率 (%)")
        plt.title("4日收益分布（选股组合）- 与策略周期一致")
        plt.legend()
        plt.savefig(os.path.join(self.figure_dir, "momentum_4d_distribution.png"), dpi=120)
        plt.close()
        
        # ========================================================================
        # 未来收益分析 - 验证反转规律，找出最佳持仓天数
        # ========================================================================
        self.log(f"\n- **未来收益分析**（验证反转规律）：")
        
        # 计算未来N天收益（使用shift向后看）
        for period in momentum_periods:
            col_name = f"future_{period}d"
            # shift(-period) 表示未来第period天的价格
            self.all_df_sorted[col_name] = self.all_df_sorted.groupby("symbol")["close"].transform(
                lambda x: (x.shift(-period) / x - 1) * 100
            )
        
        # 重新merge获取未来收益数据
        selected_future = self.all_df_sorted.merge(signal_subset, on=["date", "symbol"], how="inner")
        
        # 收集未来收益统计
        future_stats = []
        for period in momentum_periods:
            col_name = f"future_{period}d"
            fut_sel = selected_future[col_name].dropna()
            fut_all = self.all_df_sorted[col_name].dropna()
            
            med_sel = fut_sel.median()
            med_all = fut_all.median()
            mean_sel = fut_sel.mean()
            diff = med_sel - med_all
            
            # 计算胜率（未来收益>0的比例）
            win_rate = (fut_sel > 0).mean() * 100
            
            future_stats.append({
                'period': period,
                'label': period_labels[period],
                'med_sel': med_sel,
                'mean_sel': mean_sel,
                'med_all': med_all,
                'diff': diff,
                'win_rate': win_rate,
                'data_sel': fut_sel,
                'data_all': fut_all
            })
            
            self.log(f"  - {period_labels[period]}: 选股={med_sel:.2f}% | 市场={med_all:.2f}% | 超额={diff:+.2f}% | 胜率={win_rate:.1f}%")
        
        # 找出最佳持仓天数（超额收益最大）
        best_period = max(future_stats, key=lambda x: x['diff'])
        self.log(f"  - 🎯 **最佳持仓天数: {best_period['label']}**，超额收益={best_period['diff']:+.2f}%，选股胜率={best_period['win_rate']:.1f}%")
        
        # ========================================================================
        # 扣除成本后的策略胜率分析
        # ========================================================================
        # 交易成本: 印花税0.1%(卖出) + 佣金0.03%(双边) ≈ 0.13% 单边, 0.26% 双边
        cost_rate = 0.26  # 百分比形式，即0.26%
        
        self.log(f"\n- **策略胜率分析**（扣除{cost_rate}%交易成本）：")
        
        for stats in future_stats:
            period = stats['period']
            fut_sel = stats['data_sel']
            
            # 扣除成本后的收益
            net_ret = fut_sel - cost_rate
            
            # 策略胜率 = 扣除成本后收益>0的比例
            strategy_win_rate = (net_ret > 0).mean() * 100
            
            # 平均净收益
            avg_net_ret = net_ret.mean()
            
            # 盈亏比 = 平均盈利 / 平均亏损 (绝对值)
            wins = net_ret[net_ret > 0]
            losses = net_ret[net_ret < 0]
            if len(wins) > 0 and len(losses) > 0:
                profit_loss_ratio = wins.mean() / abs(losses.mean())
            else:
                profit_loss_ratio = np.nan
            
            # 更新stats
            stats['strategy_win_rate'] = strategy_win_rate
            stats['avg_net_ret'] = avg_net_ret
            stats['profit_loss_ratio'] = profit_loss_ratio
            
            self.log(f"  - {stats['label']}: 策略胜率={strategy_win_rate:.1f}% | 平均净收益={avg_net_ret:.2f}% | 盈亏比={profit_loss_ratio:.2f}")
        
        # 找出策略胜率最高的周期
        best_strategy_period = max(future_stats, key=lambda x: x.get('strategy_win_rate', 0))
        self.log(f"  - 🎯 **最高策略胜率: {best_strategy_period['label']}**，胜率={best_strategy_period['strategy_win_rate']:.1f}%，盈亏比={best_strategy_period['profit_loss_ratio']:.2f}")
        
        # 验证反转规律：过去跌 + 未来涨
        past_4d = momentum_stats[2]['med_sel']  # 过去4日收益
        future_4d = future_stats[2]['med_sel']  # 未来4日收益
        
        if past_4d < 0 and future_4d > 0:
            self.log(f"  - ✅ **反转规律验证通过**: 过去4日={past_4d:.2f}% (跌) → 未来4日={future_4d:.2f}% (涨)")
        elif past_4d < 0 and future_4d < 0:
            self.log(f"  - ⚠️ 过去4日={past_4d:.2f}% (跌)，未来4日={future_4d:.2f}% (仍跌)，可能是趋势策略？")
        else:
            self.log(f"  - 📊 过去4日={past_4d:.2f}%，未来4日={future_4d:.2f}%")
        
        # ========================================================================
        # 绘制 过去 vs 未来 对比图
        # ========================================================================
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 子图1: 过去收益 vs 未来收益 柱状图对比
        ax = axes[0, 0]
        x = np.arange(len(momentum_periods))
        width = 0.35
        past_meds = [s['med_sel'] for s in momentum_stats]
        future_meds = [s['med_sel'] for s in future_stats]
        
        bars1 = ax.bar(x - width/2, past_meds, width, label='过去N日收益', color='#e74c3c', alpha=0.8)
        bars2 = ax.bar(x + width/2, future_meds, width, label='未来N日收益', color='#2ecc71', alpha=0.8)
        ax.axhline(0, color='k', linestyle='-', linewidth=1)
        ax.set_xlabel('周期')
        ax.set_ylabel('中位收益率 (%)')
        ax.set_title('选股组合：过去收益 vs 未来收益')
        ax.set_xticks(x)
        ax.set_xticklabels([s['label'] for s in momentum_stats])
        ax.legend()
        
        # 添加数值标注
        for bar, val in zip(bars1, past_meds):
            ax.annotate(f'{val:.2f}%', (bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom' if val > 0 else 'top', fontsize=8)
        for bar, val in zip(bars2, future_meds):
            ax.annotate(f'{val:.2f}%', (bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom' if val > 0 else 'top', fontsize=8)
        
        # 子图2: 超额收益曲线
        ax = axes[0, 1]
        excess_returns = [s['diff'] for s in future_stats]
        win_rates = [s['win_rate'] for s in future_stats]
        
        ax.bar(x, excess_returns, color='#3498db', alpha=0.8, label='超额收益')
        ax.axhline(0, color='k', linestyle='-', linewidth=1)
        ax.set_xlabel('周期')
        ax.set_ylabel('超额收益 (%)', color='#3498db')
        ax.set_title('各周期超额收益与胜率')
        ax.set_xticks(x)
        ax.set_xticklabels([s['label'] for s in future_stats])
        
        # 添加胜率曲线（右轴）
        ax2 = ax.twinx()
        ax2.plot(x, win_rates, 'o-', color='#e67e22', linewidth=2, markersize=8, label='胜率')
        ax2.set_ylabel('胜率 (%)', color='#e67e22')
        ax2.axhline(50, color='#e67e22', linestyle='--', alpha=0.5)
        
        # 标注最佳周期
        best_idx = [s['period'] for s in future_stats].index(best_period['period'])
        ax.annotate(f'最佳\n{best_period["diff"]:+.2f}%', (best_idx, excess_returns[best_idx]),
                   ha='center', va='bottom', fontsize=10, fontweight='bold', color='red')
        
        # 子图3: 未来收益分布对比（各周期）
        ax = axes[1, 0]
        for idx, stats in enumerate(future_stats):
            fut_sel = stats['data_sel']
            fut_clip = fut_sel[(fut_sel > fut_sel.quantile(0.02)) & (fut_sel < fut_sel.quantile(0.98))]
            ax.hist(fut_clip, bins=50, alpha=0.4, label=f'{stats["label"]} ({stats["med_sel"]:.2f}%)', density=True)
        ax.axvline(0, color='k', linestyle='--', linewidth=1)
        ax.set_xlabel('未来收益率 (%)')
        ax.set_ylabel('密度')
        ax.set_title('各周期未来收益分布（选股组合）')
        ax.legend(fontsize=8)
        
        # 子图4: 反转验证 - 过去vs未来散点图（按周期）
        ax = axes[1, 1]
        for idx, (past_s, fut_s) in enumerate(zip(momentum_stats, future_stats)):
            ax.scatter(past_s['med_sel'], fut_s['med_sel'], s=150, label=fut_s['label'], 
                      c=[['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12'][idx]], edgecolors='black')
            ax.annotate(fut_s['label'], (past_s['med_sel'], fut_s['med_sel']), 
                       textcoords="offset points", xytext=(5, 5), fontsize=10)
        
        ax.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax.axvline(0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel('过去N日收益 (%)')
        ax.set_ylabel('未来N日收益 (%)')
        ax.set_title('反转验证：过去收益 vs 未来收益')
        
        # 添加象限标注
        ax.text(ax.get_xlim()[0] + 0.5, ax.get_ylim()[1] - 0.5, '弱势继续', fontsize=9, alpha=0.6)
        ax.text(ax.get_xlim()[1] - 2, ax.get_ylim()[1] - 0.5, '动量延续', fontsize=9, alpha=0.6)
        ax.text(ax.get_xlim()[0] + 0.5, ax.get_ylim()[0] + 0.5, '反转失败', fontsize=9, alpha=0.6)
        ax.text(ax.get_xlim()[1] - 2, ax.get_ylim()[0] + 0.5, '强势回调', fontsize=9, alpha=0.6)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_dir, "reversal_analysis.png"), dpi=120)
        plt.close()
        
        # 保存统计表格
        summary_df = pd.DataFrame({
            '周期': [s['label'] for s in future_stats],
            '过去收益(%)': [s['med_sel'] for s in momentum_stats],
            '未来收益(%)': [s['med_sel'] for s in future_stats],
            '超额收益(%)': [s['diff'] for s in future_stats],
            '胜率(%)': [s['win_rate'] for s in future_stats]
        })
        summary_df.to_csv(os.path.join(self.output_dir, "reversal_summary.csv"), index=False, encoding='utf-8-sig')
        self.log(f"  - 📊 反转分析结果已保存为 `reversal_summary.csv`")

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
        
        # ========================================================================
        # 持仓重复率分析 - 适配4天轮换策略
        # ========================================================================
        self.log("\n### 持仓重复率分析\n")
        
        # 获取持仓周期
        holding_period = self.config.get("preprocessing", {}).get("labels", {}).get("horizon", 4)
        
        # 按换仓周期分组（每 holding_period 天一组）
        all_dates_sorted = sorted(self.signal_df["date"].unique())
        rebalance_dates = all_dates_sorted[::holding_period]  # 每4天取一次
        
        if len(rebalance_dates) < 2:
            self.log("- ⚠️ 换仓次数不足，无法分析重复率")
        else:
            # 获取每个换仓日的持仓列表
            holdings_by_rebalance = {}
            for date in rebalance_dates:
                holdings = set(self.signal_df[self.signal_df["date"] == date]["symbol"].tolist())
                holdings_by_rebalance[date] = holdings
            
            # 计算连续两次换仓间的重复率
            overlap_rates = []
            overlap_counts = []
            rebalance_list = sorted(holdings_by_rebalance.keys())
            
            for i in range(1, len(rebalance_list)):
                prev_date = rebalance_list[i-1]
                curr_date = rebalance_list[i]
                prev_holdings = holdings_by_rebalance[prev_date]
                curr_holdings = holdings_by_rebalance[curr_date]
                
                # 重复股票数
                overlap = prev_holdings & curr_holdings
                overlap_count = len(overlap)
                overlap_counts.append(overlap_count)
                
                # 重复率 = 重复数 / 当前持仓数
                if len(curr_holdings) > 0:
                    overlap_rate = overlap_count / len(curr_holdings)
                    overlap_rates.append(overlap_rate)
            
            avg_overlap_rate = np.mean(overlap_rates)
            avg_overlap_count = np.mean(overlap_counts)
            max_overlap_rate = np.max(overlap_rates)
            min_overlap_rate = np.min(overlap_rates)
            
            self.log(f"- 换仓周期: {holding_period}天，共{len(rebalance_dates)}次换仓")
            self.log(f"- 平均重复率: {avg_overlap_rate:.1%}（平均{avg_overlap_count:.1f}只股票重复）")
            self.log(f"- 重复率范围: {min_overlap_rate:.1%} ~ {max_overlap_rate:.1%}")
            
            # 统计每只股票被连续持有的次数
            all_symbols = set()
            for holdings in holdings_by_rebalance.values():
                all_symbols.update(holdings)
            
            # 计算每只股票在多少个换仓周期中出现
            symbol_freq = {}
            for sym in all_symbols:
                count = sum(1 for h in holdings_by_rebalance.values() if sym in h)
                symbol_freq[sym] = count
            
            freq_series = pd.Series(symbol_freq)
            
            # 只出现1次的股票比例
            one_time_ratio = (freq_series == 1).mean()
            # 出现超过3次的股票比例
            frequent_ratio = (freq_series > 3).mean()
            avg_appearances = freq_series.mean()
            max_appearances = freq_series.max()
            
            self.log(f"- 平均持有周期: {avg_appearances:.1f}轮（最长{max_appearances}轮）")
            self.log(f"- 只出现1轮的股票: {one_time_ratio:.1%}")
            self.log(f"- 出现超过3轮的股票: {frequent_ratio:.1%}")
            
            if avg_overlap_rate > 0.5:
                self.log("  - 📊 重复率较高，说明模型偏好的股票相对稳定")
            elif avg_overlap_rate < 0.2:
                self.log("  - 📊 重复率较低，换仓频繁，交易成本可能较高")
            
            # 绘制重复率分布图
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # 左图：重复率时间序列
            ax1 = axes[0]
            ax1.plot(rebalance_list[1:], overlap_rates, 'o-', color='#3498db', markersize=4)
            ax1.axhline(avg_overlap_rate, color='red', linestyle='--', label=f'平均: {avg_overlap_rate:.1%}')
            ax1.set_xlabel("换仓日期")
            ax1.set_ylabel("重复率")
            ax1.set_title("连续换仓间的持仓重复率")
            ax1.legend()
            ax1.tick_params(axis='x', rotation=45)
            
            # 右图：股票持有周期分布
            ax2 = axes[1]
            freq_counts = freq_series.value_counts().sort_index()
            ax2.bar(freq_counts.index, freq_counts.values, color='#2ecc71', alpha=0.8, edgecolor='black')
            ax2.set_xlabel("持有轮数")
            ax2.set_ylabel("股票数量")
            ax2.set_title(f"股票持有周期分布（共{len(all_symbols)}只股票）")
            ax2.axvline(avg_appearances, color='red', linestyle='--', label=f'平均: {avg_appearances:.1f}轮')
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.figure_dir, "holding_overlap.png"), dpi=120)
            plt.close()
            
            self.log(f"- 📊 持仓重复分析图已保存为 `holding_overlap.png`")

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

    def analyze_repeated_performance(self):
        """对比“新入选股票”与“重复入选股票”的未来表现"""
        self.log("\n### 重复入选 vs 新入选 表现对比\n")
        
        # 1. 准备数据
        horizon = self.config.get("preprocessing", {}).get("labels", {}).get("horizon", 5)
        # 确保all_df_sorted存在
        if not hasattr(self, 'all_df_sorted'):
             self.all_df_sorted = self.all_df.sort_values(by=["symbol", "date"])

        # 查找或计算未来收益列
        available_periods = [int(c.split('_')[1][:-1]) for c in self.all_df_sorted.columns if c.startswith('future_') and c.endswith('d')]
        if not available_periods:
             # 如果上一步没算出，默认算4d
             eval_period = 4 
             col_name = f"future_{eval_period}d"
             self.all_df_sorted[col_name] = self.all_df_sorted.groupby("symbol")["close"].transform(
                lambda x: (x.shift(-eval_period) / x - 1) * 100
             )
        else:
            eval_period = min(available_periods, key=lambda x: abs(x - horizon))
            col_name = f"future_{eval_period}d"
        
        self.log(f"- 评估周期: 未来{eval_period}日收益 (与策略周期 {horizon}日 最接近)")

        # 2. 向量化判定 "Repeated"
        all_dates = sorted(self.all_df["date"].unique())
        date_map = {d: i for i, d in enumerate(all_dates)}
        
        sig_df = self.signal_df.copy()
        sig_df["date_idx"] = sig_df["date"].map(date_map)
        sig_df = sig_df.sort_values(["symbol", "date_idx"])
        
        # 计算该股票上一次入选的日期索引
        sig_df["prev_date_idx"] = sig_df.groupby("symbol")["date_idx"].shift(1)
        
        # 只有在 rolling 模式下（gap=1）才算严格重复
        sig_df["is_repeated"] = (sig_df["date_idx"] - sig_df["prev_date_idx"]) == 1
        sig_df["type"] = sig_df["is_repeated"].map({True: "Repeated", False: "New"})
        
        # 3. 关联未来收益
        future_ret_subset = self.all_df_sorted[["date", "symbol", col_name]].dropna()
        analysis_df = sig_df.merge(future_ret_subset, on=["date", "symbol"], how="inner")
        
        if analysis_df.empty:
            self.log("⚠️ 无法关联未来收益数据，可能数据不足")
            return

        # 4. 统计与分析
        grouped = analysis_df.groupby("type")[col_name]
        stats = grouped.agg(["mean", "median", "count"])
        stats["win_rate"] = grouped.apply(lambda x: (x > 0).mean() * 100)
        
        self.log("\n" + stats.to_string(float_format="{:.2f}".format))
        
        try:
            rep_mean = stats.loc["Repeated", "mean"]
            new_mean = stats.loc["New", "mean"]
            if rep_mean > new_mean:
                self.log(f"\n✅ **结论**: 重复入选股票表现更优 (Mean: {rep_mean:.2f}% vs {new_mean:.2f}%)，信号具有趋势持续性。")
            else:
                self.log(f"\n⚠️ **注意**: 重复入选股票表现较弱 (Mean: {rep_mean:.2f}% vs {new_mean:.2f}%)，需警惕动量衰竭。")
        except KeyError:
            pass

        # 5. 绘图
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=analysis_df, x="type", y=col_name, hue="type", palette="Set2", showfliers=False, legend=False)
        plt.axhline(0, color="gray", linestyle="--", linewidth=1)
        plt.title(f"New vs Repeated Selection: Future {eval_period}d Return")
        plt.ylabel("Future Return (%)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_dir, "repeated_vs_new_perf.png"), dpi=120)
        plt.close()
        self.log(f"- 📊 对比图已保存: `repeated_vs_new_perf.png`")

    def run(self):
        self.load_data()
        if self.signal_df.empty:
            print("未生成信号，终止诊断。")
            return
        self.analyze_risk_exposure()
        self.analyze_signal_quality()
        self.analyze_repeated_performance()
        with open(self.report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.report_lines))
        print(f"诊断报告已完成：\n- Markdown报告: {self.report_path}\n- 图表目录: {self.figure_dir}")

if __name__ == "__main__":
    diag = SignalDiagnosis()
    diag.run()