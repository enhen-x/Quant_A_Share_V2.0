# src/backtest/backtester.py
# ============================================================================
# 向量化回测引擎 (VectorBacktester)
# ============================================================================

import pandas as pd
import numpy as np
import os
import sys
import io
import warnings
import logging

# 抑制所有字体 glyph 相关的警告
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', message='.*glyph.*')
warnings.filterwarnings('ignore', message='.*Glyph.*')
warnings.filterwarnings('ignore', message='.*\\u2212.*')

# 禁用 matplotlib 的字体警告日志
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
logging.getLogger('matplotlib.backends').setLevel(logging.ERROR)
logging.getLogger('matplotlib.texmanager').setLevel(logging.ERROR)

import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'sans-serif'],
    'font.family': 'sans-serif',
    'axes.unicode_minus': False,
    'mathtext.fontset': 'dejavusans',
    'text.usetex': False,
})

from src.utils.logger import get_logger
from src.utils.config import GLOBAL_CONFIG
from src.utils.io import read_parquet, ensure_dir

logger = get_logger()


class VectorBacktester:
    """向量化回测引擎"""
    
    def __init__(self):
        self.config = GLOBAL_CONFIG
        self.paths = self.config["paths"]
        
        strategy_cfg = self.config.get("strategy", {})
        self.holding_period = self.config.get("preprocessing", {}).get("labels", {}).get("horizon", 5)
        self.rebalance_mode = strategy_cfg.get("rebalance_mode", "rolling")
        self.entry_price = strategy_cfg.get("entry_price", "close")
        logger.info(f"回测买入模式: {self.entry_price}")
        self.default_cost_rate = 0.002
        
    def load_data(self):
        """加载股票行情数据并预计算收益率"""
        path = os.path.join(self.paths["data_processed"], "all_stocks.parquet")
        df = read_parquet(path)
        df = df.sort_values(["symbol", "date"])
        
        df["vwap"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
        
        if self.entry_price == "t_close":
            df["entry_price"] = df["close"]
            df["mark_price"] = df["close"]
        elif self.entry_price == "open":
            df["entry_price"] = df.groupby("symbol")["open"].shift(-1)
            df["mark_price"] = df["close"]
        elif self.entry_price == "vwap":
            df["entry_price"] = df.groupby("symbol")["vwap"].shift(-1)
            df["mark_price"] = df["vwap"]
        else:
            df["entry_price"] = df.groupby("symbol")["close"].shift(-1)
            df["mark_price"] = df["close"]
        
        df["next_mark_price"] = df.groupby("symbol")["mark_price"].shift(-1)
        df["first_day_ret"] = (df["next_mark_price"] / df["entry_price"]) - 1.0
        df["daily_ret"] = df.groupby("symbol")["mark_price"].pct_change()
        df["exit_price"] = df.groupby("symbol")["mark_price"].shift(-(self.holding_period + 1))
        df["holding_ret"] = (df["exit_price"] / df["entry_price"]) - 1.0
        df["next_open"] = df.groupby("symbol")["open"].shift(-1)
        df["next_open_ret"] = (df["next_open"] / df["close"]) - 1.0
        
        all_dates = sorted(df["date"].unique())
        return df[["date", "symbol", "first_day_ret", "daily_ret", "holding_ret", "next_open_ret"]].dropna(subset=["date", "symbol"]), all_dates

    def run(self, signal_df: pd.DataFrame, output_dir: str, 
            cost_rate: float = None, start_date: str = None, end_date: str = None):
        """执行回测主流程"""
        current_cost = cost_rate if cost_rate is not None else self.default_cost_rate
        ensure_dir(output_dir)
        price_df, all_dates = self.load_data()
        
        valid_signal_df = signal_df[signal_df["date"].isin(all_dates)].copy()
        if valid_signal_df.empty:
            return {}
            
        actual_start = start_date if start_date else valid_signal_df["date"].min()
        actual_start = pd.to_datetime(actual_start)
        
        if end_date:
            actual_end = pd.to_datetime(end_date)
            all_dates = sorted([d for d in all_dates if actual_start <= d <= actual_end])
            price_df = price_df[(price_df["date"] >= actual_start) & (price_df["date"] <= actual_end)]
        else:
            all_dates = sorted([d for d in all_dates if d >= actual_start])
            price_df = price_df[price_df["date"] >= actual_start]
        
        date_to_idx = {d: i for i, d in enumerate(all_dates)}
        idx_to_date = {i: d for i, d in enumerate(all_dates)}
        
        target_signals = pd.DataFrame()
        weight_factor = 1.0
        
        if self.rebalance_mode == "periodic":
            rebalance_indices = range(0, len(all_dates), self.holding_period)
            rebalance_dates = [idx_to_date[i] for i in rebalance_indices]
            target_signals = valid_signal_df[valid_signal_df["date"].isin(rebalance_dates)]
            weight_factor = 1.0
        else:
            target_signals = valid_signal_df
            weight_factor = 1.0 / self.holding_period

        target_signals = target_signals[target_signals["date"] >= actual_start]
        if end_date:
            target_signals = target_signals[target_signals["date"] <= actual_end]

        expanded_signals = []
        
        for signal_date, group in target_signals.groupby("date"):
            if signal_date not in date_to_idx:
                continue
            start_idx = date_to_idx[signal_date]
            indices = [start_idx + 1 + i for i in range(self.holding_period)]
            valid_indices = [i for i in indices if i < len(all_dates)]
            if not valid_indices:
                continue
            
            current_target_dates = [idx_to_date[i] for i in valid_indices]
            
            for day_offset, holding_date in enumerate(current_target_dates):
                daily_slice = group[["symbol", "weight"]].copy()
                daily_slice["date"] = holding_date
                daily_slice["real_weight"] = daily_slice["weight"] * weight_factor
                daily_slice["is_first_day"] = (day_offset == 0)
                daily_slice["signal_date"] = signal_date
                expanded_signals.append(daily_slice)
        
        if not expanded_signals:
            return {"annual_return": 0, "sharpe": 0, "max_drawdown": 0}
        
        full_holdings = pd.concat(expanded_signals, ignore_index=True)
        
        first_day_holdings = full_holdings[full_holdings["is_first_day"]].copy()
        later_day_holdings = full_holdings[~full_holdings["is_first_day"]].copy()
        
        if len(first_day_holdings) > 0:
            first_merged = pd.merge(
                first_day_holdings,
                price_df[["date", "symbol", "first_day_ret", "next_open_ret"]].rename(columns={"date": "signal_date"}),
                on=["signal_date", "symbol"],
                how="left"
            )
            limit_up_mask = first_merged["next_open_ret"] > 0.095
            first_merged["ret_to_use"] = first_merged["first_day_ret"].fillna(0.0)
            first_merged.loc[limit_up_mask, "ret_to_use"] = 0.0
            first_merged["contrib"] = first_merged["real_weight"] * first_merged["ret_to_use"]
        else:
            first_merged = pd.DataFrame()
        
        if len(later_day_holdings) > 0:
            later_merged = pd.merge(
                later_day_holdings,
                price_df[["date", "symbol", "daily_ret"]],
                on=["date", "symbol"],
                how="left"
            )
            later_merged["ret_to_use"] = later_merged["daily_ret"].fillna(0.0)
            later_merged["contrib"] = later_merged["real_weight"] * later_merged["ret_to_use"]
        else:
            later_merged = pd.DataFrame()
        
        if len(first_merged) > 0 and len(later_merged) > 0:
            merged_all = pd.concat([first_merged, later_merged], ignore_index=True)
        elif len(first_merged) > 0:
            merged_all = first_merged
        elif len(later_merged) > 0:
            merged_all = later_merged
        else:
            return {"annual_return": 0, "sharpe": 0, "max_drawdown": 0}
        
        daily_ret = merged_all.groupby("date")["contrib"].sum()
        daily_cost = (1.0 / self.holding_period) * current_cost
        daily_ret_net = daily_ret - daily_cost
        
        equity_curve = (1 + daily_ret_net).cumprod()
        equity_curve.index = pd.to_datetime(equity_curve.index)
        
        metrics = self._calc_metrics(daily_ret_net)
        
        logger.info("=" * 60)
        logger.info(f"【收益诊断分析 - {self.rebalance_mode} 模式】")
        logger.info("=" * 60)
        logger.info(f"持仓周期: {self.holding_period} 天")
        logger.info(f"交易日数: {len(daily_ret)}")
        logger.info(f"日收益 - 均值: {daily_ret.mean():.6f} ({daily_ret.mean()*100:.4f}%)")
        logger.info(f"日收益 - 标准差: {daily_ret.std():.6f}")
        
        batch_per_day = merged_all.groupby("date").apply(lambda x: x["signal_date"].nunique(), include_groups=False)
        logger.info(f"每日持仓批次数 - 均值: {batch_per_day.mean():.2f}")
        logger.info("=" * 60)
        
        idx_code = self.config.get("preprocessing", {}).get("labels", {}).get("index_code", "000300.SH")
        idx_file = os.path.join(self.paths["data_raw"], f"index_{idx_code.replace('.', '')}.parquet")
        benchmark_curve = None
        if os.path.exists(idx_file):
            idx_df = read_parquet(idx_file)
            idx_df["date"] = pd.to_datetime(idx_df["date"])
            idx_df = idx_df.set_index("date").sort_index()
            common = equity_curve.index.intersection(idx_df.index)
            if not common.empty:
                idx_sub = idx_df.loc[common, "close"]
                benchmark_curve = idx_sub / idx_sub.iloc[0]
        
        self._plot_result(equity_curve, benchmark_curve, metrics, output_dir, 
                         title_suffix=f"(Cost={current_cost*1000:.1f}‰)")
        
        if benchmark_curve is not None:
            self._plot_daily_comparison(daily_ret_net, benchmark_curve, output_dir)
        
        logger.info(f"年化收益率: {metrics['annual_return']:.2%}")
        logger.info(f"夏普比率: {metrics['sharpe']:.2f}")
        logger.info(f"最大回撤: {metrics['max_drawdown']:.2%}")
        
        metrics["equity_curve"] = equity_curve
        return metrics

    def _calc_metrics(self, daily_ret):
        """基于日收益计算绩效指标"""
        daily_ret = daily_ret.dropna()
        if len(daily_ret) == 0:
            return {"annual_return": 0, "sharpe": 0, "max_drawdown": 0}
        
        equity = (1 + daily_ret).cumprod()
        total_return = equity.iloc[-1] - 1
        n_days = len(daily_ret)
        annual_ret = (1 + total_return) ** (252 / n_days) - 1
        
        rf = 0.02
        vol = daily_ret.std() * np.sqrt(252)
        sharpe = (annual_ret - rf) / (vol + 1e-9)
        
        drawdown = equity / equity.cummax() - 1
        max_dd = drawdown.min()
        
        return {"annual_return": annual_ret, "sharpe": sharpe, "max_drawdown": max_dd}

    def _plot_result(self, equity, benchmark, metrics, out_dir, title_suffix=""):
        """绘制回测结果图表"""
        plt.rcParams.update({
            'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'sans-serif'],
            'font.family': 'sans-serif',
            'axes.unicode_minus': False,
            'mathtext.fontset': 'dejavusans',
        })
        
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()

        try:
            fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
            ax1 = axes[0]
            equity.plot(ax=ax1, label="Strategy", color="red", linewidth=2)
            if benchmark is not None:
                benchmark.plot(ax=ax1, label="Benchmark", color="gray", linestyle="--", alpha=0.7)
            ax1.set_title(f"Backtest {title_suffix}\nAnn Ret: {metrics['annual_return']:.1%} | Sharpe: {metrics['sharpe']:.2f} | MaxDD: {metrics['max_drawdown']:.1%}")
            ax1.legend(loc="upper left")
            ax1.grid(True, linestyle="--", alpha=0.5)
            
            ax2 = axes[1]
            equity.plot(ax=ax2, label="Strategy", color="red", linewidth=2)
            if benchmark is not None:
                benchmark.plot(ax=ax2, label="Benchmark", color="gray", linestyle="--", alpha=0.7)
            ax2.set_yscale('log')
            ax2.set_title("Backtest (Log Scale)")
            ax2.legend(loc="upper left")
            ax2.grid(True, linestyle="--", alpha=0.5, which='both')
            
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "equity_curve.png"))
            plt.close()
        finally:
            sys.stderr = old_stderr

    def _plot_daily_comparison(self, daily_ret_strategy, benchmark_curve, out_dir):
        """绘制策略与大盘的对比分析图"""
        plt.rcParams.update({
            'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'sans-serif'],
            'font.family': 'sans-serif',
            'axes.unicode_minus': False,
            'mathtext.fontset': 'dejavusans',
            'text.usetex': False,
        })
        
        old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        
        try:
            benchmark_ret = benchmark_curve.pct_change().fillna(0)
            
            common_dates = daily_ret_strategy.index.intersection(benchmark_ret.index)
            if len(common_dates) == 0:
                return
            
            strategy_ret = daily_ret_strategy.loc[common_dates]
            bench_ret = benchmark_ret.loc[common_dates]
            excess_ret = strategy_ret - bench_ret
            
            strategy_cumret = (1 + strategy_ret).cumprod()
            bench_cumret = (1 + bench_ret).cumprod()
            
            fig = plt.figure(figsize=(16, 12))
            
            # 子图1: 累计收益曲线对比
            ax1 = plt.subplot(3, 1, 1)
            ax1.plot(common_dates, strategy_cumret.values, 
                    label='策略累计净值', color='red', linewidth=2.5, alpha=0.9)
            ax1.plot(common_dates, bench_cumret.values, 
                    label='大盘累计净值', color='gray', linewidth=2, linestyle='--', alpha=0.7)
            
            ax1.fill_between(common_dates, strategy_cumret.values, bench_cumret.values,
                             where=(strategy_cumret.values >= bench_cumret.values),
                             color='green', alpha=0.2, label='跑赢区域')
            ax1.fill_between(common_dates, strategy_cumret.values, bench_cumret.values,
                             where=(strategy_cumret.values < bench_cumret.values),
                             color='red', alpha=0.2, label='跑输区域')
            
            ax1.axhline(y=1.0, color='black', linestyle='-', linewidth=0.8)
            ax1.set_ylabel('累计净值', fontsize=12, fontweight='bold')
            ax1.set_yscale('log')
            ax1.set_title('策略 vs 大盘 - 累计净值对比 (对数刻度)', fontsize=14, fontweight='bold')
            ax1.legend(loc='best', fontsize=9, framealpha=0.9)
            ax1.grid(True, alpha=0.3, which='both')
            
            import matplotlib.dates as mdates
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax1.xaxis.set_major_locator(mdates.YearLocator())
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            final_strategy = (strategy_cumret.iloc[-1] - 1) * 100
            final_bench = (bench_cumret.iloc[-1] - 1) * 100
            excess_total = final_strategy - final_bench
            
            stats_box = f"策略总收益: {final_strategy:.1f}%\n"
            stats_box += f"大盘总收益: {final_bench:.1f}%\n"
            stats_box += f"超额收益: {excess_total:.1f}%"
            ax1.text(0.98, 0.98, stats_box, transform=ax1.transAxes,
                    fontsize=10, verticalalignment='top', horizontalalignment='right',
                    fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
            
            # 子图2: 30日滚动累计收益
            ax2 = plt.subplot(3, 1, 2)
            rolling_window = 30
            rolling_strategy = pd.Series((1 + strategy_ret.values)).rolling(rolling_window).apply(lambda x: (x.prod() - 1) * 100, raw=True)
            rolling_bench = pd.Series((1 + bench_ret.values)).rolling(rolling_window).apply(lambda x: (x.prod() - 1) * 100, raw=True)
            
            ax2.plot(common_dates, rolling_strategy, color='red', linewidth=2, label=f'{rolling_window}日策略收益', alpha=0.9)
            ax2.plot(common_dates, rolling_bench, color='gray', linewidth=2, linestyle='--', label=f'{rolling_window}日大盘收益', alpha=0.7)
            
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.0)
            ax2.fill_between(common_dates, 0, rolling_strategy, where=(rolling_strategy > 0), color='red', alpha=0.2)
            ax2.fill_between(common_dates, 0, rolling_bench, where=(rolling_bench > 0), color='gray', alpha=0.2)
            
            ax2.set_ylabel(f'{rolling_window}日滚动累计收益 (%)', fontsize=12)
            ax2.set_title(f'{rolling_window}日滚动累计收益对比', fontsize=12)
            ax2.legend(loc='best', fontsize=9, framealpha=0.9)
            ax2.grid(True, alpha=0.3)
            
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax2.xaxis.set_major_locator(mdates.YearLocator())
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # 子图3: 30日滚动超额收益
            ax3 = plt.subplot(3, 1, 3)
            rolling_excess_cum = pd.Series((1 + excess_ret.values)).rolling(rolling_window).apply(lambda x: (x.prod() - 1) * 100, raw=True)
            
            colors = ['green' if x > 0 else 'red' for x in rolling_excess_cum]
            ax3.bar(common_dates, rolling_excess_cum, color=colors, alpha=0.6, width=1.0)
            
            ax3.axhline(y=0, color='black', linestyle='-', linewidth=1.0)
            ax3.fill_between(common_dates, 0, rolling_excess_cum, 
                             where=(rolling_excess_cum > 0), 
                             color='green', alpha=0.2)
            ax3.fill_between(common_dates, 0, rolling_excess_cum, 
                             where=(rolling_excess_cum <= 0), 
                             color='red', alpha=0.2)
            
            ax3.set_ylabel(f'{rolling_window}日滚动超额收益 (%)', fontsize=12, fontweight='bold')
            ax3.set_xlabel('日期', fontsize=11)
            ax3.set_title(f'{rolling_window}日滚动累计超额收益 (绿色=跑赢, 红色=跑输)', fontsize=12)
            ax3.grid(True, alpha=0.3)
            
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax3.xaxis.set_major_locator(mdates.YearLocator())
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            win_days = (excess_ret > 0).sum()
            total_days = len(excess_ret)
            win_rate = win_days / total_days
            avg_win = excess_ret[excess_ret > 0].mean() * 100 if (excess_ret > 0).any() else 0
            avg_loss = excess_ret[excess_ret < 0].mean() * 100 if (excess_ret < 0).any() else 0
            
            cumulative_excess = (1 + excess_ret).cumprod() - 1
            
            stats_text = f"跑赢天数: {win_days}/{total_days} ({win_rate*100:.1f}%)\n"
            stats_text += f"累计超额: {cumulative_excess.iloc[-1]*100:.2f}%\n"
            stats_text += f"平均跑赢日: +{avg_win:.3f}%\n"
            stats_text += f"平均跑输日: {avg_loss:.3f}%"
            
            ax3.text(0.98, 0.98, stats_text, transform=ax3.transAxes,
                    fontsize=9, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "daily_comparison.png"), dpi=120)
            plt.close()
            
            logger.info(f"每日对比图已保存: {os.path.join(out_dir, 'daily_comparison.png')}")
            logger.info(f"策略总收益: {final_strategy:.2f}% | 大盘总收益: {final_bench:.2f}%")
            logger.info(f"累计超额收益: {excess_total:.2f}%")
            logger.info(f"跑赢天数: {win_days}/{total_days} ({win_rate*100:.1f}%)")
        finally:
            sys.stderr = old_stderr