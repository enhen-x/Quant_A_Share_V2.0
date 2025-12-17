# src/backtest/backtester.py

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from src.utils.logger import get_logger
from src.utils.config import GLOBAL_CONFIG
from src.utils.io import read_parquet, ensure_dir

logger = get_logger()

class VectorBacktester:
    def __init__(self):
        self.config = GLOBAL_CONFIG
        self.paths = self.config["paths"]
        
        strategy_cfg = self.config.get("strategy", {})
        self.holding_period = self.config.get("preprocessing", {}).get("labels", {}).get("horizon", 5)
        self.rebalance_mode = strategy_cfg.get("rebalance_mode", "rolling")
        
        # === 新增：买入价格模式 ===
        # "open": T+1 开盘价买入
        # "vwap": T+1 均价买入 (Volume Weighted Average Price)
        # "close": T+1 收盘价买入
        self.entry_price = strategy_cfg.get("entry_price", "close")
        logger.info(f"回测买入模式: {self.entry_price}")
        
        # 默认成本 (会被 run 方法的参数覆盖)
        self.default_cost_rate = 0.002
        
    def load_data(self):
        """加载行情数据并计算每日收益率"""
        path = os.path.join(self.paths["data_processed"], "all_stocks.parquet")
        df = read_parquet(path)
        
        df = df.sort_values(["symbol", "date"])
        
        # === 计算 VWAP (均价) ===
        df["vwap"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
        
        # === 根据 entry_price 模式计算首日收益 ===
        # 首日收益 = (T+1收盘 / 买入价) - 1
        # 后续日收益 = close-to-close
        
        if self.entry_price == "t_close":
            # T日收盘买入
            df["entry_price"] = df["close"]
        elif self.entry_price == "open":
            # T+1开盘买入
            df["entry_price"] = df.groupby("symbol")["open"].shift(-1)
        elif self.entry_price == "vwap":
            # T+1均价买入
            df["entry_price"] = df.groupby("symbol")["vwap"].shift(-1)
        else:  # close
            # T+1收盘买入
            df["entry_price"] = df.groupby("symbol")["close"].shift(-1)
        
        # 首日收益：从买入价到T+1收盘
        df["next_close"] = df.groupby("symbol")["close"].shift(-1)
        df["first_day_ret"] = (df["next_close"] / df["entry_price"]) - 1.0
        
        # 后续日收益：close-to-close
        df["daily_ret"] = df.groupby("symbol")["close"].pct_change()
        
        # 涨停判断：T+1 开盘涨幅 > 9.5%，无法买入
        df["next_open"] = df.groupby("symbol")["open"].shift(-1)
        df["next_open_ret"] = (df["next_open"] / df["close"]) - 1.0
        
        all_dates = sorted(df["date"].unique())
        return df[["date", "symbol", "first_day_ret", "daily_ret", "next_open_ret"]].dropna(subset=["date", "symbol"]), all_dates

    def run(self, signal_df: pd.DataFrame, output_dir: str, 
            cost_rate: float = None, 
            start_date: str = None, 
            end_date: str = None):
        """
        执行回测 (支持参数覆盖，用于压力测试)
        :param cost_rate: 覆盖默认交易成本
        :param start_date: 强制开始日期 (用于测特定熊市)
        :param end_date: 强制结束日期
        """
        # 确定本次运行的成本
        current_cost = cost_rate if cost_rate is not None else self.default_cost_rate
        
        ensure_dir(output_dir)
        price_df, all_dates = self.load_data()
        
        # 1. 时间对齐与截断
        valid_signal_df = signal_df[signal_df["date"].isin(all_dates)].copy()
        if valid_signal_df.empty:
            return {}
            
        # 确定实际开始时间
        actual_start = start_date if start_date else valid_signal_df["date"].min()
        actual_start = pd.to_datetime(actual_start)
        
        # 确定实际结束时间
        if end_date:
            actual_end = pd.to_datetime(end_date)
            all_dates = sorted([d for d in all_dates if actual_start <= d <= actual_end])
            price_df = price_df[(price_df["date"] >= actual_start) & (price_df["date"] <= actual_end)]
        else:
            all_dates = sorted([d for d in all_dates if d >= actual_start])
            price_df = price_df[price_df["date"] >= actual_start]
            
        date_to_idx = {d: i for i, d in enumerate(all_dates)}
        idx_to_date = {i: d for i, d in enumerate(all_dates)}
        
        # 2. 模式选择
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

        # 3. 信号膨胀 - 简化版（只需要首日标记）
        expanded_signals = []
        target_signals = target_signals[target_signals["date"] >= actual_start]
        if end_date:
            target_signals = target_signals[target_signals["date"] <= actual_end]

        for signal_date, group in target_signals.groupby("date"):
            if signal_date not in date_to_idx: continue
            start_idx = date_to_idx[signal_date]

            indices = [start_idx + 1 + i for i in range(0, self.holding_period)]
            valid_indices = [i for i in indices if i < len(all_dates)]
            if not valid_indices: continue
            
            current_target_dates = [idx_to_date[i] for i in valid_indices]
            for day_offset, holding_date in enumerate(current_target_dates):
                daily_slice = group[["symbol", "weight"]].copy()
                daily_slice["date"] = holding_date
                daily_slice["real_weight"] = daily_slice["weight"] * weight_factor
                # === 只在首日记录收益，因为 holding_ret 已经包含整个持有期 ===
                daily_slice["is_first_day"] = (day_offset == 0)
                daily_slice["signal_date"] = signal_date
                expanded_signals.append(daily_slice)
        
        if not expanded_signals: 
            return {"annual_return": 0, "sharpe": 0, "max_drawdown": 0}

        full_holdings = pd.concat(expanded_signals, ignore_index=True)
        
        # 4. 计算收益 - 使用每日收益 (close-to-close)
        
        # 分离首日和后续日持仓
        first_day_holdings = full_holdings[full_holdings["is_first_day"]].copy()
        later_day_holdings = full_holdings[~full_holdings["is_first_day"]].copy()
        
        # 4.1 首日持仓：用 signal_date 匹配 first_day_ret
        if len(first_day_holdings) > 0:
            first_merged = pd.merge(
                first_day_holdings,
                price_df[["date", "symbol", "first_day_ret", "next_open_ret"]].rename(columns={"date": "signal_date"}),
                on=["signal_date", "symbol"],
                how="left"
            )
            # 涨停处理：涨停时无法买入
            limit_up_mask = first_merged["next_open_ret"] > 0.095
            first_merged["ret_to_use"] = first_merged["first_day_ret"].fillna(0.0)
            first_merged.loc[limit_up_mask, "ret_to_use"] = 0.0
            first_merged["contrib"] = first_merged["real_weight"] * first_merged["ret_to_use"]
        else:
            first_merged = pd.DataFrame()
        
        # 4.2 后续日持仓：用 holding_date 匹配 daily_ret
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
        
        # 4.3 合并所有持仓
        if len(first_merged) > 0 and len(later_merged) > 0:
            merged_all = pd.concat([first_merged, later_merged], ignore_index=True)
        elif len(first_merged) > 0:
            merged_all = first_merged
        elif len(later_merged) > 0:
            merged_all = later_merged
        else:
            return {"annual_return": 0, "sharpe": 0, "max_drawdown": 0}
        
        # 4.4 按日期聚合收益
        daily_ret = merged_all.groupby("date")["contrib"].sum()
        
        # 计算成本：每天换1/5仓位，每次换仓买入卖出各一次
        daily_cost = (1.0 / self.holding_period) * 2 * current_cost
        daily_ret_net = daily_ret - daily_cost
        
        # 复利累积
        equity_curve = (1 + daily_ret_net).cumprod()
        equity_curve.index = pd.to_datetime(equity_curve.index)
        
        # 加载 Benchmark
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
        
        # 计算指标 - 使用日收益
        metrics = self._calc_metrics(daily_ret_net)
        self._plot_result(equity_curve, benchmark_curve, metrics, output_dir, 
                         title_suffix=f"(Cost={current_cost*1000:.1f}‰)")
        
        # 输出最终指标
        logger.info(f"年化收益率: {metrics['annual_return']:.2%}")
        logger.info(f"夏普比率: {metrics['sharpe']:.2f}")
        logger.info(f"最大回撤: {metrics['max_drawdown']:.2%}")
        
        # [修改] 返回 equity_curve 便于外部高级分析
        metrics["equity_curve"] = equity_curve
        return metrics

    def _calc_metrics_period(self, period_ret, holding_period):
        """基于周期收益计算指标（复利）"""
        period_ret = period_ret.dropna()
        if len(period_ret) == 0:
            return {"annual_return": 0, "sharpe": 0, "max_drawdown": 0}
        
        # 复利累积
        equity = (1 + period_ret).cumprod()
        total_return = equity.iloc[-1] - 1  # 总收益率
        
        # 计算周期数
        n_periods = len(period_ret)
        
        # 年化（复利公式）：一年大约有 252/holding_period 个周期
        periods_per_year = 252 / holding_period
        annual_ret = (1 + total_return) ** (periods_per_year / n_periods) - 1
        
        # 夏普比率：基于周期收益的波动率
        period_vol = period_ret.std()
        vol_annual = period_vol * np.sqrt(periods_per_year)
        sharpe = (annual_ret - 0.02) / (vol_annual + 1e-9)
        
        # 最大回撤
        drawdown = equity / equity.cummax() - 1
        max_dd = drawdown.min()
        
        return {"annual_return": annual_ret, "sharpe": sharpe, "max_drawdown": max_dd}

    def _calc_metrics(self, daily_ret):
        daily_ret = daily_ret.dropna()
        if len(daily_ret) == 0:
            return {"annual_return": 0, "sharpe": 0, "max_drawdown": 0}
        
        # === 正确的年化收益率计算 ===
        # 1. 先计算累计收益率
        equity = (1 + daily_ret).cumprod()
        total_return = equity.iloc[-1] - 1  # 总收益率
        
        # 2. 根据实际交易天数年化
        n_days = len(daily_ret)
        annual_ret = (1 + total_return) ** (252 / n_days) - 1
        
        # 3. 计算夏普比率
        rf = 0.02
        vol = daily_ret.std() * np.sqrt(252)
        sharpe = (annual_ret - rf) / (vol + 1e-9)
        
        # 4. 计算最大回撤
        drawdown = equity / equity.cummax() - 1
        max_dd = drawdown.min()
        
        return {"annual_return": annual_ret, "sharpe": sharpe, "max_drawdown": max_dd}

    def _plot_result(self, equity, benchmark, metrics, out_dir, title_suffix=""):

        # ====================================================================
        # 【修复冲突点】在每次绘图前强制设置字体和减号，以应对外部样式覆盖
        # ====================================================================
        # 1. 设置中文字体列表，确保找到一个可用的中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
        # 2. 强制使用 ASCII 减号 ('-') 代替 Unicode 减号 ('\u2212')，解决警告
        plt.rcParams['axes.unicode_minus'] = False 
        # ====================================================================

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