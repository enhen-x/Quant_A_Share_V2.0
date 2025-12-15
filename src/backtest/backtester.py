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
        
        # 默认成本 (会被 run 方法的参数覆盖)
        self.default_cost_rate = 0.002
        
    def load_data(self):
        """加载行情并增加【严谨成交】检查 (保持不变)"""
        path = os.path.join(self.paths["data_processed"], "all_stocks.parquet")
        # logger.info(f"回测引擎正在加载行情数据: {path}") # 注释掉避免刷屏
        df = read_parquet(path)
        
        df = df.sort_values(["symbol", "date"])
        
        if "pct_chg" not in df.columns:
            df["pct_chg"] = df.groupby("symbol")["close"].pct_change()
            
        # T+1 数据准备
        df["next_close"] = df.groupby("symbol")["close"].shift(-1)
        df["next_open"]  = df.groupby("symbol")["open"].shift(-1)
        
        # 1. T+1 开盘涨幅
        df["next_open_ret"] = (df["next_open"] / df["close"]) - 1.0
        
        # 2. 原始次日收益
        df["next_pct_chg"] = df.groupby("symbol")["pct_chg"].shift(-1)
        
        # 3. 标记涨停无法买入 (开盘涨幅 > 9.5%)
        limit_up_mask = df["next_open_ret"] > 0.095
        df.loc[limit_up_mask, "next_pct_chg"] = 0.0
        
        all_dates = sorted(df["date"].unique())
        return df[["date", "symbol", "next_pct_chg"]].dropna(), all_dates

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

        # 3. 信号膨胀
        expanded_signals = []
        # 只遍历还在目标时间段内的信号
        # 注意：如果指定了 start_date，我们要确保只处理该日期之后的信号
        target_signals = target_signals[target_signals["date"] >= actual_start]
        if end_date:
            target_signals = target_signals[target_signals["date"] <= actual_end]

        for date, group in target_signals.groupby("date"):
            if date not in date_to_idx: continue
            start_idx = date_to_idx[date]
            


            indices = [start_idx + 1 +i for i in range(0, self.holding_period)]
            valid_indices = [i for i in indices if i < len(all_dates)]
            if not valid_indices: continue
            
            current_target_dates = [idx_to_date[i] for i in valid_indices]
            for d in current_target_dates:
                daily_slice = group[["symbol", "weight"]].copy()
                daily_slice["date"] = d
                daily_slice["real_weight"] = daily_slice["weight"] * weight_factor
                expanded_signals.append(daily_slice)
        
        if not expanded_signals: 
            return {"annual_return": 0, "sharpe": 0, "max_drawdown": 0}

        full_holdings = pd.concat(expanded_signals, ignore_index=True)
        daily_positions = full_holdings.groupby(["date", "symbol"], as_index=False)["real_weight"].sum()
        
        # 4. 计算收益
        merged = pd.merge(daily_positions, price_df, on=["date", "symbol"], how="inner")
        merged["contrib"] = merged["real_weight"] * merged["next_pct_chg"]
        daily_ret = merged.groupby("date")["contrib"].sum()
        
        # === 修复开始 ===
        # 1. 计算当天的实际总仓位 (例如满仓是1.0，半仓是0.5，空仓是0.0)
        daily_pos_ratio = merged.groupby("date")["real_weight"].sum()
        
        # 2. 计算理论基础换手率
        base_turnover = 1.0 / self.holding_period
        
        # 3. 动态计算成本：只有持仓的部分才承担换仓成本
        # 如果 daily_pos_ratio 为 0，则 adjusted_daily_cost 为 0
        adjusted_daily_cost = base_turnover * 2 * current_cost * daily_pos_ratio
        
        # 4. 净收益 = 原始收益 - 动态成本
        daily_ret_net = daily_ret - adjusted_daily_cost
        # === 修复结束 ===
        
        # 5. 结果
        equity_curve = (1 + daily_ret_net).cumprod()
        
        # 加载 Benchmark (这里简单处理，如果截断了时间，benchmark 也要截断)
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
        
        metrics = self._calc_metrics(daily_ret_net)
        self._plot_result(equity_curve, benchmark_curve, metrics, output_dir, 
                         title_suffix=f"(Cost={current_cost*1000:.1f}‰)")
        
        # [修改] 返回 equity_curve 便于外部高级分析
        metrics["equity_curve"] = equity_curve
        return metrics

    def _calc_metrics(self, daily_ret):
        daily_ret = daily_ret.dropna()
        if len(daily_ret) == 0:
            return {"annual_return": 0, "sharpe": 0, "max_drawdown": 0}
        annual_ret = (1 + daily_ret.mean()) ** 252 - 1
        rf = 0.02
        vol = daily_ret.std() * np.sqrt(252)
        sharpe = (annual_ret - rf) / (vol + 1e-9)
        equity = (1 + daily_ret).cumprod()
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