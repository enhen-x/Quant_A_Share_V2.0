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
        # 注意：这里的 horizon 是持有的总天数
        # 如果您想 T+1 买，T+5 卖，那么涉及的天数是 1,2,3,4,5，共 5 天。建议在 config 中设为 5。
        self.holding_period = self.config.get("preprocessing", {}).get("labels", {}).get("horizon", 5)
        self.rebalance_mode = strategy_cfg.get("rebalance_mode", "rolling")
        
        self.default_cost_rate = 0.002
        
    def load_data(self):
        """加载行情并计算 VWAP 进出场的专用收益率"""
        path = os.path.join(self.paths["data_processed"], "all_stocks.parquet")
        df = read_parquet(path)
        
        df = df.sort_values(["symbol", "date"])
        
        # 1. 基础数据准备
        if "pct_chg" not in df.columns:
            df["pct_chg"] = df.groupby("symbol")["close"].pct_change()
        
        # 获取昨收 (用于计算出场日的收益)
        df["prev_close"] = df.groupby("symbol")["close"].shift(1)
            
        # 2. 计算 VWAP (如果可用)
        if "amount" in df.columns and "volume" in df.columns:
            # 加上极小值防止除零
            vwap = df["amount"] / (df["volume"] + 1e-8)
            # 停牌或无量时回退到 Close
            df["vwap"] = vwap.where(df["volume"] > 0, df["close"])
        else:
            df["vwap"] = df["close"]

        # 3. 计算三种场景的收益率
        
        # A. 入场日收益 (Entry Day): 买入价=VWAP, 结算价=Close
        # 收益 = (Close - VWAP) / VWAP
        df["ret_entry_vwap"] = (df["close"] / df["vwap"]) - 1.0
        
        # B. 出场日收益 (Exit Day): 昨结=Prev_Close, 卖出价=VWAP
        # 收益 = (VWAP - Prev_Close) / Prev_Close
        df["ret_exit_vwap"] = (df["vwap"] / df["prev_close"]) - 1.0
        
        # C. 持仓日收益 (Holding Day): 昨结=Prev_Close, 今结=Close
        # 收益 = pct_chg (已有)

        # 4. 特殊过滤：一字涨停无法买入
        # 如果 Open 涨停 (相对于昨收涨 > 9.5%)，则入场收益设为 0 (假设买不进)
        df["open_ret"] = (df["open"] / df["prev_close"]) - 1.0
        limit_up_mask = df["open_ret"] > 0.095
        df.loc[limit_up_mask, "ret_entry_vwap"] = 0.0
        
        # 5. 清理并返回
        cols = ["date", "symbol", "pct_chg", "ret_entry_vwap", "ret_exit_vwap"]
        return df[cols].dropna(), sorted(df["date"].unique())

    def run(self, signal_df: pd.DataFrame, output_dir: str, 
            cost_rate: float = None, 
            start_date: str = None, 
            end_date: str = None):
        """执行回测 (VWAP 进出场版)"""
        current_cost = cost_rate if cost_rate is not None else self.default_cost_rate
        ensure_dir(output_dir)
        price_df, all_dates = self.load_data()
        
        # 1. 时间对齐
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
        
        # 2. 信号膨胀 (Signal Expansion)
        expanded_signals = []
        target_signals = valid_signal_df[valid_signal_df["date"] >= actual_start]
        if end_date:
            target_signals = target_signals[target_signals["date"] <= actual_end]

        # 换仓模式权重系数
        weight_factor = 1.0
        if self.rebalance_mode == "rolling":
            weight_factor = 1.0 / self.holding_period

        for date, group in target_signals.groupby("date"):
            if date not in date_to_idx: continue
            start_idx = date_to_idx[date]
            
            # 生成持有期索引：从 T+1 到 T+Horizon
            # 例如 horizon=5，则持有 T+1, T+2, T+3, T+4, T+5 (共5天)
            indices = [start_idx + 1 + i for i in range(self.holding_period)]
            
            valid_indices = [i for i in indices if i < len(all_dates)]
            if not valid_indices: continue
            
            # 遍历这笔交易持有的每一天，并打上标记
            for i_step, global_idx in enumerate(valid_indices):
                d = idx_to_date[global_idx]
                daily_slice = group[["symbol", "weight"]].copy()
                daily_slice["date"] = d
                daily_slice["real_weight"] = daily_slice["weight"] * weight_factor
                
                # === 关键逻辑：区分入场、持仓、出场 ===
                if i_step == 0:
                    # 第一天：VWAP 入场
                    daily_slice["ret_type"] = "entry"
                elif i_step == self.holding_period - 1:
                    # 最后一天：VWAP 出场
                    daily_slice["ret_type"] = "exit"
                else:
                    # 中间天：持有
                    daily_slice["ret_type"] = "hold"
                
                expanded_signals.append(daily_slice)
        
        if not expanded_signals: 
            return {"annual_return": 0, "sharpe": 0, "max_drawdown": 0}

        full_holdings = pd.concat(expanded_signals, ignore_index=True)
        
        # 3. 计算收益
        # 合并行情
        merged = pd.merge(full_holdings, price_df, on=["date", "symbol"], how="inner")
        
        # 根据标记选择正确的收益率列
        conditions = [
            (merged["ret_type"] == "entry"),
            (merged["ret_type"] == "exit")
        ]
        choices = [
            merged["ret_entry_vwap"],  # 入场日：Close / VWAP - 1
            merged["ret_exit_vwap"]    # 出场日：VWAP / Prev_Close - 1
        ]
        # 默认为 pct_chg (持仓日：Close / Prev_Close - 1)
        merged["actual_ret"] = np.select(conditions, choices, default=merged["pct_chg"])
        
        # 计算贡献
        merged["contrib"] = merged["real_weight"] * merged["actual_ret"]
        daily_ret = merged.groupby("date")["contrib"].sum()
        
        # 4. 计算成本 (只在入场时扣除，或根据换手率扣除)
        # 简化版：按每日总持仓 * 基础换手率 * 费率
        # (因为每天都有 1/N 的仓位在轮动)
        daily_pos_ratio = merged.groupby("date")["real_weight"].sum()
        base_turnover = 1.0 / self.holding_period if self.rebalance_mode == "rolling" else 0
        
        # 对于 periodic 模式，通过 ret_type == entry 来精准计算换手
        if self.rebalance_mode == "periodic":
            # 找出当天是 Entry 的权重之和
            entry_weights = merged[merged["ret_type"] == "entry"].groupby("date")["real_weight"].sum()
            # 补全日期索引
            daily_turnover = entry_weights.reindex(daily_ret.index, fill_value=0.0)
            adjusted_daily_cost = daily_turnover * 2 * current_cost # 买卖各一次
        else:
            # Rolling 模式近似计算
            adjusted_daily_cost = base_turnover * 2 * current_cost * daily_pos_ratio
        
        daily_ret_net = daily_ret - adjusted_daily_cost
        
        # 5. 结果统计与绘图
        equity_curve = (1 + daily_ret_net).cumprod()
        metrics = self._calc_metrics(daily_ret_net)
        
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
        
        self._plot_result(equity_curve, benchmark_curve, metrics, output_dir, 
                         title_suffix=f"(Cost={current_cost*1000:.1f}‰)")
        
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
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False 
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # 线性坐标
        ax1 = axes[0]
        equity.plot(ax=ax1, label="Strategy", color="#d62728", linewidth=1.5)
        if benchmark is not None:
            benchmark.plot(ax=ax1, label="Benchmark", color="gray", linestyle="--", alpha=0.6)
        
        title_str = (f"Backtest {title_suffix}\n"
                     f"Ann Ret: {metrics['annual_return']:.1%} | "
                     f"Sharpe: {metrics['sharpe']:.2f} | "
                     f"MaxDD: {metrics['max_drawdown']:.1%}")
        ax1.set_title(title_str)
        ax1.legend(loc="upper left")
        ax1.grid(True, linestyle="--", alpha=0.3)
        
        # 对数坐标
        ax2 = axes[1]
        equity.plot(ax=ax2, label="Strategy", color="#d62728", linewidth=1.5)
        if benchmark is not None:
            benchmark.plot(ax=ax2, label="Benchmark", color="gray", linestyle="--", alpha=0.6)
        ax2.set_yscale('log') 
        ax2.set_title("Equity Curve (Log Scale)")
        ax2.legend(loc="upper left")
        ax2.grid(True, linestyle="--", alpha=0.3, which='both')
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "equity_curve.png"), dpi=100)
        plt.close()