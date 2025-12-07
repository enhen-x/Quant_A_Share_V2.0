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
        
        # 策略参数
        strategy_cfg = self.config.get("strategy", {})
        # 从配置读取持有天数，如果没写默认 5 天 (对应 horizon)
        self.holding_period = self.config.get("preprocessing", {}).get("labels", {}).get("horizon", 5)
        
        # 交易成本 (双边万分之三佣金 + 卖出千一印花税 + 滑点 ≈ 单边千 1.5)
        # 适当调高成本以模拟真实滑点
        self.cost_rate = 0.0015 
        
    def load_data(self):
        """加载全量行情并计算【次日】收益"""
        path = os.path.join(self.paths["data_processed"], "all_stocks.parquet")
        logger.info(f"回测引擎正在加载行情数据: {path}")
        df = read_parquet(path)
        
        # 1. 确保按日期排序
        df = df.sort_values(["symbol", "date"])
        
        # 2. 计算当日收益 (如果还没算)
        if "pct_chg" not in df.columns:
            df["pct_chg"] = df.groupby("symbol")["close"].pct_change()
            
        # 3. 【核心修正 1】计算次日收益 (Next Day Return)
        # T日的 next_pct_chg = (T+1收盘 / T+1开盘) - 1 ... 理想情况
        # 简化 V2.0: (T+1收盘 / T收盘) - 1，即假设 T+1 享有了全天涨跌
        # 这一步消除了“未来函数”，T日的信号只能吃到 next_pct_chg
        df["next_pct_chg"] = df.groupby("symbol")["pct_chg"].shift(-1)
        
        # 提取全市场交易日历 (用于计算持有期)
        all_dates = sorted(df["date"].unique())
        
        return df[["date", "symbol", "next_pct_chg"]].dropna(), all_dates

    def run(self, signal_df: pd.DataFrame, output_dir: str):
        """
        执行分仓轮动回测
        :param signal_df: [date, symbol, weight] (T日生成的信号)
        """
        logger.info(f"=== 开始回测 (持有期: {self.holding_period} 天, 分仓轮动) ===")
        ensure_dir(output_dir)
        
        price_df, all_dates = self.load_data()
        
        # --- 核心逻辑：信号膨胀 (Signal Expansion) ---
        # 如果持有 5 天，那么 T 日的信号在 T+1, T+2, ..., T+5 都是有效的
        # 我们将信号按交易日历向后投射
        
        # 1. 建立日期索引映射 (Date -> Index)
        date_to_idx = {d: i for i, d in enumerate(all_dates)}
        idx_to_date = {i: d for i, d in enumerate(all_dates)}
        
        # 2. 扩展信号
        # 我们将 T 日的信号复制 5 份，分别对应未来 5 个交易日
        expanded_signals = []
        
        logger.info("正在构建分仓持仓明细...")
        
        # 为了加速，我们不使用循环处理每一天，而是使用 DataFrame 操作
        # 但考虑到交易日历的不连续性，这里用循环处理 signal_df 的 date 分组比较稳妥
        
        # 预处理：只保留在行情日期范围内的信号
        valid_signal_df = signal_df[signal_df["date"].isin(all_dates)].copy()
        
        for date, group in valid_signal_df.groupby("date"):
            start_idx = date_to_idx[date]
            
            # 找到未来 N 个交易日 (含 T+1 到 T+N)
            # 注意：backtester 这里逻辑是 T日信号 -> T+1日持仓
            # 所以持有期是 [idx+1, idx+2, ..., idx+N]
            
            indices = [start_idx + i for i in range(1, self.holding_period + 1)]
            # 过滤掉超出日历范围的
            valid_indices = [i for i in indices if i < len(all_dates)]
            
            if not valid_indices:
                continue
                
            target_dates = [idx_to_date[i] for i in valid_indices]
            
            # 将该日的信号复制给这几天
            # 权重除以 N (因为资金分成了 N 份)
            # 比如总仓位 1.0，分成 5 份，每份 0.2。
            # group["weight"] 已经是归一化到 1.0 的 (例如 10 只股各 0.1)
            # 那么每只股在总盘子里的实际权重是 0.1 * 0.2 = 0.02
            
            actual_weight_factor = 1.0 / self.holding_period
            
            for d in target_dates:
                # 构造当天的持仓片段
                daily_slice = group[["symbol", "weight"]].copy()
                daily_slice["date"] = d
                daily_slice["real_weight"] = daily_slice["weight"] * actual_weight_factor
                expanded_signals.append(daily_slice)
        
        if not expanded_signals:
            logger.error("未能生成有效持仓，请检查信号日期是否在行情范围内。")
            return {}

        # 合并所有片段，得到每日的总持仓
        full_holdings = pd.concat(expanded_signals, ignore_index=True)
        
        # 聚合：如果同一只股票由不同的轮动组合持有（虽然 TopK 策略通常不会），累加权重
        daily_positions = full_holdings.groupby(["date", "symbol"], as_index=False)["real_weight"].sum()
        
        # --- 计算收益 ---
        logger.info("正在计算每日收益...")
        
        # Merge 行情 (T+1 收益)
        merged = pd.merge(daily_positions, price_df, on=["date", "symbol"], how="inner")
        
        # 每日持仓收益 = Sum(权重 * 次日收益)
        merged["contrib"] = merged["real_weight"] * merged["next_pct_chg"]
        daily_ret = merged.groupby("date")["contrib"].sum()
        
        # --- 扣除交易成本 ---
        # 估算逻辑：分仓模式下，每天有 1/N 的资金进行换仓 (卖旧买新)
        # 换手率 ≈ 2 * (1/N) （一买一卖）
        # 成本 = 2 * (1/N) * cost_rate
        # 这是一个近似值，比逐笔撮合快得多且足够准确
        daily_turnover = 1.0 / self.holding_period
        daily_cost = daily_turnover * 2 * self.cost_rate
        
        # 净收益
        daily_ret_net = daily_ret - daily_cost
        
        # --- 后续指标计算与绘图 (保持不变) ---
        equity_curve = (1 + daily_ret_net).cumprod()
        
        # 填充某些没有交易的日期（保持净值不变）
        equity_curve = equity_curve.reindex(all_dates).ffill().fillna(1.0)
        
        # Benchmark
        idx_code = self.config.get("preprocessing", {}).get("labels", {}).get("index_code", "000300.SH")
        idx_file = os.path.join(self.paths["data_raw"], f"index_{idx_code.replace('.', '')}.parquet")
        benchmark_curve = None
        
        if os.path.exists(idx_file):
            idx_df = read_parquet(idx_file)
            idx_df["date"] = pd.to_datetime(idx_df["date"])
            idx_df = idx_df.set_index("date").sort_index()
            # 对齐
            common = equity_curve.index.intersection(idx_df.index)
            if not common.empty:
                idx_sub = idx_df.loc[common, "close"]
                benchmark_curve = idx_sub / idx_sub.iloc[0]
        
        metrics = self._calc_metrics(daily_ret_net)
        self._plot_result(equity_curve, benchmark_curve, metrics, output_dir)
        
        # 保存结果
        res_df = pd.DataFrame({
            "strategy": equity_curve,
            "benchmark": benchmark_curve if benchmark_curve is not None else np.nan
        })
        res_df.to_csv(os.path.join(output_dir, "backtest_equity.csv"))
        
        logger.info(f"回测完成！年化: {metrics['annual_return']:.2%}, 夏普: {metrics['sharpe']:.2f}")
        return metrics

    def _calc_metrics(self, daily_ret):
        """计算指标"""
        # 剔除空值
        daily_ret = daily_ret.dropna()
        if len(daily_ret) == 0:
            return {"annual_return": 0, "sharpe": 0, "max_drawdown": 0}
            
        total_days = len(daily_ret)
        annual_ret = (1 + daily_ret.mean()) ** 252 - 1
        
        rf = 0.02
        vol = daily_ret.std() * np.sqrt(252)
        sharpe = (annual_ret - rf) / (vol + 1e-9)
        
        equity = (1 + daily_ret).cumprod()
        drawdown = equity / equity.cummax() - 1
        max_dd = drawdown.min()
        
        return {
            "annual_return": annual_ret,
            "sharpe": sharpe,
            "max_drawdown": max_dd
        }

    def _plot_result(self, equity, benchmark, metrics, out_dir):
        # 创建一个 2行1列 的图表
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # --- 子图 1: 线性坐标 (原图) ---
        ax1 = axes[0]
        equity.plot(ax=ax1, label="Strategy", color="red", linewidth=2)
        if benchmark is not None:
            benchmark.plot(ax=ax1, label="Benchmark", color="gray", linestyle="--", alpha=0.7)
        ax1.set_title(f"Backtest (Linear Scale)\nAnn Ret: {metrics['annual_return']:.1%} | Sharpe: {metrics['sharpe']:.2f}")
        ax1.legend(loc="upper left")
        ax1.grid(True, linestyle="--", alpha=0.5)

        # --- 子图 2: 对数坐标 (Log Scale) ---
        # 它可以揭示早期的波动
        ax2 = axes[1]
        equity.plot(ax=ax2, label="Strategy", color="red", linewidth=2)
        if benchmark is not None:
            benchmark.plot(ax=ax2, label="Benchmark", color="gray", linestyle="--", alpha=0.7)
        
        ax2.set_yscale('log') # <--- 关键：开启对数坐标
        ax2.set_title("Backtest (Log Scale) - Check Early Years")
        ax2.legend(loc="upper left")
        ax2.grid(True, linestyle="--", alpha=0.5, which='both') # 显示更密的网格
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "equity_curve.png"))
        plt.close()