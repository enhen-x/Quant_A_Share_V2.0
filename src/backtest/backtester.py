# src/backtest/backtester.py
# ============================================================================
# 向量化回测引擎 (VectorBacktester)
# ============================================================================
# 
# 【核心功能】
# 这是一个基于向量化计算的回测系统，用于评估量化选股策略的历史表现。
# 它接收模型生成的选股信号（signal_df），模拟交易过程，计算策略收益和风险指标。
#
# 【回测流程概览】
# 1. 初始化：读取配置（持仓周期、换仓模式、买入价格模式等）
# 2. 加载数据：读取股票行情数据，计算首日收益和每日收益
# 3. 信号处理：将选股信号"膨胀"为每日的持仓记录
# 4. 收益计算：匹配持仓与行情数据，计算加权收益
# 5. 绩效评估：计算年化收益、夏普比率、最大回撤等指标
# 6. 可视化：绘制净值曲线图表
#
# 【关键概念】
# - T日：信号产生日（模型预测日）
# - T+1日：实际买入日（遵循T+1交易规则）
# - 持仓周期(holding_period)：买入后持有的天数（默认5天）
# - 换仓模式：rolling（滚动换仓）或 periodic（定期换仓）
# ============================================================================

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from src.utils.logger import get_logger
from src.utils.config import GLOBAL_CONFIG
from src.utils.io import read_parquet, ensure_dir

# Matplotlib 字体配置（解决中文和减号显示问题）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 使用 ASCII 减号代替 Unicode 减号

logger = get_logger()


class VectorBacktester:
    """
    向量化回测引擎
    
    该类实现了一个高效的向量化回测框架，避免了传统事件驱动回测的低效循环。
    通过 pandas 的向量运算批量处理所有交易信号和持仓记录，大幅提升回测速度。
    
    主要特性:
    - 支持多种买入价格模式（开盘价/均价/收盘价）
    - 支持滚动换仓和定期换仓两种模式
    - 自动处理涨停无法买入的情况
    - 计算完整的绩效指标（年化收益、夏普、最大回撤）
    - 生成净值曲线对比图（策略 vs 基准指数）
    """
    
    def __init__(self):
        """
        初始化回测引擎
        
        从全局配置中读取：
        - paths: 数据文件路径配置
        - holding_period: 持仓周期（从 preprocessing.labels.horizon 读取，默认5天）
        - rebalance_mode: 换仓模式 ("rolling" 或 "periodic")
        - entry_price: 买入价格模式 ("open"/"vwap"/"close"/"t_close")
        """
        self.config = GLOBAL_CONFIG
        self.paths = self.config["paths"]
        
        strategy_cfg = self.config.get("strategy", {})
        # 持仓周期：买入后持有多少天，与标签计算的 horizon 保持一致
        self.holding_period = self.config.get("preprocessing", {}).get("labels", {}).get("horizon", 5)
        # 换仓模式：
        # - "rolling": 滚动换仓，每天都可以有新的买入信号，仓位分层叠加
        # - "periodic": 定期换仓，每 holding_period 天才换仓一次
        self.rebalance_mode = strategy_cfg.get("rebalance_mode", "rolling")
        
        # === 买入价格模式 ===
        # 模拟不同的买入时机，影响首日收益的计算方式
        # "t_close": T日收盘价买入（理论值，实际无法做到）
        # "open": T+1 开盘价买入（最激进）
        # "vwap": T+1 均价买入 (Volume Weighted Average Price，适中)
        # "close": T+1 收盘价买入（最保守）
        self.entry_price = strategy_cfg.get("entry_price", "close")
        logger.info(f"回测买入模式: {self.entry_price}")
        
        # 默认交易成本（双边）：0.2% = 2‰
        # 包含：印花税(卖出0.1%) + 佣金(双边约0.06%) + 滑点预估
        # 可被 run() 方法的 cost_rate 参数覆盖
        self.default_cost_rate = 0.002
        
    def load_data(self):
        """
        加载股票行情数据并预计算收益率
        
        【数据来源】
        从 data_processed/all_stocks.parquet 读取已处理的股票日线数据
        
        【计算内容】
        1. VWAP 均价：(开 + 高 + 低 + 收) / 4
        2. entry_price 买入价：根据配置的买入模式确定
        3. mark_price 盯市价格：与买入模式一致的每日价格（用于后续日收益）
        4. first_day_ret 首日收益：从买入价到当日盯市价的收益率
        5. daily_ret 每日收益：基于盯市价格的日收益率
        6. next_open_ret 次日开盘涨幅：用于判断涨停无法买入
        
        【收益率计算逻辑 - 保持一致性】
        假设 T 日产生信号，T+1 日买入：
        - 买入模式为 vwap 时：
          - 首日收益 = (T+1 vwap / 买入价 vwap) - 1 = 0（同一天的 vwap）
          - 后续收益 = (当日 vwap / 前日 vwap) - 1
        - 买入模式为 open 时：
          - 首日收益 = (T+1 收盘 / T+1 开盘) - 1
          - 后续收益 = (当日收盘 / 前日收盘) - 1
        - 买入模式为 close 时：
          - 首日收益 = 0（T+1收盘买入，当天无收益）
          - 后续收益 = (当日收盘 / 前日收盘) - 1
        
        Returns:
            tuple: (price_df, all_dates)
                - price_df: 包含收益率信息的DataFrame
                - all_dates: 排序后的所有交易日期列表
        """
        path = os.path.join(self.paths["data_processed"], "all_stocks.parquet")
        df = read_parquet(path)
        
        # 按股票代码和日期排序，确保 shift 操作正确
        df = df.sort_values(["symbol", "date"])
        
        # === 计算 VWAP (简化版均价) ===
        # 注意：这不是真正的成交量加权均价，而是简单的 OHLC 均值
        df["vwap"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
        
        # === 根据 entry_price 模式确定买入价和盯市价格 ===
        # 为了保持一致性：
        # - entry_price: T+1 日的买入价格
        # - mark_price: 每日的盯市价格（与买入模式一致）
        # - 后续日收益基于 mark_price 计算
        
        if self.entry_price == "t_close":
            # T日收盘买入（理论最优，但实际晚盘竞价难以实现）
            df["entry_price"] = df["close"]
            df["mark_price"] = df["close"]  # 盯市用收盘价
        elif self.entry_price == "open":
            # T+1开盘价买入（集合竞价或开盘后立即买入）
            df["entry_price"] = df.groupby("symbol")["open"].shift(-1)
            df["mark_price"] = df["close"]  # 开盘买入后，用收盘价盯市更合理
        elif self.entry_price == "vwap":
            # T+1均价买入（分批买入的近似）
            # 【关键修复】后续日也用 vwap-to-vwap 计算收益
            df["entry_price"] = df.groupby("symbol")["vwap"].shift(-1)
            df["mark_price"] = df["vwap"]  # 用均价盯市
        else:  # close（默认）
            # T+1收盘价买入（尾盘买入，最保守假设）
            df["entry_price"] = df.groupby("symbol")["close"].shift(-1)
            df["mark_price"] = df["close"]  # 盯市用收盘价
        
        # === 计算每日收益用于真实净值曲线 ===
        # 
        # 【关键理解】为了反映真实的每日波动和回撤，需要使用真实的每日收益：
        # 1. first_day_ret: 从买入价到首日收盘的收益（T+1 日）
        # 2. daily_ret: 后续每日的收益（mark_price 的逐日变化）
        #
        # 这样净值曲线会反映真实的日内/日间波动，不会抹平回撤
        
        # 首日收益：从买入价到 T+1 日的收盘价
        # 注意：entry_price 是 T+1 的买入价，next_mark_price 是 T+1 的盯市价格
        # 如果买入模式是 vwap，首日收益 = (T+1 vwap / T+1 vwap) - 1 = 0（买卖同价）
        # 如果买入模式是 open，首日收益 = (T+1 close / T+1 open) - 1
        df["next_mark_price"] = df.groupby("symbol")["mark_price"].shift(-1)
        df["first_day_ret"] = (df["next_mark_price"] / df["entry_price"]) - 1.0
        
        # 后续日收益：mark_price 的逐日变化
        df["daily_ret"] = df.groupby("symbol")["mark_price"].pct_change()
        
        # === 持有期收益（用于诊断）===
        df["exit_price"] = df.groupby("symbol")["mark_price"].shift(-(self.holding_period + 1))
        df["holding_ret"] = (df["exit_price"] / df["entry_price"]) - 1.0
        
        # === 涨停判断 ===
        df["next_open"] = df.groupby("symbol")["open"].shift(-1)
        df["next_open_ret"] = (df["next_open"] / df["close"]) - 1.0
        
        # 提取所有交易日期
        all_dates = sorted(df["date"].unique())
        
        # 返回包含每日收益的 DataFrame
        return df[["date", "symbol", "first_day_ret", "daily_ret", "holding_ret", "next_open_ret"]].dropna(subset=["date", "symbol"]), all_dates

    def run(self, signal_df: pd.DataFrame, output_dir: str, 
            cost_rate: float = None, 
            start_date: str = None, 
            end_date: str = None):
        """
        执行回测主流程
        
        【参数说明】
        :param signal_df: 选股信号 DataFrame，包含以下列：
                         - date: 信号产生日期（T日）
                         - symbol: 股票代码
                         - weight: 持仓权重（通常基于预测得分归一化）
        :param output_dir: 输出目录，用于保存回测结果图表
        :param cost_rate: 交易成本率，如不指定则使用默认值 0.002
        :param start_date: 强制回测开始日期（用于压力测试特定时段）
        :param end_date: 强制回测结束日期
        
        【返回值】
        dict: 包含以下字段的绩效指标字典：
             - annual_return: 年化收益率
             - sharpe: 夏普比率
             - max_drawdown: 最大回撤
             - equity_curve: 净值曲线（pd.Series）
        
        【执行流程】
        1. 时间对齐：将信号日期与行情日期对齐
        2. 模式选择：根据换仓模式确定有效信号和权重因子
        3. 信号膨胀：将每个信号扩展为持仓周期内的每日记录
        4. 收益计算：匹配行情数据计算每日加权收益
        5. 成本扣减：扣除交易成本后计算净收益
        6. 复利累积：生成净值曲线
        7. 绩效评估：计算各项风险指标
        """
        # ====================================================================
        # 步骤 0：参数初始化
        # ====================================================================
        # 确定本次运行的交易成本
        current_cost = cost_rate if cost_rate is not None else self.default_cost_rate
        
        # 创建输出目录
        ensure_dir(output_dir)
        # 加载预处理的行情数据（包含首日收益和每日收益）
        price_df, all_dates = self.load_data()
        
        # ====================================================================
        # 步骤 1：时间对齐与截断
        # ====================================================================
        # 只保留在行情数据中存在的交易日的信号
        # （防止信号中出现非交易日或数据缺失的日期）
        valid_signal_df = signal_df[signal_df["date"].isin(all_dates)].copy()
        if valid_signal_df.empty:
            return {}
            
        # 确定实际回测开始时间
        # 如果用户没有指定，则使用信号数据中最早的日期
        actual_start = start_date if start_date else valid_signal_df["date"].min()
        actual_start = pd.to_datetime(actual_start)
        
        # 确定实际回测结束时间，并过滤日期范围
        if end_date:
            actual_end = pd.to_datetime(end_date)
            all_dates = sorted([d for d in all_dates if actual_start <= d <= actual_end])
            price_df = price_df[(price_df["date"] >= actual_start) & (price_df["date"] <= actual_end)]
        else:
            all_dates = sorted([d for d in all_dates if d >= actual_start])
            price_df = price_df[price_df["date"] >= actual_start]
        
        # 创建日期索引映射（用于快速查找日期位置）
        date_to_idx = {d: i for i, d in enumerate(all_dates)}  # 日期 -> 索引
        idx_to_date = {i: d for i, d in enumerate(all_dates)}  # 索引 -> 日期
        
        # ====================================================================
        # 步骤 2：模式选择 - 确定有效信号和权重因子
        # ====================================================================
        target_signals = pd.DataFrame()
        weight_factor = 1.0
        
        if self.rebalance_mode == "periodic":
            # 【定期换仓模式】
            # 每隔 holding_period 天才执行一次换仓
            # 例如：holding_period=5，则只在第 0、5、10、15... 天换仓
            # 同一时刻只持有一批股票，仓位不重叠
            rebalance_indices = range(0, len(all_dates), self.holding_period)
            rebalance_dates = [idx_to_date[i] for i in rebalance_indices]
            target_signals = valid_signal_df[valid_signal_df["date"].isin(rebalance_dates)]
            weight_factor = 1.0  # 全仓买入
        else:
            # 【滚动换仓模式】（默认）
            # 每天都可以有新的买入信号
            # 仓位分层叠加：同时持有最多 holding_period 批股票
            # 例如：holding_period=5，则每批股票占 1/5 仓位
            target_signals = valid_signal_df
            weight_factor = 1.0 / self.holding_period  # 每批占 1/5 仓位

        # ====================================================================
        # 步骤 3：匹配信号与持有期收益
        # ====================================================================
        # 步骤 3：信号膨胀 - 将信号扩展为每日持仓记录
        # ====================================================================
        # 【核心逻辑】
        # 每个 T 日的信号会生成 T+1 到 T+holding_period 天的持仓记录
        # 例如：T日信号 -> [T+1, T+2, T+3, T+4, T+5] 共5天持仓
        # 
        # 关键字段：
        # - is_first_day: 标记是否为买入日（T+1），用于选择 first_day_ret
        # - signal_date: 原始信号日期（T日），用于匹配首日收益
        # - date: 实际持仓日期（T+1 到 T+N）
        # - real_weight: 实际仓位权重 = 原始权重 × weight_factor
        
        target_signals = target_signals[target_signals["date"] >= actual_start]
        if end_date:
            target_signals = target_signals[target_signals["date"] <= actual_end]

        expanded_signals = []
        
        for signal_date, group in target_signals.groupby("date"):
            if signal_date not in date_to_idx:
                continue
            start_idx = date_to_idx[signal_date]
            
            # 计算持仓日期：从 T+1 开始，持有 holding_period 天
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
        
        # ====================================================================
        # 步骤 4：计算每日收益 - 匹配真实的首日/后续日收益
        # ====================================================================
        # - 首日（T+1）：使用 first_day_ret（从买入价到收盘的收益）
        # - 后续日（T+2 ~ T+N）：使用 daily_ret（mark_price 的逐日变化）
        
        first_day_holdings = full_holdings[full_holdings["is_first_day"]].copy()
        later_day_holdings = full_holdings[~full_holdings["is_first_day"]].copy()
        
        # 4.1 计算首日持仓收益
        if len(first_day_holdings) > 0:
            first_merged = pd.merge(
                first_day_holdings,
                price_df[["date", "symbol", "first_day_ret", "next_open_ret"]].rename(columns={"date": "signal_date"}),
                on=["signal_date", "symbol"],
                how="left"
            )
            # 涨停处理
            limit_up_mask = first_merged["next_open_ret"] > 0.095
            first_merged["ret_to_use"] = first_merged["first_day_ret"].fillna(0.0)
            first_merged.loc[limit_up_mask, "ret_to_use"] = 0.0
            first_merged["contrib"] = first_merged["real_weight"] * first_merged["ret_to_use"]
        else:
            first_merged = pd.DataFrame()
        
        # 4.2 计算后续日持仓收益
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
        
        # 4.3 合并所有持仓收益
        if len(first_merged) > 0 and len(later_merged) > 0:
            merged_all = pd.concat([first_merged, later_merged], ignore_index=True)
        elif len(first_merged) > 0:
            merged_all = first_merged
        elif len(later_merged) > 0:
            merged_all = later_merged
        else:
            return {"annual_return": 0, "sharpe": 0, "max_drawdown": 0}
        
        # 4.4 按日期聚合，计算组合每日总收益
        daily_ret = merged_all.groupby("date")["contrib"].sum()
        
        # ====================================================================
        # 步骤 5：扣除交易成本
        # ====================================================================
        # 【成本计算】
        # - cost_rate 是双边成本（单次买入+卖出的总成本）
        # - 每天换 1/holding_period 的仓位
        # - 所以每天的成本 = (1/holding_period) × cost_rate
        daily_cost = (1.0 / self.holding_period) * current_cost
        daily_ret_net = daily_ret - daily_cost
        
        # ====================================================================
        # 步骤 6：复利累积生成净值曲线
        # ====================================================================
        equity_curve = (1 + daily_ret_net).cumprod()
        equity_curve.index = pd.to_datetime(equity_curve.index)
        
        # 计算绩效指标
        metrics = self._calc_metrics(daily_ret_net)
        
        # 诊断输出
        logger.info("=" * 60)
        logger.info(f"【收益诊断分析 - {self.rebalance_mode} 模式】")
        logger.info("=" * 60)
        logger.info(f"持仓周期: {self.holding_period} 天")
        logger.info(f"交易日数: {len(daily_ret)}")
        logger.info(f"日收益 - 均值: {daily_ret.mean():.6f} ({daily_ret.mean()*100:.4f}%)")
        logger.info(f"日收益 - 标准差: {daily_ret.std():.6f}")
        
        # 每天的持仓批次数
        batch_per_day = merged_all.groupby("date").apply(lambda x: x["signal_date"].nunique(), include_groups=False)
        logger.info(f"每日持仓批次数 - 均值: {batch_per_day.mean():.2f}")
        logger.info("=" * 60)
        # ====================================================================
        # 步骤 6：加载基准指数
        # ====================================================================
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
        
        # ====================================================================
        # 步骤 7：绘制结果和输出
        # ====================================================================
        self._plot_result(equity_curve, benchmark_curve, metrics, output_dir, 
                         title_suffix=f"(Cost={current_cost*1000:.1f}‰)")
        
        logger.info(f"年化收益率: {metrics['annual_return']:.2%}")
        logger.info(f"夏普比率: {metrics['sharpe']:.2f}")
        logger.info(f"最大回撤: {metrics['max_drawdown']:.2%}")
        
        metrics["equity_curve"] = equity_curve
        return metrics

    def _calc_metrics_period(self, period_ret, holding_period):
        """
        基于周期收益计算绩效指标（复利）
        
        【适用场景】
        当使用定期换仓模式（periodic）时，收益以周期为单位汇总。
        例如：每5天换仓一次，则 period_ret 中每个值代表5天的收益。
        
        【参数说明】
        :param period_ret: 周期收益序列（pd.Series）
        :param holding_period: 持仓周期天数（用于计算年化因子）
        
        【返回值】
        dict: 包含 annual_return, sharpe, max_drawdown 的字典
        
        【计算逻辑】
        1. 年化收益：使用复利公式，一年约有 252/holding_period 个周期
        2. 年化波动率：周期波动率 × sqrt(每年周期数)
        3. 夏普比率：(年化收益 - 无风险利率) / 年化波动率
        4. 最大回撤：净值曲线的历史回撤最大值
        """
        period_ret = period_ret.dropna()
        if len(period_ret) == 0:
            return {"annual_return": 0, "sharpe": 0, "max_drawdown": 0}
        
        # 复利累积：计算净值曲线
        equity = (1 + period_ret).cumprod()
        total_return = equity.iloc[-1] - 1  # 总收益率 = 期末净值 - 1
        
        # 计算实际回测年数
        n_periods = len(period_ret)
        total_trading_days = n_periods * holding_period
        n_years = total_trading_days / 252  # 一年约 252 个交易日
        
        # 年化收益率（正确的复利公式）
        # annual_ret = (1 + total_return) ^ (1 / n_years) - 1
        if n_years > 0:
            annual_ret = (1 + total_return) ** (1 / n_years) - 1
        else:
            annual_ret = 0
        
        logger.info(f"年化计算: 总周期={n_periods}, 总天数={total_trading_days}, 年数={n_years:.2f}, 总收益={total_return:.2%}")
        
        # 计算夏普比率
        # 周期收益转日收益来计算年化波动率
        periods_per_year = 252 / holding_period
        period_vol = period_ret.std()
        vol_annual = period_vol * np.sqrt(periods_per_year)
        sharpe = (annual_ret - 0.02) / (vol_annual + 1e-9)
        
        # 计算最大回撤
        drawdown = equity / equity.cummax() - 1
        max_dd = drawdown.min()
        
        return {"annual_return": annual_ret, "sharpe": sharpe, "max_drawdown": max_dd}

    def _calc_metrics(self, daily_ret):
        """
        基于日收益计算绩效指标
        
        【参数说明】
        :param daily_ret: 每日净收益序列（pd.Series），已扣除交易成本
        
        【返回值】
        dict: 包含以下字段：
             - annual_return: 年化收益率（复利）
             - sharpe: 夏普比率
             - max_drawdown: 最大回撤
        
        【计算公式】
        1. 年化收益 = (1 + 总收益) ^ (252 / 交易天数) - 1
        2. 年化波动率 = 日波动率 × sqrt(252)
        3. 夏普比率 = (年化收益 - 无风险利率) / 年化波动率
        4. 最大回撤 = min(净值 / 历史最高净值 - 1)
        """
        daily_ret = daily_ret.dropna()
        if len(daily_ret) == 0:
            return {"annual_return": 0, "sharpe": 0, "max_drawdown": 0}
        
        # === 年化收益率计算 ===
        # 步骤 1：计算累计净值曲线
        equity = (1 + daily_ret).cumprod()
        # 步骤 2：计算总收益率
        total_return = equity.iloc[-1] - 1
        
        # 步骤 3：年化（复利公式）
        # 假设一年有 252 个交易日
        n_days = len(daily_ret)
        annual_ret = (1 + total_return) ** (252 / n_days) - 1
        
        # === 夏普比率计算 ===
        # 无风险利率：假设 2%（年化）
        rf = 0.02
        # 年化波动率 = 日波动率 × sqrt(252)
        vol = daily_ret.std() * np.sqrt(252)
        # 夏普比率 = 超额收益 / 波动率
        sharpe = (annual_ret - rf) / (vol + 1e-9)
        
        # === 最大回撤计算 ===
        # 回撤 = 当前净值相对于历史最高点的跌幅
        drawdown = equity / equity.cummax() - 1
        max_dd = drawdown.min()  # 最大回撤是最小的负值
        
        return {"annual_return": annual_ret, "sharpe": sharpe, "max_drawdown": max_dd}

    def _plot_result(self, equity, benchmark, metrics, out_dir, title_suffix=""):
        """
        绘制回测结果图表
        
        【功能】
        生成包含两个子图的回测结果图：
        - 上图：线性刻度的净值曲线对比（策略 vs 基准）
        - 下图：对数刻度的净值曲线（便于观察长期复利效果）
        
        【参数说明】
        :param equity: 策略净值曲线（pd.Series）
        :param benchmark: 基准指数净值曲线（可为 None）
        :param metrics: 绩效指标字典（用于标题显示）
        :param out_dir: 输出目录路径
        :param title_suffix: 标题后缀（例如显示交易成本）
        
        【输出】
        保存 equity_curve.png 到指定目录
        """
        # ====================================================================
        # 【字体设置】在每次绘图前强制设置，以应对外部样式覆盖
        # ====================================================================
        # 设置中文字体列表，确保找到一个可用的中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
        # 强制使用 ASCII 减号 ('-') 代替 Unicode 减号 ('−')，解决负号显示警告
        plt.rcParams['axes.unicode_minus'] = False 
        # ====================================================================

        # 创建 2 行 1 列的子图布局
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # ------------------------------------------------------------------
        # 上图：线性刻度净值曲线
        # ------------------------------------------------------------------
        ax1 = axes[0]
        # 绘制策略净值曲线（红色实线）
        equity.plot(ax=ax1, label="Strategy", color="red", linewidth=2)
        # 绘制基准净值曲线（灰色虚线）
        if benchmark is not None:
            benchmark.plot(ax=ax1, label="Benchmark", color="gray", linestyle="--", alpha=0.7)
        # 设置标题，包含关键绩效指标
        ax1.set_title(f"Backtest {title_suffix}\nAnn Ret: {metrics['annual_return']:.1%} | Sharpe: {metrics['sharpe']:.2f} | MaxDD: {metrics['max_drawdown']:.1%}")
        ax1.legend(loc="upper left")
        ax1.grid(True, linestyle="--", alpha=0.5)
        
        # ------------------------------------------------------------------
        # 下图：对数刻度净值曲线
        # ------------------------------------------------------------------
        # 对数刻度能更好地展示长期复利增长
        # 在对数坐标下，稳定的复利增长会呈现为直线
        ax2 = axes[1]
        equity.plot(ax=ax2, label="Strategy", color="red", linewidth=2)
        if benchmark is not None:
            benchmark.plot(ax=ax2, label="Benchmark", color="gray", linestyle="--", alpha=0.7)
        ax2.set_yscale('log')  # 设置 Y 轴为对数刻度
        ax2.set_title("Backtest (Log Scale)")
        ax2.legend(loc="upper left")
        ax2.grid(True, linestyle="--", alpha=0.5, which='both')  # which='both' 显示主次网格线
        
        # 调整子图间距并保存
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "equity_curve.png"))
        plt.close()  # 关闭图形释放内存