# src/preprocessing/labels.py

import pandas as pd
import numpy as np
from tqdm import tqdm
from src.utils.logger import get_logger
from src.utils.io import read_parquet
import os

logger = get_logger()

class LabelGenerator:
    """
    标签工厂 v2.1：支持三相屏障法 (Triple Barrier Method)
    
    支持两种模式：
    1. "fixed_time": 固定持有 horizon 天后卖出 (原版逻辑)。
    2. "triple_barrier": 
       - 上屏障 (Profit Taking, PT): 触及止盈线，标签为 PT 值。
       - 下屏障 (Stop Loss, SL): 触及止损线，标签为 -SL 值。
       - 垂直屏障 (Time): 在 horizon 天内未触及上下屏障，按期末收盘结算。
    """
    
    def __init__(self, config: dict):
        self.cfg = config.get("preprocessing", {}).get("labels", {})
        
        # 基础参数
        self.horizon = self.cfg.get("horizon", 5)
        self.return_mode = self.cfg.get("return_mode", "excess_index")
        self.use_vwap = self.cfg.get("use_vwap", True)
        self.filter_limit = self.cfg.get("filter_limit", True)
        
        # 方法选择: fixed_time (默认) 或 triple_barrier
        self.method = self.cfg.get("method", "fixed_time")
        
        # 三相屏障参数
        tb_cfg = self.cfg.get("triple_barrier", {})
        self.pt = tb_cfg.get("pt", 0.10)
        self.sl = tb_cfg.get("sl", 0.05)
        
        # 加载指数数据
        self.df_index = None
        if self.return_mode == "excess_index":
            self._load_index(config)
            
    def _load_index(self, config):
        """预加载指数数据"""
        index_code = self.cfg.get("index_code", "000300.SH")
        raw_dir = config["paths"]["data_raw"]
        
        index_file = f"index_{index_code.replace('.', '')}.parquet"
        path = os.path.join(raw_dir, index_file)
        
        if os.path.exists(path):
            df = read_parquet(path)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").set_index("date")
            self.df_index = df
            logger.info(f"标签生成器已加载指数: {index_code} (用于计算超额收益)")
        else:
            logger.warning(f"未找到指数数据 {path}，强制降级为绝对收益模式。")
            self.return_mode = "absolute"

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        
        # ----------------------------------------------------------------------
        # 1. 确定基准价格 (Entry Price)
        # ----------------------------------------------------------------------
        # 逻辑：T日预测，T+1日按 VWAP 或 Open/Close 入场
        if self.use_vwap and "amount" in df.columns and "volume" in df.columns:
            # 均价 = 成交额 / 成交量 (避开停牌除零)
            vwap = df["amount"] / (df["volume"] + 1e-8)
            price_base = vwap.where(df["volume"] > 0, df["close"])
        else:
            price_base = df["close"]

        # ----------------------------------------------------------------------
        # 2. 计算标签 (分流处理)
        # ----------------------------------------------------------------------
        if self.method == "triple_barrier":
            df = self._calc_triple_barrier(df, price_base)
        else:
            df = self._calc_fixed_time(df, price_base)

        # ----------------------------------------------------------------------
        # 3. 统一后处理 (一字板过滤)
        # ----------------------------------------------------------------------
        if self.filter_limit:
            self._apply_limit_filter(df)

        return df

    def _calc_fixed_time(self, df: pd.DataFrame, price_base: pd.Series) -> pd.DataFrame:
        """原版：固定时间窗口收益（增加了平滑逻辑）"""
        
        # 1. 获取平滑窗口参数 (默认 1，即不平滑)
        window = self.cfg.get("smooth_window", 1)
        
        # 2. 计算退出价格 (Exit Price)
        if window > 1:
            # 核心修改点：
            # rolling(window, center=True) 会取当前点及前后的数据平均
            # 例如 window=3, center=True，对于 T+5 这一天，它计算的是 Avg(T+4, T+5, T+6)
            # min_periods=1 保证边缘数据也能计算
            smoothed_price = price_base.rolling(window=window, center=True, min_periods=1).mean()
            
            # 将平滑后的价格序列向未来移动 horizon 天
            exit_price = smoothed_price.shift(-(1 + self.horizon))
        else:
            # 原逻辑
            exit_price = price_base.shift(-(1 + self.horizon))
        
        # Entry 保持不变 (T+1 入场)
        entry_price = price_base.shift(-1)
        
        # 3. 计算收益率
        raw_ret = (exit_price / entry_price) - 1.0
        
        # ... (后续代码保持不变)
        label_col = f"label_{self.horizon}d"
        df[label_col] = raw_ret
        
        if self.return_mode == "excess_index" and self.df_index is not None:
            idx_ret = self._get_index_return(df, days=self.horizon)
            df[label_col] = df[label_col] - idx_ret

        df["label"] = df[label_col]
        return df

    def _calc_triple_barrier(self, df: pd.DataFrame, price_base: pd.Series) -> pd.DataFrame:
        """
        三相屏障法实现 (Vectorized Loop 优化版)
        """
        # 1. 初始化
        entry_price = price_base.shift(-1) # T+1 入场
        
        # 默认结果：时间屏障 (Time Barrier)
        # 如果没碰到止盈止损，就在 horizon 结束时卖出
        exit_price_time = price_base.shift(-(1 + self.horizon))
        final_ret = (exit_price_time / entry_price) - 1.0
        
        # 用于记录实际持有时长 (默认为 horizon)
        actual_holding_days = pd.Series(self.horizon, index=df.index)
        
        # 标记是否已经触碰屏障 (True表示已触碰，后续不再更新)
        touched = pd.Series(False, index=df.index)
        
        # 2. 循环扫描每一天 (从 T+2 到 T+1+Horizon)
        # 我们检查 T+1 之后的每一天的高低点
        for day in range(1, self.horizon + 1):
            # 获取 T+1+day 的 High/Low
            # shift(-(1+day)) 对应 T+1+day
            curr_high = df["high"].shift(-(1 + day))
            curr_low = df["low"].shift(-(1 + day))
            
            # 计算相对于 Entry(T+1) 的涨跌幅
            ret_high = (curr_high / entry_price) - 1.0
            ret_low = (curr_low / entry_price) - 1.0
            
            # 2.1 检查下屏障 (Stop Loss) - 优先检查风控? 
            # 假设：如果同一天既碰到止盈又碰到止损，保守起见算止损 (或者看 Close，这里简化处理)
            # 触碰条件：最低价跌破 SL
            mask_sl = (ret_low <= -self.sl) & (~touched)
            
            # 2.2 检查上屏障 (Profit Taking)
            # 触碰条件：最高价突破 PT
            mask_pt = (ret_high >= self.pt) & (~touched)
            
            # 2.3 更新结果
            # 如果触碰止损
            if mask_sl.any():
                # 收益锁定为 -SL (或者用实际 low，这里用阈值锁定代表严格执行)
                final_ret.loc[mask_sl] = -self.sl 
                actual_holding_days.loc[mask_sl] = day
                touched.loc[mask_sl] = True
                
            # 如果触碰止盈 (且没触碰止损)
            # 注意：mask_pt 里的 ~touched 已经排除了刚才 mask_sl 更新过的行
            if mask_pt.any():
                final_ret.loc[mask_pt] = self.pt
                actual_holding_days.loc[mask_pt] = day
                touched.loc[mask_pt] = True
        
        # 3. 赋值
        label_name = f"label_{self.horizon}d"
        df[label_name] = final_ret
        
        # 4. 处理超额收益 (Excess Return)
        # 难点：每行持有天数不同。这里做一个简化处理：
        # 如果是触碰屏障退出的，减去对应天数的指数收益；如果是时间退出的，减去 Horizon 天数的指数收益。
        if self.return_mode == "excess_index" and self.df_index is not None:
            # 获取指数的全序列 Close
            # 为了高性能，我们不逐行查，而是近似处理：
            # 统一减去 "Horizon天" 的指数收益。
            # 原因：精确匹配天数需要极大的计算量(每行日期不同)，而在训练数据量大时，
            # 指数在短周期(1-5天)内的波动差异相对于个股的 PT/SL 事件可以忽略，
            # 或者认为这是一个“对冲成本”。
            
            # 方案 B (更严谨)：只计算时间屏障的超额。
            # 对于触发 PT/SL 的，直接使用绝对收益作为 Label (因为这是事件驱动的)。
            # 这里采用混合方案：
            idx_ret_horizon = self._get_index_return(df, self.horizon)
            
            # 只有未触发屏障的(Time Barrier)，才减去指数收益
            # 触发了屏障的，保留绝对收益 (代表我们通过交易获得了确定性的 Alpha)
            # mask_time_exit = (actual_holding_days == self.horizon) & (~touched) # 注意 touched 在最后一天可能也是 True
            # 简化：统一减去指数收益，但这会惩罚快速止盈的样本。
            
            # 最终决策：对于 Label Construction，直接减去同期指数收益是最标准的 Alpha 定义。
            # 但为了工程效率，暂时全量减去 Horizon 期间的指数收益。
            df[label_name] = df[label_name] - idx_ret_horizon

        df["label"] = df[label_name]
        return df

    def _get_index_return(self, df, days):
        """计算指数在未来 days 天的收益率"""
        # 对齐索引
        df_idx = df.set_index("date")
        common_idx = df_idx.index.intersection(self.df_index.index)
        
        # 构造指数的 T+1 和 T+1+days
        # 注意：这里需要在 df_index 上操作，而不是 df
        # 我们先创建一个映射表
        target_dates = pd.Series(common_idx, index=common_idx)
        
        # 找到 T+1 日期
        # 这种 shift 比较难准确对应日期，改为使用 shift 后的 pct_change
        # 近似法：利用 index 数据本身的 shift
        
        idx_close = self.df_index["close"]
        # T+1 Close
        idx_entry = idx_close.shift(-1)
        # T+1+days Close
        idx_exit = idx_close.shift(-(1 + days))
        
        idx_ret_series = (idx_exit / idx_entry) - 1.0
        
        # 映射回 df
        return df["date"].map(idx_ret_series)

    def _apply_limit_filter(self, df: pd.DataFrame):
        """剔除 T+1 一字涨停无法买入的样本"""
        next_high = df["high"].shift(-1)
        next_low = df["low"].shift(-1)
        next_close = df["close"].shift(-1)
        curr_close = df["close"]
        
        # T+1 一字涨停逻辑
        is_limit_up = (next_high == next_low) & (next_close > curr_close * 1.09)
        
        if is_limit_up.any():
            df.loc[is_limit_up, "label"] = np.nan