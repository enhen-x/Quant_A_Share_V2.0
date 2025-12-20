# src/preprocessing/labels.py

import pandas as pd
import numpy as np
from tqdm import tqdm
from src.utils.logger import get_logger
from src.utils.io import read_parquet
import os

logger = get_logger()

class LabelGenerator:
    def __init__(self, config: dict):
        self.cfg = config.get("preprocessing", {}).get("labels", {})
        
        # 基础参数
        self.horizon = self.cfg.get("horizon", 5)
        self.return_mode = self.cfg.get("return_mode", "excess_index")
        self.use_vwap = self.cfg.get("use_vwap", True)
        self.filter_limit = self.cfg.get("filter_limit", True)
        
        # [新增] Winsorize 配置
        self.enable_winsorize = self.cfg.get("enable_winsorize", False)
        self.winsorize_limits = self.cfg.get("winsorize_limits", [0.01, 0.99])
        
        # 方法选择: fixed_time (默认) 或 triple_barrier
        self.method = self.cfg.get("method", "fixed_time")
        
        # 三相屏障参数
        tb_cfg = self.cfg.get("triple_barrier", {})
        self.pt = tb_cfg.get("pt", 0.10)
        self.sl = tb_cfg.get("sl", 0.05)
        
        # [新增] 双头模型配置
        self.dual_head_cfg = config.get("model", {}).get("dual_head", {})
        self.dual_head_enabled = self.dual_head_cfg.get("enable", False)
        
        # 加载指数数据
        self.df_index = None
        if self.return_mode == "excess_index" or self._need_index_for_dual_head():
            self._load_index(config)
    
    def _need_index_for_dual_head(self) -> bool:
        """检查双头模型是否需要指数数据"""
        if not self.dual_head_enabled:
            return False
        cls_cfg = self.dual_head_cfg.get("classification", {})
        return cls_cfg.get("label_mode", "absolute") == "excess_index"

    # ... (原有 _load_index 保持不变) ...
    def _load_index(self, config):
        index_code = self.cfg.get("index_code", "000300.SH")
        raw_dir = config["paths"]["data_raw"]
        index_file = f"index_{index_code.replace('.', '')}.parquet"
        path = os.path.join(raw_dir, index_file)
        if os.path.exists(path):
            df = read_parquet(path)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").set_index("date")
            self.df_index = df
            logger.info(f"标签生成器已加载指数: {index_code}")
        else:
            logger.warning(f"未找到指数数据 {path}，降级为绝对收益模式。")
            self.return_mode = "absolute"

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        
        # 1. 准备价格曲线 (VWAP)
        if "amount" in df.columns and "volume" in df.columns:
            vwap = df["amount"] / (df["volume"] + 1e-8)
            price_vwap = vwap.where(df["volume"] > 0, df["close"])
        else:
            price_vwap = df["close"]

        price_for_exit = price_vwap
        price_for_entry = price_vwap

        # 2. 计算标签
        if self.method == "triple_barrier":
            df = self._calc_triple_barrier(df, price_for_entry) 
        else:
            df = self._calc_fixed_time(df, price_for_entry, price_for_exit)

        # 3. 统一后处理
        if self.filter_limit:
            self._apply_limit_filter(df)
            
        # [新增] 4. 去极值 (Winsorization)
        # 这一步是为了清洗类似 461% 这样的数据错误或极端新股行情
        if self.enable_winsorize:
            self._apply_winsorization(df)
        
        # [新增] 5. 双头模型分类标签生成
        if self.dual_head_enabled:
            cls_cfg = self.dual_head_cfg.get("classification", {})
            if cls_cfg.get("enable", True):
                df = self._generate_classification_label(df, price_for_entry, price_for_exit)

        return df
    
    def _generate_classification_label(self, df: pd.DataFrame, price_entry, price_exit) -> pd.DataFrame:
        """
        [新增] 生成二分类标签 (0=跌, 1=涨)
        
        Args:
            df: 数据框
            price_entry: 入场价格序列
            price_exit: 出场价格序列
        """
        cls_cfg = self.dual_head_cfg.get("classification", {})
        label_mode = cls_cfg.get("label_mode", "absolute")
        threshold = cls_cfg.get("threshold", 0.0)
        
        if label_mode == "absolute":
            # 使用绝对收益
            entry_price = price_entry.shift(-1)
            exit_price = price_exit.shift(-(1 + self.horizon))
            raw_ret = (exit_price / entry_price) - 1.0
            logger.info(f"分类标签使用绝对涨幅模式, 阈值: {threshold:.4f}")
        else:
            # 使用超额收益 (已计算的 label 列)
            raw_ret = df["label"]
            logger.info(f"分类标签使用超额涨幅模式, 阈值: {threshold:.4f}")
        
        # 生成 0/1 标签
        df["label_cls"] = (raw_ret > threshold).astype(int)
        
        # 统计分布
        pos_count = (df["label_cls"] == 1).sum()
        neg_count = (df["label_cls"] == 0).sum()
        total = pos_count + neg_count
        if total > 0:
            logger.info(f"分类标签分布: 正样本(涨)={pos_count:,} ({pos_count/total:.1%}), "
                       f"负样本(跌)={neg_count:,} ({neg_count/total:.1%})")
        
        return df

    # ... (原有 _calc_fixed_time, _calc_triple_barrier, _get_index_return, _apply_limit_filter 保持不变) ...
    def _calc_fixed_time(self, df, price_entry_series, price_exit_series):
        window = self.cfg.get("smooth_window", 1)
        if window > 1:
            smoothed_price = price_exit_series.rolling(window=window, center=True, min_periods=1).mean()
            exit_price = smoothed_price.shift(-(1 + self.horizon))
        else:
            exit_price = price_exit_series.shift(-(1 + self.horizon))
        
        entry_price = price_entry_series.shift(-1)
        raw_ret = (exit_price / entry_price) - 1.0
        
        label_col = f"label_{self.horizon}d"
        df[label_col] = raw_ret
        
        if self.return_mode == "excess_index" and self.df_index is not None:
            idx_ret = self._get_index_return(df, days=self.horizon)
            df[label_col] = df[label_col] - idx_ret

        df["label"] = df[label_col]
        return df

    def _calc_triple_barrier(self, df, price_base):
        entry_price = price_base.shift(-1)
        exit_price_time = price_base.shift(-(1 + self.horizon))
        final_ret = (exit_price_time / entry_price) - 1.0
        actual_holding_days = pd.Series(self.horizon, index=df.index)
        touched = pd.Series(False, index=df.index)
        
        for day in range(1, self.horizon + 1):
            curr_high = df["high"].shift(-(1 + day))
            curr_low = df["low"].shift(-(1 + day))
            ret_high = (curr_high / entry_price) - 1.0
            ret_low = (curr_low / entry_price) - 1.0
            
            mask_sl = (ret_low <= -self.sl) & (~touched)
            mask_pt = (ret_high >= self.pt) & (~touched)
            
            if mask_sl.any():
                final_ret.loc[mask_sl] = -self.sl 
                actual_holding_days.loc[mask_sl] = day
                touched.loc[mask_sl] = True
            if mask_pt.any():
                final_ret.loc[mask_pt] = self.pt
                actual_holding_days.loc[mask_pt] = day
                touched.loc[mask_pt] = True
        
        label_name = f"label_{self.horizon}d"
        df[label_name] = final_ret
        
        if self.return_mode == "excess_index" and self.df_index is not None:
            idx_ret_horizon = self._get_index_return(df, self.horizon)
            df[label_name] = df[label_name] - idx_ret_horizon

        df["label"] = df[label_name]
        return df

    def _get_index_return(self, df, days):
        df_idx = df.set_index("date")
        common_idx = df_idx.index.intersection(self.df_index.index)
        idx_close = self.df_index["close"]
        idx_entry = idx_close.shift(-1)
        idx_exit = idx_close.shift(-(1 + days))
        idx_ret_series = (idx_exit / idx_entry) - 1.0
        return df["date"].map(idx_ret_series)

    def _apply_limit_filter(self, df):
        next_high = df["high"].shift(-1)
        next_low = df["low"].shift(-1)
        next_close = df["close"].shift(-1)
        curr_close = df["close"]
        is_limit_up = (next_high == next_low) & (next_close > curr_close * 1.09)
        if is_limit_up.any():
            df.loc[is_limit_up, "label"] = np.nan

    def _apply_winsorization(self, df: pd.DataFrame):
        """
        [新增] 标签去极值
        """
        if "label" not in df.columns:
            return

        # 使用分位数进行截断 (Clip)
        lower_q = self.winsorize_limits[0]
        upper_q = self.winsorize_limits[1]
        
        # 计算全局阈值可能会受到时间影响，但在单只股票序列中计算比较危险(样本少)
        # 这里采用简单的绝对阈值截断，或者基于该股票历史分布
        # 为了高性能和简单，这里演示基于当前 batch (单只股票) 的分布
        # *注*：如果是单只股票处理，极值可能就是它自己，所以通常建议设置绝对阈值
        # 比如：4天收益率超过 50% 视为异常
        
        # 方案 A: 绝对值截断 (Hard Clip) - 推荐，稳定
        # 4天涨 40% 已经是妖股极限，超过这个通常是数据错误
        df["label"] = df["label"].clip(lower=-0.4, upper=0.4)
        
        # 方案 B: 分位数截断 (Quantile Clip) - 需小心样本太少
        # q_low = df["label"].quantile(lower_q)
        # q_high = df["label"].quantile(upper_q)
        # df["label"] = df["label"].clip(lower=q_low, upper=q_high)