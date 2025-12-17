# src/strategy/signal.py

import pandas as pd
import numpy as np
import os
from src.utils.logger import get_logger
from src.utils.config import GLOBAL_CONFIG
from src.utils.io import read_parquet

logger = get_logger()

class TopKSignalStrategy:
    
    def __init__(self, top_k=None):
        self.conf = GLOBAL_CONFIG["strategy"]
        
        if top_k is not None:
            self.top_k = top_k
        else:
            self.top_k = self.conf.get("top_k", 5)
            
        self.min_score = self.conf.get("min_pred", 0.0)
        
        # === [新增] 读取平滑配置 ===
        # 默认为 True，保持原有行为
        self.enable_smoothing = self.conf.get("enable_score_smoothing", True)
        
        self.pos_cfg = self.conf.get("position_control", {})
        
        filter_cfg = GLOBAL_CONFIG.get("preprocessing", {}).get("filter", {})
        self.min_price = filter_cfg.get("min_price", 0.0)      
        self.max_price = filter_cfg.get("max_price", 99999.0)  
        self.min_amount = filter_cfg.get("min_turnover", 0) 
        self.min_turnover_rate = filter_cfg.get("min_turnover_rate", 0.0)
        self.min_mcap = filter_cfg.get("min_mcap", 0)
        self.max_mcap = filter_cfg.get("max_mcap", float("inf"))
        self.exclude_st = filter_cfg.get("exclude_st", True)
        self.paths = GLOBAL_CONFIG["paths"]

    # ... (_load_filter_data, _calc_position_ratio 保持不变) ...
    def _load_filter_data(self):
        """加载辅助数据: amount, turnover, close, pct_chg"""
        stock_path = os.path.join(self.paths["data_processed"], "all_stocks.parquet")
        if not os.path.exists(stock_path):
            logger.error("缺少 all_stocks.parquet")
            return None, None
            
        df_stocks = read_parquet(stock_path)
        
        # 兼容列名
        if "turnover" not in df_stocks.columns and "turn" in df_stocks.columns:
            df_stocks = df_stocks.rename(columns={"turn": "turnover"})
            
        # 必须加载 pct_chg (涨跌幅) 用于判断今日是否涨停
        if "pct_chg" not in df_stocks.columns:
            df_stocks = df_stocks.sort_values(["symbol", "date"])
            df_stocks["pct_chg"] = df_stocks.groupby("symbol")["close"].pct_change()
            
        cols = ["date", "symbol", "turnover", "amount", "close", "pct_chg"] 
        existing_cols = [c for c in cols if c in df_stocks.columns]
        
        return df_stocks[existing_cols], None

    def _calc_position_ratio(self, dates: pd.Series) -> pd.DataFrame:
        """根据大盘指数计算每日建议仓位 (双均线策略)"""
        unique_dates = sorted(dates.unique())
        pos_df = pd.DataFrame({"date": unique_dates, "pos_ratio": 1.0})

        if not self.pos_cfg.get("enable", False):
            return pos_df

        index_code = self.pos_cfg.get("index_code", "000300.SH")
        idx_file = f"index_{index_code.replace('.', '')}.parquet"
        idx_path = os.path.join(self.paths["data_raw"], idx_file)
        
        if not os.path.exists(idx_path):
            return pos_df

        df_idx = read_parquet(idx_path)
        df_idx["date"] = pd.to_datetime(df_idx["date"])
        df_idx = df_idx.sort_values("date").set_index("date")
        
        fast_w = self.pos_cfg.get("fast_ma", 20)
        slow_w = self.pos_cfg.get("slow_ma", 60)
        ratios = self.pos_cfg.get("ratios", [1.0, 0.5, 0.0])
        
        df_idx["ma_fast"] = df_idx["close"].rolling(window=fast_w).mean()
        df_idx["ma_slow"] = df_idx["close"].rolling(window=slow_w).mean()
        
        conditions = [
            (df_idx["close"] > df_idx["ma_fast"]),
            (df_idx["close"] <= df_idx["ma_fast"]) & (df_idx["close"] > df_idx["ma_slow"]),
            (df_idx["close"] <= df_idx["ma_slow"])
        ]
        
        df_idx["target_pos"] = np.select(conditions, ratios, default=0.0)
        
        df_ratio = df_idx[["target_pos"]].reset_index().rename(columns={"target_pos": "pos_ratio"})
        pos_df = pd.merge(pos_df[["date"]], df_ratio, on="date", how="left")
        pos_df["pos_ratio"] = pos_df["pos_ratio"].ffill().fillna(1.0)
        
        return pos_df

    def generate(self, pred_df: pd.DataFrame) -> pd.DataFrame:
        initial_count = len(pred_df)
        
        # 默认使用原始分进行排序
        sort_score_col = 'pred_score'
        
        # === 根据配置决定是否进行平滑 ===
        if self.enable_smoothing:
            SMOOTH_WINDOW = 2
            # 确保数据排序
            pred_df = pred_df.sort_values(by=["symbol", "date"])
            
            # 计算平滑分
            pred_df['score_smoothed'] = pred_df.groupby('symbol')['pred_score'].transform(
                lambda x: x.rolling(window=SMOOTH_WINDOW, min_periods=1).mean()
            )
            # 切换排序字段
            sort_score_col = 'score_smoothed'
        else:
            # 如果不平滑，为了防止逻辑中断，可以把 score_smoothed 设为和 pred_score 一样
            # 或者直接让后续逻辑引用 pred_score
            pass

        df_stocks, _ = self._load_filter_data()
        
        # 合并行情
        if df_stocks is not None:
            pred_df = pd.merge(pred_df, df_stocks, on=["date", "symbol"], how="inner", suffixes=("", "_raw"))

        # ==========================
        # 执行过滤 (Filter)
        # ==========================
        
        # 1. 价格区间
        close_col = "close_raw" if "close_raw" in pred_df.columns else "close"
        if close_col in pred_df.columns:
            pred_df = pred_df[(pred_df[close_col] >= self.min_price) & (pred_df[close_col] <= self.max_price)]

        # 2. 换手率
        if "turnover" in pred_df.columns and self.min_turnover_rate > 0:
            pred_df["turnover"] = pred_df["turnover"].fillna(0)
            pred_df = pred_df[pred_df["turnover"] >= self.min_turnover_rate]
            
        # 3. 成交额
        if "amount" in pred_df.columns and self.min_amount > 0:
            pred_df["amount"] = pred_df["amount"].fillna(0)
            pred_df = pred_df[pred_df["amount"] >= self.min_amount]

        # 4. 市值过滤
        if "amount" in pred_df.columns and "turnover" in pred_df.columns:
            valid_mask = pred_df["turnover"] > 0.001
            pred_df.loc[valid_mask, "est_mcap"] = (
                pred_df.loc[valid_mask, "amount"] / (pred_df.loc[valid_mask, "turnover"] * 0.01)
            )
            pred_df["est_mcap"] = pred_df["est_mcap"].fillna(0)
            
            if self.min_mcap > 0:
                pred_df = pred_df[pred_df["est_mcap"] >= self.min_mcap]
            if self.max_mcap < float("inf"):
                pred_df = pred_df[pred_df["est_mcap"] <= self.max_mcap]

        # 5. 板块过滤
        pool_cfg = GLOBAL_CONFIG["data"]["stock_pool"]
        if not pool_cfg.get("include_kcb", False):
            pred_df = pred_df[~pred_df["symbol"].str.startswith("688")]
        if not pool_cfg.get("include_cyb", False):
            pred_df = pred_df[~pred_df["symbol"].str.startswith("300")]
        if not pool_cfg.get("include_bj", False):
            pred_df = pred_df[~pred_df["symbol"].str.match(r"^(8|4|92)")]

        # 6. 今日涨停过滤
        if "pct_chg" in pred_df.columns:
            limit_up_mask = pred_df["pct_chg"] > 0.095
            if limit_up_mask.sum() > 0:
                pred_df = pred_df[~limit_up_mask]

        # 7. 最低分数 (动态使用 sort_score_col)
        # 注意：如果不开启平滑，这里就是用 pred_score 过滤
        if self.min_score > -999:
            # 确保列存在 (防止 score_smoothed 是 NaN 的情况被错误过滤，如果不平滑，列可能不存在)
            if sort_score_col in pred_df.columns:
                pred_df = pred_df[pred_df[sort_score_col] >= self.min_score]

        filtered_count = len(pred_df)
        if filtered_count == 0:
            return pd.DataFrame(columns=["date", "symbol", "weight"])

        # ==========================
        # 排序选股 & 仓位控制
        # ==========================
        # [修改] 使用动态确定的列名进行排序
        sorted_df = pred_df.sort_values(by=["date", sort_score_col], ascending=[True, False])
        
        top_picks = sorted_df.groupby("date").head(self.top_k).copy()
        
        # 基础权重
        top_picks["base_weight"] = 1.0 / self.top_k
        
        # 动态仓位系数
        pos_df = self._calc_position_ratio(top_picks["date"])
        top_picks = pd.merge(top_picks, pos_df, on="date", how="left")
        
        # 最终权重
        top_picks["weight"] = top_picks["base_weight"] * top_picks["pos_ratio"]
        
        # 返回结果 (带 pos_ratio)
        signal_df = top_picks[["date", "symbol", "weight", "pos_ratio"]].copy()
        
        return signal_df.sort_values(["date", "symbol"]).reset_index(drop=True)