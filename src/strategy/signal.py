# src/strategy/signal.py

import pandas as pd
import numpy as np
import os
from src.utils.logger import get_logger
from src.utils.config import GLOBAL_CONFIG
from src.utils.io import read_parquet

logger = get_logger()

class TopKSignalStrategy:
    def __init__(self):
        self.conf = GLOBAL_CONFIG["strategy"]
        self.top_k = self.conf.get("top_k", 10)
        self.min_score = self.conf.get("min_pred", 0.0)
        
        # === 读取配置 (兼容旧命名) ===
        filter_cfg = GLOBAL_CONFIG.get("preprocessing", {}).get("filter", {})
        
        # 1. 价格 (Price)
        self.min_price = filter_cfg.get("min_price", 0.0)      # 新增
        self.max_price = filter_cfg.get("max_price", 99999.0)  # 保留
        
        # 2. 成交额 (Amount) - 【关键兼容】
        # 使用旧名字 min_turnover 代表成交额
        self.min_amount = filter_cfg.get("min_turnover", 0) 
        
        # 3. 换手率 (Turnover Rate) - 【新增明确命名】
        self.min_turnover_rate = filter_cfg.get("min_turnover_rate", 0.0)
        
        # 4. 市值 (Market Cap) - 动态计算
        self.min_mcap = filter_cfg.get("min_mcap", 0)
        self.max_mcap = filter_cfg.get("max_mcap", float("inf"))

        # 5. 其他
        self.exclude_st = filter_cfg.get("exclude_st", True)
        
        self.paths = GLOBAL_CONFIG["paths"]
        
        logger.info(f"策略风控参数 | 价格: {self.min_price}~{self.max_price}, "
                    f"换手率>={self.min_turnover_rate}%, 成交额>={self.min_amount}")
        logger.info(f"市值限制: {self.min_mcap/1e8}亿 ~ {self.max_mcap/1e8}亿")

    def _load_filter_data(self):
        """加载辅助数据: amount, turnover, close"""
        stock_path = os.path.join(self.paths["data_processed"], "all_stocks.parquet")
        if not os.path.exists(stock_path):
            logger.error("缺少 all_stocks.parquet，无法进行风控过滤！")
            return None, None
            
        df_stocks = read_parquet(stock_path)
        # 确保包含计算市值所需的列
        cols = ["date", "symbol", "turnover", "amount", "close"]
        df_stocks = df_stocks[[c for c in cols if c in df_stocks.columns]]
        
        meta_path = os.path.join(self.paths["data_meta"], "all_stocks_meta.parquet")
        if os.path.exists(meta_path):
            df_meta = read_parquet(meta_path)[["symbol", "name"]]
        else:
            df_meta = pd.DataFrame(columns=["symbol", "name"])
            
        return df_stocks, df_meta

    def generate(self, pred_df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"正在生成信号 (应用兼容性配置)...")
        initial_count = len(pred_df)
        
        df_stocks, df_meta = self._load_filter_data()
        
        # 合并行情
        if df_stocks is not None:
            pred_df = pd.merge(pred_df, df_stocks, on=["date", "symbol"], how="inner", suffixes=("", "_raw"))
            
        # 合并元数据
        if not df_meta.empty:
            pred_df = pd.merge(pred_df, df_meta, on="symbol", how="left")

        # ==========================
        # 执行过滤 (Filter)
        # ==========================
        
        # 1. 价格区间 (min_price & max_price)
        close_col = "close_raw" if "close_raw" in pred_df.columns else "close"
        if close_col in pred_df.columns:
            pred_df = pred_df[
                (pred_df[close_col] >= self.min_price) & 
                (pred_df[close_col] <= self.max_price)
            ]

        # 2. 换手率 (min_turnover_rate) -> 对应列 turnover
        if "turnover" in pred_df.columns and self.min_turnover_rate > 0:
            pred_df["turnover"] = pred_df["turnover"].fillna(0)
            pred_df = pred_df[pred_df["turnover"] >= self.min_turnover_rate]
            
        # 3. 成交额 (min_turnover 配置) -> 对应列 amount
        if "amount" in pred_df.columns and self.min_amount > 0:
            pred_df["amount"] = pred_df["amount"].fillna(0)
            pred_df = pred_df[pred_df["amount"] >= self.min_amount]

        # 4. 市值过滤 (min_mcap & max_mcap) - 动态估算
        # 逻辑：市值 = 成交额 / (换手率%)
        if "amount" in pred_df.columns and "turnover" in pred_df.columns:
            # 避免除以 0
            valid_mcap_mask = pred_df["turnover"] > 0.001
            # 计算临时市值列
            pred_df.loc[valid_mcap_mask, "est_mcap"] = (
                pred_df.loc[valid_mcap_mask, "amount"] / 
                (pred_df.loc[valid_mcap_mask, "turnover"] * 0.01)
            )
            
            # 应用过滤
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

        # 6. ST 过滤
        if self.exclude_st and "name" in pred_df.columns:
            mask_st = pred_df["name"].str.contains("ST", na=False) | pred_df["name"].str.contains("退", na=False)
            pred_df = pred_df[~mask_st]

        # 7. 最低分数
        if self.min_score > -999:
            pred_df = pred_df[pred_df["pred_score"] >= self.min_score]

        filtered_count = len(pred_df)
        drop_rate = 1 - filtered_count/initial_count if initial_count > 0 else 0
        logger.info(f"过滤后剩余: {filtered_count} (剔除率 {drop_rate:.1%})")

        if filtered_count == 0:
            return pd.DataFrame(columns=["date", "symbol", "weight"])

        # ==========================
        # 排序选股
        # ==========================
        sorted_df = pred_df.sort_values(by=["date", "pred_score"], ascending=[True, False])
        top_picks = sorted_df.groupby("date").head(self.top_k).copy()
        top_picks["weight"] = 1.0 / self.top_k
        
        signal_df = top_picks[["date", "symbol", "weight"]].copy()
        return signal_df.sort_values(["date", "symbol"]).reset_index(drop=True)