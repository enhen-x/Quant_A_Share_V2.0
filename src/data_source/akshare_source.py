# src/data_source/akshare_source.py

import akshare as ak
import pandas as pd
import time
import random  # 新增 random 库
from src.data_source.base import BaseDataSource
from src.utils.logger import get_logger
from src.utils.config import GLOBAL_CONFIG

logger = get_logger()

class AkShareSource(BaseDataSource):
    def __init__(self):
        pass

    def _fetch_with_retry(self, func, *args, max_retries=5, base_sleep=5, **kwargs):
        """
        通用重试装饰器 (升级版: 指数退避策略)
        max_retries: 增加重试次数到 5 次
        base_sleep: 基础等待时间增加到 5 秒
        """
        for i in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # 指数退避: 每次失败等待时间翻倍 (5s -> 10s -> 20s -> 40s ...)
                # 加上一点随机数防止并发时的共振
                sleep_time = base_sleep * (2 ** i) + random.uniform(0, 2)
                
                logger.warning(f"接口触发风控或网络波动 (尝试 {i+1}/{max_retries})，暂停 {sleep_time:.1f}秒后重试... 错误: {e}")
                time.sleep(sleep_time)
        
        logger.error("达到最大重试次数，服务端拒绝连接，放弃调用。")
        return None

    def get_stock_list(self) -> pd.DataFrame:
        """获取 A 股列表"""
        logger.info("正在获取 A 股股票列表...")
        df = None
        
        try:
            # 获取列表通常只需要请求一次，不容易触发风控
            df = self._fetch_with_retry(ak.stock_zh_a_spot_em, max_retries=3, base_sleep=3)
        except Exception:
            pass

        if df is None or df.empty:
            logger.warning("主接口获取失败，切换至备用轻量接口 (stock_info_a_code_name)...")
            try:
                df = self._fetch_with_retry(ak.stock_info_a_code_name, max_retries=3, base_sleep=3)
                if df is not None:
                    df = df.rename(columns={"code": "symbol", "name": "name"})
            except Exception:
                return pd.DataFrame()

        if df is None or df.empty:
            return pd.DataFrame()

        # 字段标准化
        if "代码" in df.columns:
            df = df.rename(columns={"代码": "symbol", "名称": "name"})
        
        df = df[["symbol", "name"]]
        
        # 规则过滤
        stock_pool_cfg = GLOBAL_CONFIG.get("data", {}).get("stock_pool", {})
        if stock_pool_cfg.get("exclude_st", True):
            df = df[~df["name"].str.contains("ST", na=False)]
            df = df[~df["name"].str.contains("退", na=False)]

        if not stock_pool_cfg.get("include_kcb", False):
            df = df[~df["symbol"].str.startswith("688")]
        if not stock_pool_cfg.get("include_cyb", False):
            df = df[~df["symbol"].str.startswith("300")]
        if not stock_pool_cfg.get("include_bj", False):
            df = df[~df["symbol"].str.match(r"^(8|4|92)")]

        return df.reset_index(drop=True)

    def get_price(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """获取个股日线历史数据"""
        start_dt = start_date.replace("-", "")
        end_dt = end_date.replace("-", "")
        
        # === 核心修改点：随机延时 ===
        # 每次请求前随机睡眠 0.5 ~ 1.5 秒
        # 这种不规律的请求间隔能极大降低被识别为机器人的概率
        time.sleep(random.uniform(3, 6)) 

        try:
            df = self._fetch_with_retry(
                ak.stock_zh_a_hist,
                symbol=symbol, 
                period="daily", 
                start_date=start_dt, 
                end_date=end_dt, 
                adjust="qfq"
            )
        except Exception:
            return pd.DataFrame()

        if df is None or df.empty:
            return pd.DataFrame()

        rename_map = {
            "日期": "date", "开盘": "open", "收盘": "close", 
            "最高": "high", "最低": "low", "成交量": "volume", "成交额": "amount"
        }
        df = df.rename(columns=rename_map)
        df["date"] = pd.to_datetime(df["date"])
        
        required_cols = ["open", "high", "low", "close", "volume", "amount"]
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0.0
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df[["date"] + required_cols]

    def get_trade_calendar(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取交易日历"""
        try:
            df = self._fetch_with_retry(ak.tool_trade_date_hist_sina)
            df = df.rename(columns={"trade_date": "date"})
            df["date"] = pd.to_datetime(df["date"])
            mask = (df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))
            return df[mask].reset_index(drop=True)
        except Exception as e:
            logger.error(f"获取交易日历失败: {e}")
            return pd.DataFrame()