# src/data_source/datahub.py

import os
import pandas as pd
from typing import Optional, List

from src.utils.config import GLOBAL_CONFIG
from src.utils.logger import get_logger
from src.utils.io import read_parquet, read_csv
from src.data_source.akshare_source import AkShareSource
from src.data_source.baostock_source import BaostockSource

# 如果未来要支持 Baostock，可以再导入 BaostockSource

logger = get_logger()

class DataHub:
    """
    数据中枢：负责调度数据源（网络下载）与本地存储（文件读取）。
    上层策略或训练代码应通过 DataHub 获取数据，而不是直接调用 Source。
    """

    def __init__(self):
        self.config = GLOBAL_CONFIG
        self.paths = self.config["paths"]
        
        # 初始化数据源
        # 这里默认使用 AkShare，如果未来支持多源切换，可判断 config['data']['source']
        self.source = BaostockSource()
        
    # ==========================
    # 1. 股票列表与元数据
    # ==========================

    def get_stock_list(self) -> pd.DataFrame:
        """从数据源获取最新的股票列表（经过配置过滤）"""
        return self.source.get_stock_list()

    def get_stock_list_from_local(self) -> List[str]:
        """
        从本地 raw 目录的文件名推断股票列表。
        适用于离线模式，或者只需处理已下载股票的场景。
        """
        raw_dir = self.paths["data_raw"]
        if not os.path.exists(raw_dir):
            logger.warning(f"本地原始数据目录不存在: {raw_dir}")
            return []

        symbols = []
        for fname in os.listdir(raw_dir):
            if fname.endswith(".parquet") or fname.endswith(".csv"):
                # 文件名通常是 "600000.parquet" -> "600000"
                code = os.path.splitext(fname)[0]
                symbols.append(code)
        
        return sorted(list(set(symbols)))

    def get_trade_calendar(self, start_date=None, end_date=None) -> pd.DataFrame:
        """获取交易日历"""
        s_date = start_date or self.config["data"]["start_date"]
        e_date = end_date or self.config["data"]["end_date"]
        return self.source.get_trade_calendar(s_date, e_date)

    # ==========================
    # 2. 个股行情 (Price)
    # ==========================

    def fetch_price(self, symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        [网络] 强制从数据源下载行情数据。
        """
        s_date = start_date or self.config["data"]["start_date"]
        e_date = end_date or self.config["data"]["end_date"]
        return self.source.get_price(symbol, s_date, e_date)

    def load_local_price(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        [本地] 读取本地已下载的行情数据 (Raw Data)。
        优先读取 Parquet，其次 CSV。
        """
        raw_dir = self.paths["data_raw"]
        
        # 1. 尝试 Parquet
        parquet_path = os.path.join(raw_dir, f"{symbol}.parquet")
        if os.path.exists(parquet_path):
            return read_parquet(parquet_path)
            
        # 2. 尝试 CSV
        csv_path = os.path.join(raw_dir, f"{symbol}.csv")
        if os.path.exists(csv_path):
            return read_csv(csv_path)

        return None

    # ==========================
    # 3. 指数行情 (Index)
    # ==========================

    def fetch_index_price(self, index_code: str) -> pd.DataFrame:
        """
        [网络] 获取指数行情 (例如 000300.SH)。
        使用 AkShare 接口。
        """
        import akshare as ak
        
        # 格式转换: "000300.SH" -> "sh000300" (AkShare 格式)
        code = index_code.split(".")[0]
        exchange = index_code.split(".")[-1].lower() if "." in index_code else ""
        
        # 简单判断市场前缀
        ak_code = f"{exchange}{code}" # sh000300
        if not exchange: 
            # 如果没后缀，尝试根据代码判断（简易逻辑）
            ak_code = f"sh{code}" if code.startswith("000") else f"sz{code}"

        logger.info(f"正在下载指数数据: {index_code} -> {ak_code}")
        
        try:
            # ak.stock_zh_index_daily 接口获取指数历史
            df = ak.stock_zh_index_daily(symbol=ak_code)
            
            if df is None or df.empty:
                logger.warning(f"指数数据为空: {ak_code}")
                return pd.DataFrame()

            # 标准化列名
            df = df.rename(columns={
                "date": "date", 
                "open": "open", 
                "high": "high", 
                "low": "low", 
                "close": "close", 
                "volume": "volume"
            })
            
            # 确保日期格式
            df["date"] = pd.to_datetime(df["date"])
            
            # 筛选时间范围
            start_date = pd.to_datetime(self.config["data"]["start_date"])
            end_date = pd.to_datetime(self.config["data"]["end_date"])
            mask = (df["date"] >= start_date) & (df["date"] <= end_date)
            
            return df[mask].reset_index(drop=True)

        except Exception as e:
            logger.error(f"获取指数数据失败 {index_code}: {e}")
            return pd.DataFrame()