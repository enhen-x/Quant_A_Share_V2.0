# src/data_source/datahub.py

import os
import pandas as pd
from typing import Optional, List

from src.utils.config import GLOBAL_CONFIG
from src.utils.logger import get_logger
from src.utils.io import read_parquet, read_csv
from src.data_source.akshare_source import AkShareSource
from src.data_source.baostock_source import BaostockSource

logger = get_logger()

class DataHub:
    """
    数据中枢：负责调度数据源（网络下载）与本地存储（文件读取）。
    """

    def __init__(self):
        self.config = GLOBAL_CONFIG
        self.paths = self.config["paths"]
        self._source = None  # 初始为 None
        
        # 默认使用 AkShare 作为数据源（因为 AkShare 对列表和日历支持更好）
        # 也可以根据配置切换
    @property
    def source(self):
        """懒加载：只有在真正需要联网下载时，才初始化 Source 并登录"""
        if self._source is None:
            # 这里才引用和实例化
            from src.data_source.baostock_source import BaostockSource
            self._source = BaostockSource()
        return self._source
        # 如果需要 Baostock 的行情，可以在 fetch_price 里单独处理，或者在这里做更复杂的工厂模式
        
    # ==========================
    # 1. 股票列表 (List)
    # ==========================

    def get_stock_list(self) -> pd.DataFrame:
        """[网络] 从数据源获取最新的股票列表"""
        return self.source.get_stock_list()

    def load_local_stock_list(self) -> pd.DataFrame:
        """[本地] 读取本地已保存的股票列表元数据"""
        meta_path = os.path.join(self.paths["data_meta"], "all_stocks_meta.parquet")
        if os.path.exists(meta_path):
            return read_parquet(meta_path)
        logger.warning(f"本地股票列表不存在: {meta_path}")
        return pd.DataFrame()

    # ==========================
    # 2. 交易日历 (Calendar)
    # ==========================

    def get_trade_calendar(self, start_date=None, end_date=None) -> pd.DataFrame:
        """[网络] 获取交易日历"""
        # 如果不传参，默认取配置里的范围；如果传参（如初始化时），则用参数
        s_date = start_date or self.config["data"]["start_date"]
        e_date = end_date or self.config["data"]["end_date"]
        return self.source.get_trade_calendar(s_date, e_date)

    def load_local_trade_calendar(self) -> pd.DataFrame:
        """
        [本地] 读取本地交易日历
        """
        cal_path = os.path.join(self.paths["data_meta"], "trade_calendar.parquet")
        if os.path.exists(cal_path):
            logger.info(f"正在从 DataHub 读取本地交易日历: {cal_path}")
            return read_parquet(cal_path)
        
        logger.error(f"本地交易日历文件缺失: {cal_path}，请先运行 init_stock_pool.py")
        return pd.DataFrame()

    # ==========================
    # 3. 个股行情 (Price)
    # ==========================

    def fetch_price(self, symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """[网络] 下载行情"""
        s_date = start_date or self.config["data"]["start_date"]
        e_date = end_date or self.config["data"]["end_date"]
        # 这里为了演示，假设使用 source (AkShareSource 或 BaostockSource)
        return self.source.get_price(symbol, s_date, e_date)

    def load_local_price(self, symbol: str) -> Optional[pd.DataFrame]:
        """[本地] 读取原始行情 (Raw)"""
        raw_dir = self.paths["data_raw"]
        parquet_path = os.path.join(raw_dir, f"{symbol}.parquet")
        if os.path.exists(parquet_path):
            return read_parquet(parquet_path)
        return None

    # ==========================
    # 4. 指数行情 (Index)
    # ==========================
    
    def fetch_index_price(self, index_code: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        [网络] 下载指数行情
        :param index_code: 如 "000300.SH"
        :param start_date: 开始日期，默认使用配置中的 start_date
        :param end_date: 结束日期，默认使用配置中的 end_date
        """
        s_date = start_date or self.config["data"]["start_date"]
        e_date = end_date or self.config["data"]["end_date"]
        
        # 通过 BaostockSource 获取指数数据
        return self.source.get_index_price(index_code, s_date, e_date)
    

    # 5. 清洗后数据 (Cleaned Data) [新增]
    # ==========================
    def load_cleaned_price(self, symbol: str) -> Optional[pd.DataFrame]:
        """读取清洗后的个股数据 (从 data/raw_cleaned)"""
        # 注意：这里读取的是 config 中定义的 data_cleaned 路径
        cleaned_dir = self.paths.get("data_cleaned", 
                                   os.path.join(self.paths["data_root"], "raw_cleaned"))
        path = os.path.join(cleaned_dir, f"{symbol}.parquet")
        
        if os.path.exists(path):
            return read_parquet(path)
        return None

    def get_cleaned_stock_list(self) -> List[str]:
        """获取所有已清洗的股票代码列表"""
        cleaned_dir = self.paths.get("data_cleaned", 
                                   os.path.join(self.paths["data_root"], "raw_cleaned"))
        if not os.path.exists(cleaned_dir):
            return []
        
        files = [f for f in os.listdir(cleaned_dir) if f.endswith(".parquet")]
        return [f.replace(".parquet", "") for f in files]