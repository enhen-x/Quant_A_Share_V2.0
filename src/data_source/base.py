# src/data_source/base.py
from abc import ABC, abstractmethod
import pandas as pd

class BaseDataSource(ABC):
    """数据源抽象基类"""

    @abstractmethod
    def get_stock_list(self) -> pd.DataFrame:
        """
        获取全市场股票列表。
        返回 DataFrame 需包含: symbol, name, list_date
        """
        pass

    @abstractmethod
    def get_price(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取日线行情。
        返回 DataFrame 需包含: date, open, high, low, close, volume, amount
        """
        pass

    @abstractmethod
    def get_trade_calendar(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取交易日历。
        返回 DataFrame 需包含: date
        """
        pass