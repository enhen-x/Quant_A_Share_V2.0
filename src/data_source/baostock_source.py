# src/data_source/baostock_source.py

import baostock as bs
import pandas as pd
import datetime
from src.data_source.base import BaseDataSource
from src.utils.logger import get_logger
from src.utils.config import GLOBAL_CONFIG

logger = get_logger()

class BaostockSource(BaseDataSource):
    def __init__(self):
        """初始化并登录 Baostock"""
        self.system = bs.login()
        if self.system.error_code != '0':
            logger.error(f"Baostock 登录失败: {self.system.error_msg}")
        else:
            logger.info("Baostock 登录成功")

    def __del__(self):
        """析构时退出登录"""
        bs.logout()

    def get_stock_list(self) -> pd.DataFrame:
        """
        获取 A 股股票列表
        使用 query_all_stock 接口获取指定日期的全市场代码
        """
        logger.info("正在从 Baostock 获取全市场股票列表...")
        
        data_list = []
        rs = None
        
        # 尝试获取最近 10 天内的有效列表 (因为如果是周末或节假日，当天可能没数据)
        for i in range(10):
            date_target = (datetime.datetime.now() - datetime.timedelta(days=i)).strftime("%Y-%m-%d")
            # 接口: query_all_stock(day="YYYY-MM-DD")
            rs = bs.query_all_stock(day=date_target)
            
            if rs.error_code == '0' and rs.next():
                # 只要能取到第一行，说明这天有数据，重置游标并读取
                # Baostock 的 rs.next() 会消耗一行，所以我们这里判断后需要重新循环获取
                # 或者更简单的：直接读完，看长度
                current_list = []
                # 注意：刚才调用了一次 next()，如果不重新 query，第一行就丢了
                # 所以确认有数据后，需要重新 query 一次，或者手动把第一行加进去
                # 为简单起见，这里逻辑是：只要 rs 里有数据，我们就用这个 rs
                
                # 重新查询一遍确保完整
                rs = bs.query_all_stock(day=date_target)
                while rs.error_code == '0' and rs.next():
                    current_list.append(rs.get_row_data())
                
                if current_list:
                    data_list = current_list
                    logger.info(f"成功获取股票列表，日期: {date_target}")
                    break
        
        if not data_list:
            logger.warning("Baostock 未返回股票列表（连续10天无数据），请检查网络或 Baostock 服务状态。")
            return pd.DataFrame()

        df = pd.DataFrame(data_list, columns=rs.fields)
        # query_all_stock 返回列: ["code", "tradeStatus", "code_name"]
        
        # === 1. 字段标准化 ===
        df = df.rename(columns={
            "code": "symbol",
            "code_name": "name"
        })
        
        # 转换 symbol: "sh.600000" -> "600000"
        df["bs_code"] = df["symbol"] # 保留原始带前缀代码，后续查价格可能用到
        df["symbol"] = df["symbol"].apply(lambda x: x.split(".")[-1])
        
        # 补充 list_date (query_all_stock 不返回上市日期，给个默认值避免报错)
        df["list_date"] = "1990-01-01" 

        # === 2. 规则过滤 ===
        stock_pool_cfg = GLOBAL_CONFIG.get("data", {}).get("stock_pool", {})
        
        # 过滤停牌 (tradeStatus=1 为交易)
        if stock_pool_cfg.get("only_tradable", True):
             # 注意 Baostock 返回的是字符串 "1"
             df = df[df["tradeStatus"] == "1"]

        # 过滤 ST
        if stock_pool_cfg.get("exclude_st", True):
            df = df[~df["name"].str.contains("ST", na=False)]
            df = df[~df["name"].str.contains("退", na=False)]

        # 过滤板块
        if not stock_pool_cfg.get("include_kcb", False):
            df = df[~df["symbol"].str.startswith("688")]
        if not stock_pool_cfg.get("include_cyb", False):
            df = df[~df["symbol"].str.startswith("300")]
        if not stock_pool_cfg.get("include_bj", False):
            df = df[~df["symbol"].str.match(r"^(8|4|92)")]

        logger.info(f"股票列表获取完成，共 {len(df)} 只。")
        return df.reset_index(drop=True)

    def get_price(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        获取日线数据
        """
        # 构造 Baostock 需要的带前缀代码 (sh.xxxxxx)
        if "." not in symbol:
            if symbol.startswith("6"):
                bs_symbol = f"sh.{symbol}"
            elif symbol.startswith(("0", "3")):
                bs_symbol = f"sz.{symbol}"
            elif symbol.startswith(("4", "8")):
                bs_symbol = f"bj.{symbol}"
            else:
                bs_symbol = f"sh.{symbol}"
        else:
            bs_symbol = symbol

        # adjustflag: 2=前复权
        fields = "date,open,high,low,close,volume,amount,turn"
        
        rs = bs.query_history_k_data_plus(
            bs_symbol,
            fields,
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="2"
        )
        
        data_list = []
        while (rs.error_code == '0') and rs.next():
            data_list.append(rs.get_row_data())
            
        if not data_list:
            return pd.DataFrame()

        df = pd.DataFrame(data_list, columns=fields.split(","))
        
        # 类型转换
        df["date"] = pd.to_datetime(df["date"])
        num_cols = ["open", "high", "low", "close", "volume", "amount", "turn"]
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
        df = df.rename(columns={"turn": "turnover"})

        return df

    def get_trade_calendar(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取交易日历"""
        rs = bs.query_trade_dates(start_date=start_date, end_date=end_date)
        
        data_list = []
        while (rs.error_code == '0') and rs.next():
            data_list.append(rs.get_row_data())
            
        if not data_list:
            return pd.DataFrame()

        df = pd.DataFrame(data_list, columns=rs.fields)
        # 筛选交易日
        df = df[df["is_trading_day"] == "1"]
        df = df.rename(columns={"calendar_date": "date"})
        df["date"] = pd.to_datetime(df["date"])
        
        return df[["date"]].reset_index(drop=True)