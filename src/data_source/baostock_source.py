# src/data_source/baostock_source.py

import datetime
import io
import os
import contextlib

import baostock as bs
import pandas as pd

from src.data_source.base import BaseDataSource
from src.utils.config import GLOBAL_CONFIG
from src.utils.logger import get_logger

logger = get_logger()


class BaostockSource(BaseDataSource):
    def __init__(self):
        """Initialize and login to Baostock."""
        retry_cfg = GLOBAL_CONFIG.get("data", {}).get("retry", {})
        self.quiet_baostock_console = bool(retry_cfg.get("quiet_baostock_console", True))

        with self._silent_baostock_console():
            self.system = bs.login()
        if self.system.error_code != "0":
            logger.error(f"Baostock login failed: {self.system.error_msg}")
        else:
            logger.info("Baostock login success")

    @contextlib.contextmanager
    def _silent_baostock_console(self):
        """
        Suppress third-party direct console output.
        Keep our own logger output unchanged.
        """
        if not self.quiet_baostock_console:
            yield
            return

        devnull_fd = None
        stdout_fd = None
        stderr_fd = None
        try:
            try:
                devnull_fd = os.open(os.devnull, os.O_RDWR)
                stdout_fd = os.dup(1)
                stderr_fd = os.dup(2)
                os.dup2(devnull_fd, 1)
                os.dup2(devnull_fd, 2)
            except OSError:
                # Fallback to Python-level redirect only.
                devnull_fd = None
                stdout_fd = None
                stderr_fd = None

            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                yield
        finally:
            if stdout_fd is not None:
                try:
                    os.dup2(stdout_fd, 1)
                except OSError:
                    pass
                try:
                    os.close(stdout_fd)
                except OSError:
                    pass
            if stderr_fd is not None:
                try:
                    os.dup2(stderr_fd, 2)
                except OSError:
                    pass
                try:
                    os.close(stderr_fd)
                except OSError:
                    pass
            if devnull_fd is not None:
                try:
                    os.close(devnull_fd)
                except OSError:
                    pass

    def __del__(self):
        """Logout safely on object cleanup."""
        try:
            if bs:
                with self._silent_baostock_console():
                    bs.logout()
        except (AttributeError, ImportError, TypeError):
            pass
        except Exception:
            pass

    def get_stock_list(self) -> pd.DataFrame:
        """Fetch A-share stock list from Baostock."""
        logger.info("Fetching stock list from Baostock...")

        data_list = []
        rs = None

        # Try the latest 10 days, because weekend/holiday may have no data.
        for i in range(10):
            date_target = (datetime.datetime.now() - datetime.timedelta(days=i)).strftime("%Y-%m-%d")
            with self._silent_baostock_console():
                rs = bs.query_all_stock(day=date_target)

            if rs.error_code != "0":
                logger.warning(
                    f"query_all_stock failed on {date_target}: "
                    f"error_code={rs.error_code}, error_msg={rs.error_msg}"
                )
                continue

            current_list = []
            with self._silent_baostock_console():
                while rs.next():
                    current_list.append(rs.get_row_data())

            if current_list:
                data_list = current_list
                logger.info(f"stock list fetched on {date_target}")
                break

        if not data_list:
            logger.warning("Baostock returned no stock list in recent 10 days")
            return pd.DataFrame()

        df = pd.DataFrame(data_list, columns=rs.fields)
        df = df.rename(columns={"code": "symbol", "code_name": "name"})

        # Keep full code and strip market prefix for local filename usage.
        df["bs_code"] = df["symbol"]
        df["symbol"] = df["symbol"].apply(lambda x: x.split(".")[-1])

        # query_all_stock does not return list date.
        df["list_date"] = "1990-01-01"

        stock_pool_cfg = GLOBAL_CONFIG.get("data", {}).get("stock_pool", {})

        if stock_pool_cfg.get("only_tradable", True):
            df = df[df["tradeStatus"] == "1"]

        if stock_pool_cfg.get("exclude_st", True):
            df = df[~df["name"].str.contains("ST", na=False)]
            df = df[~df["name"].str.contains("退", na=False)]

        if not stock_pool_cfg.get("include_kcb", False):
            df = df[~df["symbol"].str.startswith("688")]
        if not stock_pool_cfg.get("include_cyb", False):
            df = df[~df["symbol"].str.startswith("300")]
        if not stock_pool_cfg.get("include_bj", False):
            df = df[~df["symbol"].str.match(r"^(8|4|92)")]

        logger.info(f"stock list fetched: {len(df)} symbols")
        return df.reset_index(drop=True)

    def get_price(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch daily price data."""
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

        fields = "date,open,high,low,close,volume,amount,turn"
        with self._silent_baostock_console():
            rs = bs.query_history_k_data_plus(
                bs_symbol,
                fields,
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="2",
            )
        if rs.error_code != "0":
            raise RuntimeError(
                f"Baostock get_price failed for {bs_symbol}: "
                f"error_code={rs.error_code}, error_msg={rs.error_msg}"
            )

        data_list = []
        with self._silent_baostock_console():
            while rs.next():
                data_list.append(rs.get_row_data())

        if not data_list:
            return pd.DataFrame()

        df = pd.DataFrame(data_list, columns=fields.split(","))
        df["date"] = pd.to_datetime(df["date"])
        for col in ["open", "high", "low", "close", "volume", "amount", "turn"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.rename(columns={"turn": "turnover"})
        return df

    def get_index_price(self, index_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch daily index data."""
        if "." in index_code:
            code, market = index_code.split(".")
            bs_symbol = f"{market.lower()}.{code}"
        else:
            bs_symbol = f"sh.{index_code}"

        logger.info(f"Fetching index data from Baostock: {bs_symbol}")

        fields = "date,open,high,low,close,volume,amount"
        with self._silent_baostock_console():
            rs = bs.query_history_k_data_plus(
                bs_symbol,
                fields,
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="3",
            )
        if rs.error_code != "0":
            raise RuntimeError(
                f"Baostock get_index_price failed for {bs_symbol}: "
                f"error_code={rs.error_code}, error_msg={rs.error_msg}"
            )

        data_list = []
        with self._silent_baostock_console():
            while rs.next():
                data_list.append(rs.get_row_data())

        if not data_list:
            logger.warning(f"Baostock returned empty index data: {bs_symbol}")
            return pd.DataFrame()

        df = pd.DataFrame(data_list, columns=fields.split(","))
        df["date"] = pd.to_datetime(df["date"])
        for col in ["open", "high", "low", "close", "volume", "amount"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        logger.info(f"Index data fetched: {len(df)} rows")
        return df.sort_values("date").reset_index(drop=True)

    def get_trade_calendar(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch trade calendar."""
        with self._silent_baostock_console():
            rs = bs.query_trade_dates(start_date=start_date, end_date=end_date)
        if rs.error_code != "0":
            raise RuntimeError(
                "Baostock get_trade_calendar failed: "
                f"error_code={rs.error_code}, error_msg={rs.error_msg}"
            )

        data_list = []
        with self._silent_baostock_console():
            while rs.next():
                data_list.append(rs.get_row_data())

        if not data_list:
            return pd.DataFrame()

        df = pd.DataFrame(data_list, columns=rs.fields)
        df = df[df["is_trading_day"] == "1"]
        df = df.rename(columns={"calendar_date": "date"})
        df["date"] = pd.to_datetime(df["date"])
        return df[["date"]].reset_index(drop=True)
