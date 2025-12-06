# scripts/clean_and_check.py

import os
import sys
import pandas as pd
from tqdm import tqdm

# 路径适配
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.config import GLOBAL_CONFIG
from src.utils.io import read_parquet, save_parquet, ensure_dir, save_csv
from src.utils.logger import get_logger
from src.data_source.datahub import DataHub  # <--- 重新引入 DataHub

logger = get_logger()

class DataCleaner:
    def __init__(self):
        self.raw_dir = GLOBAL_CONFIG["paths"]["data_raw"]
        self.cleaned_dir = GLOBAL_CONFIG["paths"].get("data_cleaned", 
                                                      os.path.join(GLOBAL_CONFIG["paths"]["data_root"], "raw_cleaned"))
        self.report_path = os.path.join(self.cleaned_dir, "data_quality_report.csv")
        
        ensure_dir(self.cleaned_dir)
        
        # === 修改点：通过 DataHub 读取本地交易日历 ===
        self.datahub = DataHub()
        logger.info("正在调用 DataHub 读取本地交易日历...")
        
        self.calendar_df = self.datahub.load_local_trade_calendar()
        
        if self.calendar_df.empty:
            logger.error("交易日历加载失败，请检查是否已运行 init_stock_pool.py")
            raise FileNotFoundError("Trade calendar is empty")
            
        self.trade_dates = set(pd.to_datetime(self.calendar_df["date"]).dt.date)

    def check_and_clean_single(self, file_path: str) -> dict:
        # ... (这部分逻辑保持不变，复用之前的代码即可) ...
        # 为节省篇幅，这里仅展示关键的 check_and_clean_single 逻辑
        # 你可以直接复制上一个回复中的 check_and_clean_single 方法实现
        
        file_name = os.path.basename(file_path)
        symbol = file_name.replace(".parquet", "")
        
        try:
            df = read_parquet(file_path)
        except Exception as e:
            logger.error(f"读取失败 {file_name}: {e}")
            return None

        if "date" not in df.columns:
            return {"symbol": symbol, "status": "ERROR_NO_DATE"}
        
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        
        initial_count = len(df)
        stats = {
            "symbol": symbol,
            "total_rows": initial_count,
            "start_date": df["date"].min(),
            "end_date": df["date"].max(),
            "n_duplicates": 0, "n_nan_price": 0, "n_zero_price": 0,
            "n_suspension": 0, "n_missing_days": 0, "clean_rows": 0,
            "status": "OK"
        }

        # 1. 去重
        if df["date"].duplicated().any():
            stats["n_duplicates"] = df["date"].duplicated().sum()
            df = df.drop_duplicates(subset=["date"], keep="last")

        # 2. 价格清洗
        price_cols = ["open", "high", "low", "close"]
        existing_cols = [c for c in price_cols if c in df.columns]
        nan_mask = df[existing_cols].isnull().any(axis=1)
        stats["n_nan_price"] = nan_mask.sum()
        df = df[~nan_mask]
        
        zero_mask = (df[existing_cols] <= 1e-6).any(axis=1)
        stats["n_zero_price"] = zero_mask.sum()
        df = df[~zero_mask]

        # 3. 停牌统计
        if "volume" in df.columns:
            suspension_mask = (df["volume"] < 1e-6)
            stats["n_suspension"] = suspension_mask.sum()

        # 4. 日期缺失检查
        if not df.empty:
            s_date = df["date"].min().date()
            e_date = df["date"].max().date()
            expected_dates = {d for d in self.trade_dates if s_date <= d <= e_date}
            actual_dates = set(df["date"].dt.date)
            missing_dates = expected_dates - actual_dates
            stats["n_missing_days"] = len(missing_dates)

        stats["clean_rows"] = len(df)
        if len(df) > 0:
            save_path = os.path.join(self.cleaned_dir, f"{symbol}.parquet")
            save_parquet(df, save_path)
        else:
            stats["status"] = "EMPTY_AFTER_CLEAN"

        return stats

    def run(self):
        # ... (保持不变) ...
        logger.info(f"=== 开始数据清洗与质检 (DataHub Mode) ===")
        # 剩下的 run 方法逻辑与之前一致
        if not os.path.exists(self.raw_dir):
            logger.warning("原始数据目录不存在")
            return

        files = [f for f in os.listdir(self.raw_dir) if f.endswith(".parquet") and f[0].isdigit()]
        
        results = []
        for f in tqdm(files, desc="Cleaning"):
            stats = self.check_and_clean_single(os.path.join(self.raw_dir, f))
            if stats: results.append(stats)
            
        if results:
            df_report = pd.DataFrame(results)
            # 简单的计算
            df_report["suspension_ratio"] = df_report["n_suspension"] / df_report["total_rows"]
            denom = df_report["clean_rows"] + df_report["n_missing_days"]
            df_report["missing_ratio"] = df_report["n_missing_days"] / (denom.replace(0, 1))
            df_report = df_report.sort_values(by=["missing_ratio"], ascending=False)
            
            save_csv(df_report, self.report_path)
            logger.info(f"清洗完成，报告已保存: {self.report_path}")

if __name__ == "__main__":
    cleaner = DataCleaner()
    cleaner.run()