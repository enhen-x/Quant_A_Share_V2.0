# scripts/clean_and_check.py

import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm

# 路径适配
current_dir = os.path.dirname(os.path.abspath(__file__))
# 从当前文件位置 (scripts/analisis) 返回两级到项目根目录
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.config import GLOBAL_CONFIG
from src.utils.io import read_parquet, save_parquet, ensure_dir, save_csv
from src.utils.logger import get_logger
from src.data_source.datahub import DataHub

logger = get_logger()

class DataCleaner:
    def __init__(self):
        self.raw_dir = GLOBAL_CONFIG["paths"]["data_raw"]
        # 输出目录：清洗后的数据
        self.cleaned_dir = GLOBAL_CONFIG["paths"].get("data_cleaned", 
                                                      os.path.join(GLOBAL_CONFIG["paths"]["data_root"], "raw_cleaned"))
        self.report_path = os.path.join(self.cleaned_dir, "data_quality_report.csv")
        
        ensure_dir(self.cleaned_dir)
        
        # === 1. 获取清洗/筛选阈值 ===
        # 优先读取 preprocessing.quality 配置，如果没有则使用默认值
        # 默认值参考：停牌率 < 10%, 日均换手 > 1%
        quality_cfg = GLOBAL_CONFIG.get("preprocessing", {}).get("quality", {})
        
        self.limit_suspension = quality_cfg.get("max_suspension_rate", 0.1) 
        self.limit_turnover = quality_cfg.get("min_avg_turnover", 1.0)     # 单位通常为 %，Baostock返回的是百分比
        
        logger.info(f"清洗阈值设定: 最大停牌率={self.limit_suspension:.1%}, 最低日均换手={self.limit_turnover}%")

        # === 2. 加载交易日历 ===
        self.datahub = DataHub()
        logger.info("正在读取本地交易日历以计算缺失率...")
        
        self.calendar_df = self.datahub.load_local_trade_calendar()
        
        if self.calendar_df.empty:
            logger.error("交易日历加载失败，请先运行 scripts/init_stock_pool.py")
            raise FileNotFoundError("Trade calendar is empty")
            
        self.trade_dates = set(pd.to_datetime(self.calendar_df["date"]).dt.date)

    def check_and_clean_single(self, file_path: str) -> dict:
        """
        处理单只股票：
        1. 读取并进行行级清洗（去重、去0值）。
        2. 计算统计指标（停牌率、换手率）。
        3. 判定是否保留该股票（REJECT vs OK）。
        4. 如果 OK，保存至 data_cleaned 目录。
        """
        file_name = os.path.basename(file_path)
        symbol = file_name.replace(".parquet", "")
        
        try:
            df = read_parquet(file_path)
        except Exception as e:
            logger.error(f"读取失败 {file_name}: {e}")
            return {"symbol": symbol, "status": "ERROR_READ"}

        if "date" not in df.columns:
            return {"symbol": symbol, "status": "ERROR_NO_DATE"}
        
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        
        initial_count = len(df)
        
        # 初始化统计字典
        stats = {
            "symbol": symbol,
            "status": "OK",             # 最终状态
            "reason": "",               # 拒绝原因
            "total_rows": initial_count,
            "start_date": df["date"].min() if not df.empty else None,
            "end_date": df["date"].max() if not df.empty else None,
            "n_duplicates": 0,
            "n_zero_price": 0,
            "n_suspension": 0,
            "suspension_ratio": 0.0,
            "avg_turnover": 0.0,
            "n_missing_days": 0,
            "clean_rows": 0
        }

        if df.empty:
            stats["status"] = "REJECT_EMPTY"
            return stats

        # ==========================
        # Step A: 行级清洗 (Row-level Cleaning)
        # ==========================
        
        # 1. 去重
        if df["date"].duplicated().any():
            stats["n_duplicates"] = df["date"].duplicated().sum()
            df = df.drop_duplicates(subset=["date"], keep="last")

        # 2. 价格异常处理 (Close <= 0 或 NaN)
        # 确保关键列存在
        price_cols = ["open", "high", "low", "close", "volume"]
        valid_cols = [c for c in price_cols if c in df.columns]
        
        # 剔除 NaN
        nan_mask = df[valid_cols].isnull().any(axis=1)
        df = df[~nan_mask]
        
        # 剔除 0 价格 (Volume 为 0 是停牌，不是脏数据，但 Price 为 0 肯定是脏数据)
        if "close" in df.columns:
            zero_price_mask = (df["close"] <= 1e-4)
            stats["n_zero_price"] = zero_price_mask.sum()
            df = df[~zero_price_mask]

        if df.empty:
            stats["status"] = "REJECT_EMPTY_AFTER_CLEAN"
            return stats

        # ==========================
        # Step B: 指标计算 (Metrics)
        # ==========================

        # 1. 停牌统计 (Volume = 0)
        # 注意：这里我们保留停牌的数据行，但要统计比例用于后续筛选
        if "volume" in df.columns:
            suspension_mask = (df["volume"] < 1e-6)
            stats["n_suspension"] = suspension_mask.sum()
            stats["suspension_ratio"] = stats["n_suspension"] / len(df)
        
        # 2. 换手率统计 (Liquidity)
        if "turnover" in df.columns:
            # 剔除停牌期间的换手率（通常为0）来计算平均活跃度，或者直接算整体均值
            # 建议：计算整体均值，因为长期停牌本身就是流动性风险
            stats["avg_turnover"] = df["turnover"].mean()
        
        # 3. 日期缺失 (Missing Days)
        s_date = df["date"].min().date()
        e_date = df["date"].max().date()
        # 理论交易日
        expected_dates = {d for d in self.trade_dates if s_date <= d <= e_date}
        actual_dates = set(df["date"].dt.date)
        missing_dates = expected_dates - actual_dates
        stats["n_missing_days"] = len(missing_dates)
        
        # 计算缺失率 (相对于由于上市时间决定的理论天数)
        denom = len(expected_dates)
        missing_ratio = len(missing_dates) / denom if denom > 0 else 0.0

        stats["clean_rows"] = len(df)

        # ==========================
        # Step C: 标的级筛选 (Stock-level Filter) [核心修改点]
        # ==========================
        
        # 规则 1: 停牌率过高
        if stats["suspension_ratio"] > self.limit_suspension:
            stats["status"] = "REJECT"
            stats["reason"] = "HIGH_SUSPENSION"
            return stats # 直接返回，不保存

        # 规则 2: 流动性枯竭 (僵尸股)
        # 注意：Baostock turnover 单位通常是 % (例如 1.5 代表 1.5%)
        # 如果你用的是其他源，请确认单位。这里假设阈值 1.0 代表 1%
        if "turnover" in df.columns and stats["avg_turnover"] < self.limit_turnover:
            stats["status"] = "REJECT"
            stats["reason"] = "LOW_LIQUIDITY"
            return stats

        # 规则 3: 数据严重缺失 (可选，防止上市时间虽长但中间缺了一大块)
        if missing_ratio > 0.5: # 缺失超过 50%
            stats["status"] = "REJECT"
            stats["reason"] = "HIGH_MISSING"
            return stats

        # ==========================
        # Step D: 保存有效数据
        # ==========================
        save_path = os.path.join(self.cleaned_dir, f"{symbol}.parquet")
        save_parquet(df, save_path)
        
        return stats

    def run(self):
        logger.info(f"=== 开始数据清洗与质检 (v2.0 增强版) ===")
        logger.info(f"源数据: {self.raw_dir}")
        logger.info(f"输出目标: {self.cleaned_dir}")
        
        if not os.path.exists(self.raw_dir):
            logger.warning("原始数据目录不存在")
            return

        # 获取所有 raw 数据
        files = [f for f in os.listdir(self.raw_dir) if f.endswith(".parquet") and f[0].isdigit()]
        logger.info(f"待处理文件数: {len(files)}")
        
        results = []
        rejected_count = 0
        
        for f in tqdm(files, desc="Cleaning"):
            stats = self.check_and_clean_single(os.path.join(self.raw_dir, f))
            if stats:
                results.append(stats)
                if stats["status"] != "OK":
                    rejected_count += 1
            
        # 生成报告
        if results:
            df_report = pd.DataFrame(results)
            
            # 简单排序：先看 Rejected 的，再按缺失率排
            df_report = df_report.sort_values(by=["status", "suspension_ratio"], ascending=[False, False])
            
            save_csv(df_report, self.report_path)
            
            logger.info("-" * 40)
            logger.info(f"清洗完成！")
            logger.info(f"  - 总处理: {len(files)}")
            logger.info(f"  - 有效保留 (OK): {len(files) - rejected_count}")
            logger.info(f"  - 剔除 (REJECT): {rejected_count}")
            logger.info(f"  - 质量报告位置: {self.report_path}")
            
            if rejected_count > 0:
                logger.info("  - 剔除原因分布:")
                print(df_report[df_report["status"]=="REJECT"]["reason"].value_counts())
            logger.info("-" * 40)

if __name__ == "__main__":
    cleaner = DataCleaner()
    cleaner.run()