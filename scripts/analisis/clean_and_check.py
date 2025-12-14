# scripts/clean_and_check.py

import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm

# 路径适配
current_dir = os.path.dirname(os.path.abspath(__file__))
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
        # 详细报告（每只股票一行）
        self.detail_report_path = os.path.join(self.cleaned_dir, "data_quality_report.csv")
        # [新增] 汇总报告（全局统计）
        self.summary_report_path = os.path.join(self.cleaned_dir, "data_cleaning_summary.csv")
        
        ensure_dir(self.cleaned_dir)
        
        # === 1. 获取清洗/筛选阈值 ===
        quality_cfg = GLOBAL_CONFIG.get("preprocessing", {}).get("quality", {})
        
        self.limit_suspension = quality_cfg.get("max_suspension_rate", 0.1) 
        self.limit_turnover = quality_cfg.get("min_avg_turnover", 1.0)
        
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
        处理单只股票
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
            "status": "OK",             
            "reason": "",               
            "total_rows": initial_count,
            "start_date": df["date"].min() if not df.empty else None,
            "end_date": df["date"].max() if not df.empty else None,
            "n_duplicates": 0,
            "n_zero_price": 0,
            "n_suspension": 0,
            "suspension_ratio": 0.0,
            "avg_turnover": 0.0,
            "n_missing_days": 0,
            "clean_rows": 0   # 行级清洗后的行数（不论该股票最终是否被 Reject）
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
        price_cols = ["open", "high", "low", "close", "volume"]
        valid_cols = [c for c in price_cols if c in df.columns]
        
        # 剔除 NaN
        nan_mask = df[valid_cols].isnull().any(axis=1)
        df = df[~nan_mask]
        
        # 剔除 0 价格
        if "close" in df.columns:
            zero_price_mask = (df["close"] <= 1e-4)
            stats["n_zero_price"] = zero_price_mask.sum()
            df = df[~zero_price_mask]

        if df.empty:
            stats["status"] = "REJECT_EMPTY_AFTER_CLEAN"
            return stats

        stats["clean_rows"] = len(df)

        # ==========================
        # Step B: 指标计算 (Metrics)
        # ==========================
        # 1. 停牌统计
        if "volume" in df.columns:
            suspension_mask = (df["volume"] < 1e-6)
            stats["n_suspension"] = suspension_mask.sum()
            stats["suspension_ratio"] = stats["n_suspension"] / len(df)
        
        # 2. 换手率统计
        if "turnover" in df.columns:
            stats["avg_turnover"] = df["turnover"].mean()
        
        # 3. 日期缺失
        s_date = df["date"].min().date()
        e_date = df["date"].max().date()
        expected_dates = {d for d in self.trade_dates if s_date <= d <= e_date}
        actual_dates = set(df["date"].dt.date)
        missing_dates = expected_dates - actual_dates
        stats["n_missing_days"] = len(missing_dates)
        
        denom = len(expected_dates)
        missing_ratio = len(missing_dates) / denom if denom > 0 else 0.0

        # ==========================
        # Step C: 标的级筛选 (Stock-level Filter)
        # ==========================
        
        # 规则 1: 停牌率过高
        if stats["suspension_ratio"] > self.limit_suspension:
            stats["status"] = "REJECT"
            stats["reason"] = "HIGH_SUSPENSION"
            return stats 

        # 规则 2: 流动性枯竭(数据清洗阶段已注释掉)
        # if "turnover" in df.columns and stats["avg_turnover"] < self.limit_turnover:
        #     stats["status"] = "REJECT"
        #     stats["reason"] = "LOW_LIQUIDITY"
        #     return stats
        
        if "turnover" in df.columns and stats["avg_turnover"] < self.limit_turnover:
             # 仅做记录，不拒绝
             pass

        # 规则 3: 数据严重缺失
        if missing_ratio > 0.5:
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

        files = [f for f in os.listdir(self.raw_dir) if f.endswith(".parquet") and f[0].isdigit()]
        logger.info(f"待处理文件数: {len(files)}")
        
        results = []
        
        for f in tqdm(files, desc="Cleaning"):
            stats = self.check_and_clean_single(os.path.join(self.raw_dir, f))
            if stats:
                results.append(stats)
            
        # 生成详细报告 & 汇总报告
        if results:
            # 1. 保存详细报告
            df_report = pd.DataFrame(results)
            df_report = df_report.sort_values(by=["status", "suspension_ratio"], ascending=[False, False])
            save_csv(df_report, self.detail_report_path)
            
            # 2. 生成并保存汇总报告
            self._save_summary_report(results)
            
            # 3. 简单日志
            rejected_count = len(df_report[df_report["status"] != "OK"])
            logger.info("-" * 40)
            logger.info(f"清洗完成！")
            logger.info(f"  - 总处理股票: {len(files)}")
            logger.info(f"  - 有效保留 (OK): {len(files) - rejected_count}")
            logger.info(f"  - 剔除股票 (REJECT): {rejected_count}")
            logger.info(f"  - 详细报告位置: {self.detail_report_path}")
            logger.info("-" * 40)

    def _save_summary_report(self, results: list):
        """
        计算全局行级损失并生成报告
        """
        # 1. 基础聚合
        # 输入总行数
        total_input_rows = sum(r.get("total_rows", 0) for r in results)
        
        # 最终输出行数 (仅统计 Status=OK 的 Clean Rows)
        total_output_rows = sum(r.get("clean_rows", 0) for r in results if r["status"] == "OK")
        
        # 2. 计算各环节丢弃的行数
        
        # A. 行级清洗丢弃 (Duplicates / Zero Price)
        # 这些是在所有股票中都会发生的，不论该股票最后是否被剔除
        dropped_by_duplicates = sum(r.get("n_duplicates", 0) for r in results)
        dropped_by_zero_price = sum(r.get("n_zero_price", 0) for r in results)
        
        # B. 标的级剔除造成的损失 (Stock Rejection)
        # 如果一只股票被剔除，它剩下的所有行 (clean_rows) 都被视为损失
        dropped_by_suspension = sum(r["clean_rows"] for r in results if r["reason"] == "HIGH_SUSPENSION")
        dropped_by_liquidity = sum(r["clean_rows"] for r in results if r["reason"] == "LOW_LIQUIDITY")
        dropped_by_missing = sum(r["clean_rows"] for r in results if r["reason"] == "HIGH_MISSING")
        dropped_by_other = sum(r["clean_rows"] for r in results if r["status"] != "OK" 
                               and r["reason"] not in ["HIGH_SUSPENSION", "LOW_LIQUIDITY", "HIGH_MISSING"])

        dropped_total = total_input_rows - total_output_rows
        
        # 3. 构造报告列表
        report_data = []
        
        # 总体
        report_data.append({
            "Category": "SUMMARY", "Item": "Total Input Rows", 
            "Count": total_input_rows, "Ratio (%)": 100.0
        })
        report_data.append({
            "Category": "SUMMARY", "Item": "Final Output Rows", 
            "Count": total_output_rows, 
            "Ratio (%)": round((total_output_rows / total_input_rows * 100), 2) if total_input_rows else 0
        })
        report_data.append({
            "Category": "SUMMARY", "Item": "Total Dropped", 
            "Count": dropped_total, 
            "Ratio (%)": round((dropped_total / total_input_rows * 100), 2) if total_input_rows else 0
        })
        
        # 细节
        details = [
            ("ROW_CLEANING", "Dropped (Duplicates)", dropped_by_duplicates),
            ("ROW_CLEANING", "Dropped (Zero/NaN Price)", dropped_by_zero_price),
            ("STOCK_REJECT", "Dropped (High Suspension)", dropped_by_suspension),
            ("STOCK_REJECT", "Dropped (Low Liquidity)", dropped_by_liquidity),
            ("STOCK_REJECT", "Dropped (High Missing Data)", dropped_by_missing),
            ("STOCK_REJECT", "Dropped (Other Reasons)", dropped_by_other),
        ]
        
        # 按丢弃数量排序
        details.sort(key=lambda x: x[2], reverse=True)
        
        for cat, item, count in details:
            if count > 0:
                report_data.append({
                    "Category": cat,
                    "Item": item,
                    "Count": count,
                    "Ratio (%)": round((count / total_input_rows * 100), 2) if total_input_rows else 0
                })
                
        # 4. 保存与打印
        df_summary = pd.DataFrame(report_data)
        save_csv(df_summary, self.summary_report_path)
        
        logger.info("=" * 50)
        logger.info(f"📊 数据清洗统计报告已保存: {self.summary_report_path}")
        logger.info("-" * 50)
        if not df_summary.empty:
            print(df_summary[["Item", "Count", "Ratio (%)"]].to_string(index=False))
            
            # 找出最大杀手
            df_reasons = df_summary[df_summary["Category"].isin(["ROW_CLEANING", "STOCK_REJECT"])]
            if not df_reasons.empty:
                max_row = df_reasons.loc[df_reasons["Count"].idxmax()]
                logger.info("-" * 50)
                logger.info(f"🚫 样本削减最大因素: 【{max_row['Item']}】")
        logger.info("=" * 50)

if __name__ == "__main__":
    cleaner = DataCleaner()
    cleaner.run()