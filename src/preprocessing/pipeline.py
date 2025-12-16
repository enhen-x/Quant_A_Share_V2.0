# src/preprocessing/pipeline.py

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.utils.config import GLOBAL_CONFIG
from src.utils.logger import get_logger
# [修改点 1] 引入 save_csv
from src.utils.io import save_parquet, ensure_dir, read_parquet, save_csv
from src.data_source.datahub import DataHub
from src.preprocessing.features import FeatureGenerator
from src.preprocessing.labels import LabelGenerator
from src.preprocessing.neutralization import FeatureNeutralizer  # [新增引用]

logger = get_logger()

class PreprocessPipeline:
    def __init__(self):
        self.config = GLOBAL_CONFIG
        self.datahub = DataHub()
        
        # 初始化组件
        self.feature_eng = FeatureGenerator(self.config)
        self.label_gen = LabelGenerator(self.config)
        self.neutralizer = FeatureNeutralizer(self.config)  # [新增初始化]
        
        # 路径
        self.output_dir = self.config["paths"]["data_processed"]
        ensure_dir(self.output_dir)
        
        # [修改点 2] 定义统计报告保存路径
        self.report_path = os.path.join(self.output_dir, "data_filter_summary.csv")
        
        # 批处理配置
        self.batch_cfg = self.config.get("preprocessing", {}).get("batch", {})
        
        # 读取过滤配置
        self.filter_cfg = self.config.get("preprocessing", {}).get("filter", {})

        # === 过滤统计计数器 ===
        self.filter_stats = {
            "total_rows_input": 0,    # 初始总行数
            "total_rows_output": 0,   # 最终保留行数
            # 各环节丢弃计数
            "dropped_by_price": 0,          # 价格限制
            "dropped_by_turnover_rate": 0,  # 换手率
            "dropped_by_amount": 0,         # 成交额
            "dropped_by_mcap": 0,           # 市值
            "dropped_by_sector": 0,         # 板块 (科创/创业/北交)
            "dropped_by_st": 0,             # ST 股
            "dropped_by_nan": 0             # 最终计算特征后的 NaN
        }

    def _load_meta_data(self):
        """加载元数据用于 ST 过滤"""
        meta_path = os.path.join(self.config["paths"]["data_meta"], "all_stocks_meta.parquet")
        if os.path.exists(meta_path):
            return read_parquet(meta_path)[["symbol", "name"]]
        return pd.DataFrame(columns=["symbol", "name"])

    def _apply_strict_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        严格过滤逻辑 (Row-level Filtering) + [统计功能]
        """
        if df.empty: return df
        
        # 记录该股票初始行数
        initial_rows = len(df)
        self.filter_stats["total_rows_input"] += initial_rows
        
        current_df = df
        
        # --- 1. 价格过滤 (min_price, max_price) ---
        min_price = self.filter_cfg.get("min_price", 0.0)
        max_price = self.filter_cfg.get("max_price", 99999.0)
        if "close" in current_df.columns:
            prev_len = len(current_df)
            current_df = current_df[(current_df["close"] >= min_price) & (current_df["close"] <= max_price)].copy()
            self.filter_stats["dropped_by_price"] += (prev_len - len(current_df))

        if current_df.empty: return current_df

        # --- 2. 换手率过滤 (Turnover Rate %) ---
        min_turnover_rate = self.filter_cfg.get("min_turnover_rate", 0.0)
        if "turnover" in current_df.columns and min_turnover_rate > 0:
            current_df["turnover"] = current_df["turnover"].fillna(0)
            prev_len = len(current_df)
            current_df = current_df[current_df["turnover"] >= min_turnover_rate].copy()
            self.filter_stats["dropped_by_turnover_rate"] += (prev_len - len(current_df))

        if current_df.empty: return current_df

        # --- 3. 成交额过滤 (Amount) ---
        min_amount = self.filter_cfg.get("min_turnover", 0) 
        if "amount" in current_df.columns and min_amount > 0:
            current_df["amount"] = current_df["amount"].fillna(0)
            prev_len = len(current_df)
            current_df = current_df[current_df["amount"] >= min_amount].copy()
            self.filter_stats["dropped_by_amount"] += (prev_len - len(current_df))

        if current_df.empty: return current_df

        # --- 4. 市值过滤 (动态计算) ---
        min_mcap = self.filter_cfg.get("min_mcap", 0)
        max_mcap = self.filter_cfg.get("max_mcap", float("inf"))
        
        if (min_mcap > 0 or max_mcap < float("inf")) and "amount" in current_df.columns and "turnover" in current_df.columns:
            prev_len = len(current_df)
            valid_mask = current_df["turnover"] > 0.001
            
            est_mcap = pd.Series(np.nan, index=current_df.index)
            est_mcap.loc[valid_mask] = current_df.loc[valid_mask, "amount"] / (current_df.loc[valid_mask, "turnover"] * 0.01)
            
            keep_mask = pd.Series(True, index=current_df.index)
            if min_mcap > 0:
                has_mcap_but_small = (est_mcap < min_mcap)
                keep_mask = keep_mask & (~has_mcap_but_small)
            if max_mcap < float("inf"):
                has_mcap_but_large = (est_mcap > max_mcap)
                keep_mask = keep_mask & (~has_mcap_but_large)

            current_df = current_df[keep_mask].copy()
            self.filter_stats["dropped_by_mcap"] += (prev_len - len(current_df))

        if current_df.empty: return current_df

        # --- 5. 板块过滤 ---
        pool_cfg = self.config["data"]["stock_pool"]
        prev_len = len(current_df)
        if not pool_cfg.get("include_kcb", False):
            current_df = current_df[~current_df["symbol"].str.startswith("688")].copy()
        if not pool_cfg.get("include_cyb", False):
            current_df = current_df[~current_df["symbol"].str.startswith("300")].copy()
        if not pool_cfg.get("include_bj", False):
            current_df = current_df[~current_df["symbol"].str.match(r"^(8|4|92)")].copy() 
        self.filter_stats["dropped_by_sector"] += (prev_len - len(current_df))

        if current_df.empty: return current_df

        # --- 6. ST 过滤 ---
        if self.filter_cfg.get("exclude_st", True):
            prev_len = len(current_df)
            df_meta = self._load_meta_data()
            if not df_meta.empty:
                st_symbols = set(df_meta[df_meta["name"].str.contains("ST|退", na=False)]["symbol"])
                if st_symbols:
                    current_df = current_df[~current_df["symbol"].isin(st_symbols)].copy()
            self.filter_stats["dropped_by_st"] += (prev_len - len(current_df))

        return current_df
    
    def run(self):
        logger.info("=== 开始执行特征工程流水线 (含严格前置过滤) ===")
        
        stock_list = self.datahub.get_cleaned_stock_list()
        if not stock_list:
            logger.error("未找到清洗后的股票数据")
            return
            
        logger.info(f"扫描到清洗后股票: {len(stock_list)} 只")
        processed_list = []
        
        for symbol in tqdm(stock_list, desc="Feature Engineering"):
            try:
                # A. 读取
                df = self.datahub.load_cleaned_price(symbol)
                if df is None or df.empty:
                    continue
                df["symbol"] = symbol
                
                # B. 严格过滤
                df = self._apply_strict_filter(df)
                if df.empty: continue

                # C. 计算特征
                df = self.feature_eng.run(df)
                
                # D. 生成标签
                df = self.label_gen.run(df)
                
                # E. 清洗 NaN
                check_cols = [c for c in df.columns if c.startswith("feat_") or c in ["close", "volume"]]
                prev_len = len(df)
                df_clean = df.dropna(subset=check_cols).reset_index(drop=True)
                
                self.filter_stats["dropped_by_nan"] += (prev_len - len(df_clean))
                
                if df_clean.empty: continue
                
                # F. 保存单文件
                if self.batch_cfg.get("save_each", True):
                    save_path = os.path.join(self.output_dir, f"{symbol}.parquet")
                    save_parquet(df_clean, save_path)
                
                if self.batch_cfg.get("concat_all", True):
                    processed_list.append(df_clean)
                    self.filter_stats["total_rows_output"] += len(df_clean)
                    
            except Exception as e:
                logger.error(f"处理 {symbol} 失败: {e}")
                
        # 合并保存大文件
        if self.batch_cfg.get("concat_all", True) and processed_list:
            logger.info("正在合并全量特征矩阵...")
            full_df = pd.concat(processed_list, ignore_index=True)
            if "date" in full_df.columns:
                full_df = full_df.sort_values(by=["date", "symbol"])
            
            # 在保存 all_stocks.parquet 之前执行中性化
            full_df = self.neutralizer.run(full_df)
            concat_file = self.batch_cfg.get("concat_file", "all_stocks.parquet")
            out_path = os.path.join(self.output_dir, concat_file)
            save_parquet(full_df, out_path)
            logger.info(f"全量特征文件已保存: {out_path}")
            
        # [修改点 3] 生成并保存报告
        self._save_filter_report()
        logger.info("特征工程流水线执行完毕。")

    def _save_filter_report(self):
        """保存并打印数据过滤统计报告"""
        total_in = self.filter_stats["total_rows_input"]
        total_out = self.filter_stats["total_rows_output"]
        
        if total_in == 0:
            logger.warning("输入数据量为 0，无法生成报告。")
            return

        dropped_total = total_in - total_out
        
        # 1. 构造报告数据 List
        report_data = []
        
        # (1) 总体统计
        report_data.append({
            "Category": "SUMMARY", 
            "Item": "Total Input Rows", 
            "Count": total_in, 
            "Ratio (%)": 100.0
        })
        report_data.append({
            "Category": "SUMMARY", 
            "Item": "Final Output Rows", 
            "Count": total_out, 
            "Ratio (%)": round((total_out / total_in) * 100, 2)
        })
        report_data.append({
            "Category": "SUMMARY", 
            "Item": "Total Dropped", 
            "Count": dropped_total, 
            "Ratio (%)": round((dropped_total / total_in) * 100, 2)
        })

        # (2) 细分原因统计
        drop_reasons = {k: v for k, v in self.filter_stats.items() if k.startswith("dropped_")}
        sorted_reasons = sorted(drop_reasons.items(), key=lambda item: item[1], reverse=True)
        
        for reason, count in sorted_reasons:
            if count > 0:
                report_data.append({
                    "Category": "FILTER_DETAIL",
                    "Item": reason.replace("dropped_by_", "").upper(),
                    "Count": count,
                    "Ratio (%)": round((count / total_in) * 100, 2)
                })

        # 2. 转换为 DataFrame 并保存 CSV
        df_report = pd.DataFrame(report_data)
        save_csv(df_report, self.report_path)
        
        # 3. 同时也打印到控制台，方便查看
        logger.info("=" * 50)
        logger.info(f"📊 过滤统计报告已保存: {self.report_path}")
        logger.info("-" * 50)
        # 简单打印最重要的几行
        print(df_report[["Item", "Count", "Ratio (%)"]].to_string(index=False))
        
        if sorted_reasons:
            max_reason = sorted_reasons[0][0].replace("dropped_by_", "").upper()
            logger.info("-" * 50)
            logger.info(f"🚫 样本削减最大因素: 【{max_reason}】")
        logger.info("=" * 50)