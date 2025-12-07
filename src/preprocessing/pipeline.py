# src/preprocessing/pipeline.py

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.utils.config import GLOBAL_CONFIG
from src.utils.logger import get_logger
from src.utils.io import save_parquet, ensure_dir, read_parquet
from src.data_source.datahub import DataHub
from src.preprocessing.features import FeatureGenerator
from src.preprocessing.labels import LabelGenerator

logger = get_logger()

class PreprocessPipeline:
    def __init__(self):
        self.config = GLOBAL_CONFIG
        self.datahub = DataHub()
        
        # 初始化组件
        self.feature_eng = FeatureGenerator(self.config)
        self.label_gen = LabelGenerator(self.config)
        
        # 路径
        self.output_dir = self.config["paths"]["data_processed"]
        ensure_dir(self.output_dir)
        
        # 批处理配置
        self.batch_cfg = self.config.get("preprocessing", {}).get("batch", {})
        
        # 读取过滤配置
        self.filter_cfg = self.config.get("preprocessing", {}).get("filter", {})

    def _load_meta_data(self):
        """加载元数据用于 ST 过滤"""
        meta_path = os.path.join(self.config["paths"]["data_meta"], "all_stocks_meta.parquet")
        if os.path.exists(meta_path):
            return read_parquet(meta_path)[["symbol", "name"]]
        return pd.DataFrame(columns=["symbol", "name"])

    def _apply_strict_filter(self, df: pd.DataFrame) -> pd.DataFrame:
            """
            [新增] 严格过滤逻辑 (修复 SettingWithCopyWarning)
            确保训练集只包含符合风控标准的样本 (Row-level Filtering)
            """
            if df.empty: return df
            
            initial_rows = len(df)
            
            # 1. 价格过滤
            min_price = self.filter_cfg.get("min_price", 0.0)
            max_price = self.filter_cfg.get("max_price", 99999.0)
            if "close" in df.columns:
                # [Fix] 增加 .copy() 断开切片关联
                df = df[(df["close"] >= min_price) & (df["close"] <= max_price)].copy()

            # 2. 换手率过滤 (Turnover Rate %)
            min_turnover_rate = self.filter_cfg.get("min_turnover_rate", 0.0)
            if "turnover" in df.columns and min_turnover_rate > 0:
                df["turnover"] = df["turnover"].fillna(0)
                # [Fix] 增加 .copy()
                df = df[df["turnover"] >= min_turnover_rate].copy()

            # 3. 成交额过滤 (Amount)
            # 兼容旧配置 min_turnover 作为 amount
            min_amount = self.filter_cfg.get("min_turnover", 0) 
            if "amount" in df.columns and min_amount > 0:
                df["amount"] = df["amount"].fillna(0)
                # [Fix] 增加 .copy()
                df = df[df["amount"] >= min_amount].copy()

            # 4. 市值过滤 (动态计算)
            min_mcap = self.filter_cfg.get("min_mcap", 0)
            max_mcap = self.filter_cfg.get("max_mcap", float("inf"))
            
            if (min_mcap > 0 or max_mcap < float("inf")) and "amount" in df.columns and "turnover" in df.columns:
                # 市值 = 成交额 / (换手率 * 0.01)
                valid_mask = df["turnover"] > 0.001
                # 使用临时 Series 进行筛选，不直接修改 df
                est_mcap = df.loc[valid_mask, "amount"] / (df.loc[valid_mask, "turnover"] * 0.01)
                
                s_mcap = pd.Series(index=df.index, dtype=float)
                s_mcap.loc[valid_mask] = est_mcap
                
                if min_mcap > 0:
                    df = df[s_mcap >= min_mcap].copy() # [Fix] 增加 .copy()
                    s_mcap = s_mcap[s_mcap >= min_mcap] # 对齐索引
                    
                if max_mcap < float("inf"):
                    df = df[s_mcap <= max_mcap].copy() # [Fix] 增加 .copy()

            # 5. 板块过滤 (根据代码前缀)
            pool_cfg = self.config["data"]["stock_pool"]
            if not pool_cfg.get("include_kcb", False):
                df = df[~df["symbol"].str.startswith("688")].copy()
            if not pool_cfg.get("include_cyb", False):
                df = df[~df["symbol"].str.startswith("300")].copy()
            if not pool_cfg.get("include_bj", False):
                df = df[~df["symbol"].str.match(r"^(8|4|92)")].copy()

            # 6. ST 过滤 (静态)
            if self.filter_cfg.get("exclude_st", True):
                df_meta = self._load_meta_data()
                if not df_meta.empty:
                    st_symbols = df_meta[df_meta["name"].str.contains("ST|退", na=False)]["symbol"].tolist()
                    if st_symbols:
                        df = df[~df["symbol"].isin(st_symbols)].copy()

            return df
    
    
    def run(self):
        logger.info("=== 开始执行特征工程流水线 (含严格前置过滤) ===")
        logger.info(f"过滤配置: {self.filter_cfg}")
        
        # 1. 获取任务列表
        stock_list = self.datahub.get_cleaned_stock_list()
        if not stock_list:
            logger.error("未找到清洗后的股票数据，请先运行 clean_and_check.py")
            return
            
        logger.info(f"扫描到清洗后股票: {len(stock_list)} 只")
        
        processed_list = []
        
        # 2. 循环处理
        for symbol in tqdm(stock_list, desc="Feature Engineering"):
            try:
                # A. 读取
                df = self.datahub.load_cleaned_price(symbol)
                if df is None or df.empty:
                    continue
                
                # 补充 Symbol 列
                df["symbol"] = symbol
                
                # --- [新增] B. 尽早执行过滤 ---
                # 在计算特征之前就过滤，可以节省大量算力
                # 尤其是 Row-level 的过滤 (如某天成交额不足)，必须现在做
                df = self._apply_strict_filter(df)
                
                if df.empty:
                    continue

                # C. 计算特征
                df = self.feature_eng.run(df)
                
                # D. 生成标签
                df = self.label_gen.run(df)
                
                # E. 清洗 NaN [修改版]
                # 1. 识别特征列 (以 feat_ 开头) 和 基础列
                # 我们只剔除那些连特征都算不出来的行 (比如刚上市前几天无法算MA60)
                check_cols = [c for c in df.columns if c.startswith("feat_") or c in ["close", "volume"]]
                
                # 2. 仅对特征列进行 dropna，保留 Label 为 NaN 的行 (用于预测)
                df_clean = df.dropna(subset=check_cols).reset_index(drop=True)
                
                if df_clean.empty:
                    continue
                
                # F. 保存单文件
                if self.batch_cfg.get("save_each", True):
                    save_path = os.path.join(self.output_dir, f"{symbol}.parquet")
                    save_parquet(df_clean, save_path)
                
                # 加入合并列表
                if self.batch_cfg.get("concat_all", True):
                    processed_list.append(df_clean)
                    
            except Exception as e:
                logger.error(f"处理 {symbol} 失败: {e}")
                
        # 3. 合并保存大文件
        if self.batch_cfg.get("concat_all", True) and processed_list:
            logger.info("正在合并全量特征矩阵...")
            full_df = pd.concat(processed_list, ignore_index=True)
            
            # 排序
            if "date" in full_df.columns:
                full_df = full_df.sort_values(by=["date", "symbol"])
            
            concat_file = self.batch_cfg.get("concat_file", "all_stocks.parquet")
            out_path = os.path.join(self.output_dir, concat_file)
            save_parquet(full_df, out_path)
            
            logger.info(f"全量特征文件已保存: {out_path}")
            logger.info(f"最终样本量: {full_df.shape[0]} 行 (已剔除不合规样本)")
            
        logger.info("特征工程流水线执行完毕。")