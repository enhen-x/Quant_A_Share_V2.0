# src/preprocessing/pipeline.py

import os
import pandas as pd
from tqdm import tqdm
from src.utils.config import GLOBAL_CONFIG
from src.utils.logger import get_logger
from src.utils.io import save_parquet, ensure_dir
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

    def run(self):
        logger.info("=== 开始执行特征工程流水线 ===")
        
        # 1. 获取任务列表 (清洗过的股票)
        stock_list = self.datahub.get_cleaned_stock_list()
        if not stock_list:
            logger.error("未找到清洗后的股票数据，请先运行 clean_and_check.py")
            return
            
        logger.info(f"待处理股票数量: {len(stock_list)}")
        
        processed_list = []
        
        # 2. 循环处理
        for symbol in tqdm(stock_list, desc="Feature Engineering"):
            try:
                # A. 读取
                df = self.datahub.load_cleaned_price(symbol)
                if df is None or df.empty:
                    continue
                    
                # B. 计算特征
                df = self.feature_eng.run(df)
                
                # C. 生成标签
                df = self.label_gen.run(df)
                
                # D. 清洗 NaN (特征计算产生的 NaN，如 MA 的前几日；标签产生的 NaN，如最后几日)
                # 注意：在这里 dropna 会导致最后 horizon 天的数据丢失，这是正常的，
                # 但如果是用于"预测" (inference)，则不能 drop 最后几行。
                # 为了简化，这里生成的是"训练集"，所以我们 dropna。
                # *改进*：我们可以保留 NaN 供预测使用，但在 save 时区分。
                # 这里暂且直接 Drop，保证训练数据的纯净性。
                
                # 仅 Drop 特征和 Label 均为空的行
                # 实际操作：Drop 包含 NaN 的行（严格模式）
                df_clean = df.dropna().reset_index(drop=True)
                
                if df_clean.empty:
                    continue
                
                # 补充 Symbol 列 (方便合并后识别)
                df_clean["symbol"] = symbol
                
                # E. 保存单文件 (可选)
                if self.batch_cfg.get("save_each", True):
                    save_path = os.path.join(self.output_dir, f"{symbol}.parquet")
                    save_parquet(df_clean, save_path)
                
                # 加入合并列表 (为了内存考虑，如果数据量太大，这里可能需要优化，比如分批 Merge)
                if self.batch_cfg.get("concat_all", True):
                    processed_list.append(df_clean)
                    
            except Exception as e:
                logger.error(f"处理 {symbol} 失败: {e}")
                
        # 3. 合并保存大文件 (All Stocks)
        if self.batch_cfg.get("concat_all", True) and processed_list:
            logger.info("正在合并全量特征矩阵...")
            full_df = pd.concat(processed_list, ignore_index=True)
            
            # 按日期和代码排序
            if "date" in full_df.columns:
                full_df = full_df.sort_values(by=["date", "symbol"])
            
            concat_file = self.batch_cfg.get("concat_file", "all_stocks.parquet")
            out_path = os.path.join(self.output_dir, concat_file)
            save_parquet(full_df, out_path)
            
            logger.info(f"全量特征文件已保存: {out_path}")
            logger.info(f"形状: {full_df.shape}")
            
        logger.info("特征工程流水线执行完毕。")