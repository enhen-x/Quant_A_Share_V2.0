"""
临时脚本：为现有数据集添加缺失的特征
用于快速修复特征不匹配问题
"""
import sys
import os
import pandas as pd
import numpy as np

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.io import read_parquet, save_parquet
from src.utils.logger import get_logger
from src.preprocessing.features import FeatureGenerator

logger = get_logger()

def add_missing_features():
    """为现有数据集添加缺失的 feat_macd_dif 和 feat_kdj_d"""
    
    data_path = "data/processed/all_stocks.parquet"
    
    if not os.path.exists(data_path):
        logger.error(f"数据文件不存在: {data_path}")
        return False
    
    logger.info(f"加载数据: {data_path}")
    df = read_parquet(data_path)
    
    logger.info(f"数据形状: {df.shape}")
    logger.info(f"现有特征列: {[c for c in df.columns if c.startswith('feat_')]}")
    
    # 检查是否缺少特征
    missing_features = []
    if 'feat_macd_dif' not in df.columns:
        missing_features.append('feat_macd_dif')
    if 'feat_kdj_d' not in df.columns:
        missing_features.append('feat_kdj_d')
    
    if not missing_features:
        logger.info("所有特征都存在，无需添加")
        return True
    
    logger.info(f"缺失特征: {missing_features}")
    logger.info("正在计算缺失特征...")
    
    # 按股票分组处理
    def calculate_missing_features(group):
        # MACD DIF
        if 'feat_macd_dif' in missing_features:
            ema_12 = group['close'].ewm(span=12, adjust=False).mean()
            ema_26 = group['close'].ewm(span=26, adjust=False).mean()
            dif = ema_12 - ema_26
            group['feat_macd_dif'] = dif / group['close']
        
        # KDJ D
        if 'feat_kdj_d' in missing_features:
            if 'feat_kdj_k' in group.columns:
                group['feat_kdj_d'] = group['feat_kdj_k'].ewm(alpha=1/3, adjust=False).mean()
                # 更新 feat_kdj_j（因为它依赖 kdj_d）
                group['feat_kdj_j'] = 3 * group['feat_kdj_k'] - 2 * group['feat_kdj_d']
        
        return group
    
    logger.info("按股票计算特征...")
    df = df.groupby('symbol', group_keys=False).apply(calculate_missing_features)
    
    # 保存更新后的数据
    logger.info(f"保存更新后的数据到: {data_path}")
    save_parquet(df, data_path)
    
    logger.info(f"✅ 成功添加缺失特征: {missing_features}")
    logger.info(f"更新后的特征列: {[c for c in df.columns if c.startswith('feat_')]}")
    
    return True

if __name__ == "__main__":
    success = add_missing_features()
    sys.exit(0 if success else 1)
