# scripts/init_stock_pool.py

import os
import sys
# 路径适配
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import pandas as pd
import akshare as ak
from src.utils.io import save_parquet, ensure_dir
from src.utils.logger import get_logger
from src.utils.config import GLOBAL_CONFIG



logger = get_logger()

def main():
    logger.info("正在获取全市场股票名单 (ak.stock_info_a_code_name)...")
    
    try:
        # 1. 获取最基础的名单 (仅 code, name)
        df = ak.stock_info_a_code_name()
        
        if df is None or df.empty:
            logger.error("接口返回为空，请检查网络。")
            return

        # 2. 标准化列名
        df = df.rename(columns={"code": "symbol", "name": "name"})
        
        # 3. 只保留这两列，其他都不管
        df = df[["symbol", "name"]]
        
        # 4. 保存
        meta_dir = GLOBAL_CONFIG["paths"]["data_meta"]
        save_path = os.path.join(meta_dir, "all_stocks_meta.parquet")
        
        ensure_dir(meta_dir)
        save_parquet(df, save_path)
        
        logger.info(f"全量名单已保存: {save_path}")
        logger.info(f"共包含 {len(df)} 只股票。")
        logger.info("过滤工作将在 download_data.py 中进行。")

    except Exception as e:
        logger.error(f"获取失败: {e}")

if __name__ == "__main__":
    main()