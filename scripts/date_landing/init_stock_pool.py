# scripts/init_stock_pool.py

import os
import sys
# 路径适配
current_dir = os.path.dirname(os.path.abspath(__file__))
# 从当前文件位置 (scripts/analisis) 返回两级到项目根目录
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import akshare as ak
from src.utils.io import save_parquet, ensure_dir
from src.utils.logger import get_logger
from src.utils.config import GLOBAL_CONFIG

logger = get_logger()

def main():
    logger.info("=== 步骤 1/2: 获取全市场股票名单 ===")
    try:
        # 1. 获取最基础的名单 (仅 code, name)
        df = ak.stock_info_a_code_name()
        
        if df is None or df.empty:
            logger.error("股票名单接口返回为空，请检查网络。")
            return

        # 2. 标准化列名
        df = df.rename(columns={"code": "symbol", "name": "name"})
        df = df[["symbol", "name"]]
        
        # 3. 保存股票列表
        meta_dir = GLOBAL_CONFIG["paths"]["data_meta"]
        ensure_dir(meta_dir)
        
        save_path = os.path.join(meta_dir, "all_stocks_meta.parquet")
        save_parquet(df, save_path)
        
        logger.info(f"全量名单已保存: {save_path}")
        logger.info(f"共包含 {len(df)} 只股票。")

    except Exception as e:
        logger.error(f"获取股票名单失败: {e}")
        return

    # ==========================================
    # 新增部分：获取并保存交易日历
    # ==========================================
    logger.info("=== 步骤 2/2: 获取全历史交易日历 ===")
    try:
        # 获取新浪的交易日历数据（包含历史和未来一年的安排）
        df_cal = ak.tool_trade_date_hist_sina()
        
        if df_cal is None or df_cal.empty:
            logger.error("交易日历接口返回为空。")
        else:
            # 标准化
            df_cal = df_cal.rename(columns={"trade_date": "date"})
            df_cal["date"] = pd.to_datetime(df_cal["date"])
            
            # 保存
            cal_path = os.path.join(meta_dir, "trade_calendar.parquet")
            save_parquet(df_cal, cal_path)
            logger.info(f"交易日历已保存: {cal_path}")
            
    except Exception as e:
        logger.error(f"获取交易日历失败: {e}")

    logger.info("初始化完成。")

if __name__ == "__main__":
    main()