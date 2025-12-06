# scripts/download_data.py

import os
import sys
import argparse
import pandas as pd
from tqdm import tqdm

# 路径适配
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data_source.datahub import DataHub
from src.utils.config import GLOBAL_CONFIG
from src.utils.io import save_parquet, ensure_dir, read_parquet
from src.utils.logger import get_logger

logger = get_logger()

def parse_args():
    parser = argparse.ArgumentParser(description="批量下载 A 股行情数据")
    parser.add_argument("--force", "-f", action="store_true", help="强制重新下载 (覆盖模式)")
    return parser.parse_args()

def filter_stocks(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    核心筛选逻辑：读取 main.yaml 配置，对股票进行过滤
    """
    pool_cfg = GLOBAL_CONFIG["data"]["stock_pool"]
    
    logger.info("=== 开始根据配置筛选股票 ===")
    
    df = df_all.copy()
    initial_count = len(df)
    
    # 1. 过滤 ST / 退市 (依赖 name 字段)
    if pool_cfg.get("exclude_st", True):
        df = df[~df["name"].str.contains("ST", na=False)]
        df = df[~df["name"].str.contains("退", na=False)]
        
    # 2. 过滤板块 (依赖 symbol 字段)
    if not pool_cfg.get("include_kcb", False): # 科创板 688
        df = df[~df["symbol"].str.startswith("688")]
        
    if not pool_cfg.get("include_cyb", False): # 创业板 300
        df = df[~df["symbol"].str.startswith("300")]
        
    if not pool_cfg.get("include_bj", False):  # 北交所 8xx, 4xx, 92x
        df = df[~df["symbol"].str.match(r"^(8|4|92)")]
    
    filtered_count = len(df)
    drop_count = initial_count - filtered_count
    logger.info(f"筛选结果: 原始 {initial_count} -> 剩余 {filtered_count} (剔除 {drop_count})")
    
    return df

def download_stocks(datahub: DataHub, force_reload: bool = False):
    """批量下载流程"""
    
    # 1. 读取本地全量名单
    meta_dir = GLOBAL_CONFIG["paths"]["data_meta"]
    meta_path = os.path.join(meta_dir, "all_stocks_meta.parquet")
    
    if not os.path.exists(meta_path):
        logger.error(f"未找到股票名单文件: {meta_path}")
        logger.error("请先运行: python scripts/init_stock_pool.py")
        return

    df_meta = read_parquet(meta_path)

    # 2. 执行筛选
    df_target = filter_stocks(df_meta)
    
    if df_target.empty:
        logger.error("筛选后列表为空，停止下载。")
        return

    stocks = df_target["symbol"].tolist()
    
    # 3. 准备下载
    raw_dir = GLOBAL_CONFIG["paths"]["data_raw"]
    ensure_dir(raw_dir)

    mode_str = "【全量覆盖】" if force_reload else "【断点续传】"
    logger.info(f"准备下载 {len(stocks)} 只股票. 模式: {mode_str}")
    
    success_count = 0
    fail_count = 0
    skipped_count = 0
    
    pbar = tqdm(stocks, desc="Downloading", unit="stock")
    
    for symbol in pbar:
        try:
            save_path = os.path.join(raw_dir, f"{symbol}.parquet")

            # 断点续传：如果文件存在且不是强制模式，直接跳过
            if not force_reload and os.path.exists(save_path):
                skipped_count += 1
                pbar.set_postfix({"Skip": skipped_count, "OK": success_count})
                continue

            # 下载 (DataHub 内部调用 Baostock)
            df = datahub.fetch_price(symbol)
            
            if not df.empty:
                save_parquet(df, save_path)
                success_count += 1
            else:
                fail_count += 1
                
            pbar.set_postfix({"Skip": skipped_count, "OK": success_count, "Fail": fail_count})

        except KeyboardInterrupt:
            logger.warning("\n用户强制中断！")
            break
        except Exception as e:
            logger.error(f"下载 {symbol} 出错: {e}")
            fail_count += 1

    logger.info(f"下载结束. 成功: {success_count}, 跳过: {skipped_count}, 失败: {fail_count}")

def download_indices(datahub: DataHub, force_reload: bool = False):
    """下载指数 (保持不变)"""
    index_code = GLOBAL_CONFIG.get("preprocessing", {}).get("labels", {}).get("index_code", "000300.SH")
    raw_dir = GLOBAL_CONFIG["paths"]["data_raw"]
    filename = f"index_{index_code.replace('.', '')}.parquet"
    save_path = os.path.join(raw_dir, filename)

    if not force_reload and os.path.exists(save_path):
        logger.info(f"指数 {index_code} 已存在，跳过。")
        return

    logger.info(f"正在下载基准指数: {index_code}")
    df = datahub.fetch_index_price(index_code)
    
    if not df.empty:
        save_parquet(df, save_path)
    else:
        logger.error(f"指数下载失败: {index_code}")

def main():
    args = parse_args()
    datahub = DataHub() # 确保 DataHub 用的是 BaostockSource
    
    download_stocks(datahub, force_reload=args.force)
    download_indices(datahub, force_reload=args.force)

if __name__ == "__main__":
    main()