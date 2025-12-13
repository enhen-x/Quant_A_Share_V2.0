# scripts/download_data.py

import os
import sys
import argparse
import pandas as pd
from tqdm import tqdm

# 路径适配
current_dir = os.path.dirname(os.path.abspath(__file__))
# 从当前文件位置 (scripts/analisis) 返回两级到项目根目录
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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
        
    # 2. 板块筛选 (依赖 symbol 字段的前缀)
    # 2.1 科创板 (688)
    if not pool_cfg.get("include_kcb", False): 
        df = df[~df["symbol"].str.startswith("688")]
        
    # 2.2 创业板 (30)
    if not pool_cfg.get("include_cyb", False): 
        df = df[~df["symbol"].str.startswith("30")]
        
    # 2.3 北交所 (8, 4, 92)
    if not pool_cfg.get("include_bj", False):
        df = df[~df["symbol"].str.match(r"^(8|4|92)")]

    # 2.4 沪市主板 (60)
    if not pool_cfg.get("include_sh", True):
        df = df[~df["symbol"].str.startswith("60")]

    # 2.5 深市主板 (00)
    if not pool_cfg.get("include_sz", True):
        df = df[~df["symbol"].str.startswith("00")]
    
    filtered_count = len(df)
    drop_count = initial_count - filtered_count
    logger.info(f"筛选结果: 原始 {initial_count} -> 剩余 {filtered_count} (剔除 {drop_count})")
    
    return df

def download_stocks(datahub: DataHub, force_reload: bool = False):
    """批量下载流程"""
    meta_dir = GLOBAL_CONFIG["paths"]["data_meta"]
    meta_path = os.path.join(meta_dir, "all_stocks_meta.parquet")
    
    if not os.path.exists(meta_path):
        logger.error(f"未找到股票名单文件: {meta_path}，请先运行 init_stock_pool.py")
        return

    df_meta = read_parquet(meta_path)
    df_target = filter_stocks(df_meta)
    
    if df_target.empty:
        logger.error("筛选后列表为空，停止下载。")
        return

    stocks = df_target["symbol"].tolist()
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

            if not force_reload and os.path.exists(save_path):
                skipped_count += 1
                pbar.set_postfix({"Skip": skipped_count, "OK": success_count})
                continue

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
    """下载指数"""
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

# ==============================================================================
# 合并函数
# ==============================================================================
def merge_data_files():
    """
    根据 config/main.yaml 中的 batch 配置，将下载的分散 parquet 合并为一个大文件。
    """
    # 1. 获取配置
    batch_cfg = GLOBAL_CONFIG.get("preprocessing", {}).get("batch", {})
    concat_all = batch_cfg.get("concat_all", False)
    
    if not concat_all:
        logger.info("配置 batch.concat_all 为 False，跳过合并步骤。")
        return

    logger.info("=== 开始执行数据合并任务 ===")
    
    # 2. 准备路径
    raw_dir = GLOBAL_CONFIG["paths"]["data_raw"]
    processed_dir = GLOBAL_CONFIG["paths"]["data_processed"]
    ensure_dir(processed_dir)
    
    filename = batch_cfg.get("concat_file", "all_stocks.parquet")
    output_path = os.path.join(processed_dir, filename)
    
    # 3. 收集文件列表 (过滤掉非股票文件，如指数文件)
    if not os.path.exists(raw_dir):
        logger.warning(f"源目录 {raw_dir} 不存在，无法合并。")
        return

    all_files = os.listdir(raw_dir)
    # 简单的过滤逻辑：只取数字命名的parquet文件，避免把 index_xxx.parquet 合进去
    stock_files = [f for f in all_files if f.endswith(".parquet") and f[0].isdigit()]
    
    if not stock_files:
        logger.warning("未发现可合并的股票数据文件。")
        return

    logger.info(f"发现 {len(stock_files)} 个股票文件，开始读取并合并...")

    # 4. 批量读取
    df_list = []
    # 使用 tqdm 显示进度
    for f in tqdm(stock_files, desc="Merging"):
        path = os.path.join(raw_dir, f)
        try:
            df = read_parquet(path)
            # 这里的 df 应该已经包含了 symbol 列 (在 datahub/source 中处理过)
            # 如果不确定，可以在这里再次校验或补全 symbol
            if "symbol" not in df.columns:
                # 从文件名提取 symbol (例如 "600000.parquet" -> "600000")
                symbol = os.path.splitext(f)[0]
                df["symbol"] = symbol
            
            df_list.append(df)
        except Exception as e:
            logger.error(f"读取文件 {f} 失败: {e}")

    # 5. 合并并保存
    if df_list:
        try:
            full_df = pd.concat(df_list, ignore_index=True)
            
            # 可选：简单按日期和代码排序，方便查看
            if "date" in full_df.columns and "symbol" in full_df.columns:
                full_df = full_df.sort_values(by=["symbol", "date"]).reset_index(drop=True)
                
            save_parquet(full_df, output_path)
            logger.info(f"合并完成！文件已保存至: {output_path}")
            logger.info(f"总行数: {len(full_df)}, 包含股票数: {full_df['symbol'].nunique()}")
        except Exception as e:
            logger.error(f"合并 DataFrame 时发生错误: {e}")
    else:
        logger.warning("没有读取到有效数据，合并取消。")


def main():
    args = parse_args()
    datahub = DataHub()
    
    # 1. 下载个股
    download_stocks(datahub, force_reload=args.force)
    
    # 2. 下载指数
    download_indices(datahub, force_reload=args.force)
    
    # 3. 执行合并 
    # merge_data_files()

if __name__ == "__main__":
    main()