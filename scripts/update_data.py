# scripts/update_data.py

import os
import sys
import argparse
import datetime
import pandas as pd
from tqdm import tqdm

# 路径适配
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data_source.datahub import DataHub
from src.utils.config import GLOBAL_CONFIG
from src.utils.io import read_parquet, save_parquet, ensure_dir
from src.utils.logger import get_logger

logger = get_logger()

class DataUpdater:
    def __init__(self):
        self.config = GLOBAL_CONFIG
        self.paths = self.config["paths"]
        self.datahub = DataHub()
        
        # 今天的日期
        self.today = datetime.datetime.now().strftime("%Y-%m-%d")
        
        # 加载本地交易日历 (用于判断是否需要更新)
        self.calendar_path = os.path.join(self.paths["data_meta"], "trade_calendar.parquet")
        self.trade_dates = []
        self._load_local_calendar()

    def _load_local_calendar(self):
        """加载本地日历，如果不存在则初始化为空"""
        if os.path.exists(self.calendar_path):
            df = read_parquet(self.calendar_path)
            self.trade_dates = pd.to_datetime(df["date"]).dt.date.tolist()
            self.trade_dates.sort()
        else:
            self.trade_dates = []

    def get_last_date(self, df: pd.DataFrame) -> str:
        """获取 DataFrame 中的最后日期"""
        if df is None or df.empty or "date" not in df.columns:
            return None
        return df["date"].max().strftime("%Y-%m-%d")

    def get_next_date(self, date_str: str) -> str:
        """给定日期，返回下一天"""
        if not date_str:
            return self.config["data"]["start_date"]
        
        dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        next_dt = dt + datetime.timedelta(days=1)
        return next_dt.strftime("%Y-%m-%d")

    # ==========================================
    # 1. 更新交易日历
    # ==========================================
    def update_calendar(self):
        logger.info(">>> 步骤 1/3: 检查并更新交易日历...")
        
        # 简单策略：交易日历数据量小，直接重新获取覆盖，确保包含未来的日期
        try:
            # 获取范围：从配置开始日期 到 未来一年
            start_date = self.config["data"]["start_date"]
            future_date = (datetime.datetime.now() + datetime.timedelta(days=365)).strftime("%Y-%m-%d")
            
            df_cal = self.datahub.get_trade_calendar(start_date, future_date)
            
            if not df_cal.empty:
                save_parquet(df_cal, self.calendar_path)
                # 刷新内存中的日历
                self._load_local_calendar()
                logger.info(f"交易日历已更新，最新日期覆盖至: {self.get_last_date(df_cal)}")
            else:
                logger.warning("交易日历接口未返回数据，跳过更新。")
        except Exception as e:
            logger.error(f"更新交易日历失败: {e}")

    # ==========================================
    # 2. 更新指数
    # ==========================================
    def update_index(self):
        index_code = self.config["preprocessing"]["labels"]["index_code"]
        logger.info(f">>> 步骤 2/3: 更新基准指数 ({index_code})...")
        
        file_name = f"index_{index_code.replace('.', '')}.parquet"
        file_path = os.path.join(self.paths["data_raw"], file_name)
        
        df_local = pd.DataFrame()
        start_fetch_date = self.config["data"]["start_date"]
        
        # 1. 读取本地
        if os.path.exists(file_path):
            df_local = read_parquet(file_path)
            last_date = self.get_last_date(df_local)
            if last_date:
                # 如果本地最新日期 >= 今天，说明不用更新
                if last_date >= self.today:
                    logger.info(f"指数 {index_code} 已是最新 ({last_date})，无需更新。")
                    return
                start_fetch_date = self.get_next_date(last_date)
        
        # 2. 下载增量
        logger.info(f"正在下载指数增量数据: {start_fetch_date} -> {self.today}")
        df_new = self.datahub.fetch_index_price(index_code) # 注意：部分接口可能不支持 start/end 参数，需在 fetch 内部过滤
        
        # 如果接口返回了全量，我们需要自行截取
        if not df_new.empty:
            df_new["date"] = pd.to_datetime(df_new["date"])
            mask = df_new["date"] >= pd.to_datetime(start_fetch_date)
            df_delta = df_new[mask]
            
            if not df_delta.empty:
                # 3. 合并
                if not df_local.empty:
                    df_final = pd.concat([df_local, df_delta], axis=0)
                    df_final = df_final.drop_duplicates(subset=["date"]).sort_values("date")
                else:
                    df_final = df_delta
                
                save_parquet(df_final, file_path)
                logger.info(f"指数更新完成，新增 {len(df_delta)} 条记录。")
            else:
                logger.info("未发现新的指数交易数据。")
        else:
            logger.warning("指数数据下载失败或为空。")

    # ==========================================
    # 3. 更新个股
    # ==========================================
    def update_stocks(self):
        logger.info(">>> 步骤 3/3: 增量更新个股数据...")
        
        # 1. 获取目标股票池
        # 这里复用 download_data.py 中的逻辑，先读 meta 再 filter
        # 为了简单，我们这里直接读取 meta 列表，并根据本地是否有文件来决定策略
        meta_path = os.path.join(self.paths["data_meta"], "all_stocks_meta.parquet")
        if not os.path.exists(meta_path):
            logger.error("元数据不存在，请先运行 init_stock_pool.py")
            return
            
        df_meta = read_parquet(meta_path)
        # 这里可以加入 filter_stocks 逻辑，为了代码简洁暂时略过，假设 meta 已经是全量
        # 建议：如果只想更新 data/raw 下已有的文件，可以遍历文件夹
        
        # 策略：遍历 data/raw 下已有的 parquet 文件进行更新
        # 这样避免了"更新脚本"意外下载了之前被配置剔除的股票
        raw_dir = self.paths["data_raw"]
        existing_files = [f for f in os.listdir(raw_dir) if f.endswith(".parquet") and f[0].isdigit()]
        
        if not existing_files:
            logger.warning("data/raw 下没有任何股票文件，请先运行 download_data.py 进行首次下载。")
            return
            
        update_count = 0
        skip_count = 0
        
        # 获取最新的市场交易日 (Market Last Date)
        if self.trade_dates:
            market_last_date = self.trade_dates[-1] # datetime.date 对象
        else:
            market_last_date = datetime.date.today()

        pbar = tqdm(existing_files, desc="Updating Stocks")
        
        for file_name in pbar:
            symbol = file_name.replace(".parquet", "")
            file_path = os.path.join(raw_dir, file_name)
            
            try:
                # 1. 读取本地最后一行 (优化：不需要读全量，但 parquet 读尾部比较麻烦，先读全量)
                # 如果文件很大，可以考虑只读 meta 信息，但在日线级别通常很快
                df_local = read_parquet(file_path)
                last_date_str = self.get_last_date(df_local)
                
                if not last_date_str:
                    # 文件损坏或为空，重新下载全量
                    start_date = self.config["data"]["start_date"]
                else:
                    last_date = datetime.datetime.strptime(last_date_str, "%Y-%m-%d").date()
                    
                    # 检查是否已经是最新
                    # 如果本地最后日期 >= 市场最后交易日，跳过
                    if last_date >= market_last_date:
                        skip_count += 1
                        continue
                        
                    start_date = self.get_next_date(last_date_str)

                # 2. 下载增量
                # 为了防止 start_date > end_date 报错，加个判断
                if start_date > self.today:
                    skip_count += 1
                    continue
                    
                df_new = self.datahub.fetch_price(symbol, start_date=start_date, end_date=self.today)
                
                if not df_new.empty:
                    # 3. 合并与去重
                    df_final = pd.concat([df_local, df_new], axis=0)
                    df_final = df_final.drop_duplicates(subset=["date"], keep="last")
                    df_final = df_final.sort_values("date").reset_index(drop=True)
                    
                    save_parquet(df_final, file_path)
                    update_count += 1
                else:
                    # 没下载到数据（可能是停牌）
                    skip_count += 1
                    
                pbar.set_postfix({"Upd": update_count, "Skip": skip_count})
                
            except Exception as e:
                logger.error(f"更新 {symbol} 失败: {e}")
        
        logger.info(f"更新完成。已更新: {update_count}, 跳过(无需更新/停牌): {skip_count}")

def main():
    parser = argparse.ArgumentParser(description="增量更新本地数据")
    parser.parse_args()
    
    updater = DataUpdater()
    
    # 1. 先更日历
    updater.update_calendar()
    
    # 2. 更指数
    updater.update_index()
    
    # 3. 更个股
    updater.update_stocks()
    
    logger.info("="*50)
    logger.info("⚠️  数据更新已完成 (data/raw)。")
    logger.info("下一步建议：")
    logger.info("1. python scripts/clean_and_check.py (清洗新数据)")
    logger.info("2. python scripts/run_eda.py (检查数据质量)")
    logger.info("3. python scripts/rebuild_features.py (重算特征)")
    logger.info("="*50)

if __name__ == "__main__":
    main()