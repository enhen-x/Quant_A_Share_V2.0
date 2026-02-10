# scripts/update_data.py

import os
import sys
import argparse
import datetime
import pandas as pd
import subprocess
from tqdm import tqdm

# 路径适配
current_dir = os.path.dirname(os.path.abspath(__file__))
# 从当前文件位置 (scripts/date_landing) 返回两级到项目根目录
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_source.datahub import DataHub
from src.utils.config import GLOBAL_CONFIG
from src.utils.io import read_parquet, save_parquet, ensure_dir
from src.utils.logger import get_logger
import glob

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
        df_new = self.datahub.fetch_index_price(index_code, start_date=start_fetch_date, end_date=self.today) 
        
        if not df_new.empty:
            df_new["date"] = pd.to_datetime(df_new["date"])
            
            # 3. 合并
            if not df_local.empty:
                df_final = pd.concat([df_local, df_new], axis=0)
                df_final = df_final.drop_duplicates(subset=["date"]).sort_values("date")
            else:
                df_final = df_new
            
            save_parquet(df_final, file_path)
            logger.info(f"指数更新完成，新增 {len(df_new)} 条记录。")
        else:
            logger.info("未发现新的指数交易数据或数据下载失败。")

    # ==========================================
    # 3. 更新个股
    # ==========================================
    def update_stocks(self):
        logger.info(">>> 步骤 3/3: 增量更新个股数据...")
        
        meta_path = os.path.join(self.paths["data_meta"], "all_stocks_meta.parquet")
        if not os.path.exists(meta_path):
            logger.error("元数据不存在，请先运行 init_stock_pool.py")
            return
            
        raw_dir = self.paths["data_raw"]
        # 仅更新 data/raw 下已有的文件
        existing_files = [f for f in os.listdir(raw_dir) if f.endswith(".parquet") and f[0].isdigit()]
        
        if not existing_files:
            logger.warning("data/raw 下没有任何股票文件，请先运行 download_data.py 进行首次下载。")
            return
            
        update_count = 0
        skip_count = 0
        
        # 获取最新的市场交易日
        if self.trade_dates:
            market_last_date = self.trade_dates[-1] 
        else:
            market_last_date = datetime.date.today()

        pbar = tqdm(existing_files, desc="Updating Stocks")
        
        for file_name in pbar:
            symbol = file_name.replace(".parquet", "")
            file_path = os.path.join(raw_dir, file_name)
            
            try:
                # 1. 读取本地最后一行
                df_local = read_parquet(file_path)
                last_date_str = self.get_last_date(df_local)
                
                if not last_date_str:
                    start_date = self.config["data"]["start_date"]
                else:
                    last_date = datetime.datetime.strptime(last_date_str, "%Y-%m-%d").date()
                    
                    # 检查是否已经是最新
                    if last_date >= market_last_date:
                        skip_count += 1
                        continue
                        
                    start_date = self.get_next_date(last_date_str)

                # 为了防止 start_date > end_date 报错
                if start_date > self.today:
                    skip_count += 1
                    continue
                    
                df_new = self.datahub.fetch_price(symbol, start_date=start_date, end_date=self.today)
                
                if not df_new.empty:
                    # 合并与去重
                    df_final = pd.concat([df_local, df_new], axis=0)
                    df_final = df_final.drop_duplicates(subset=["date"], keep="last")
                    df_final = df_final.sort_values("date").reset_index(drop=True)
                    
                    save_parquet(df_final, file_path)
                    update_count += 1
                else:
                    skip_count += 1
                    
                pbar.set_postfix({"Upd": update_count, "Skip": skip_count})
                
            except Exception as e:
                logger.error(f"更新 {symbol} 失败: {e}")
        
        logger.info(f"更新完成。已更新: {update_count}, 跳过(无需更新/停牌): {skip_count}")

def verify_data_freshness(step_name, data_dir, file_pattern="*.parquet", single_file=None):
    """
    验证数据目录或单个文件中的最新日期
    :param step_name: 步骤名称
    :param data_dir: 数据目录 (相对于 project_root)
    :param file_pattern: 文件匹配模式
    :param single_file: 如果指定，只检查该单文件 (相对于 project_root)
    """
    logger.info(f"\n📅 [{step_name}] 数据新鲜度检查:")
    
    try:
        if single_file:
            # 检查单个文件
            file_path = os.path.join(project_root, single_file)
            if not os.path.exists(file_path):
                logger.warning(f"   文件不存在: {single_file}")
                return
            df = read_parquet(file_path)
            if df is not None and not df.empty and "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                latest = df["date"].max()
                earliest = df["date"].min()
                n_dates = df["date"].nunique()
                logger.info(f"   📄 {os.path.basename(single_file)}")
                logger.info(f"      日期范围: {earliest.strftime('%Y-%m-%d')} ~ {latest.strftime('%Y-%m-%d')} ({n_dates} 个交易日)")
                
                # 检查是否有 label 列，统计其 NaN 情况
                if "label" in df.columns:
                    label_valid = df["label"].notna().sum()
                    label_nan = df["label"].isna().sum()
                    label_latest = df[df["label"].notna()]["date"].max() if label_valid > 0 else None
                    logger.info(f"      标签(label): 有效={label_valid:,}, NaN={label_nan:,}")
                    if label_latest:
                        logger.info(f"      标签最新有效日期: {label_latest.strftime('%Y-%m-%d')}")
                    else:
                        logger.warning(f"      ⚠️ 标签全部为 NaN!")
                
                # 检查 feat_ 列情况
                feat_cols = [c for c in df.columns if c.startswith("feat_")]
                if feat_cols:
                    feat_latest = df.dropna(subset=feat_cols[:3])["date"].max() if not df.dropna(subset=feat_cols[:3]).empty else None
                    if feat_latest:
                        logger.info(f"      特征最新有效日期: {feat_latest.strftime('%Y-%m-%d')} ({len(feat_cols)} 个特征列)")
            else:
                logger.warning(f"   文件为空或缺少 date 列: {single_file}")
            return

        # 检查目录下的文件
        dir_path = os.path.join(project_root, data_dir)
        if not os.path.exists(dir_path):
            logger.warning(f"   目录不存在: {data_dir}")
            return
        
        files = [f for f in os.listdir(dir_path) if f.endswith(".parquet") and f[0].isdigit()]
        if not files:
            logger.warning(f"   目录下没有股票数据文件: {data_dir}")
            return
        
        # 随机抽样几只股票检查
        import random
        sample_files = random.sample(files, min(5, len(files)))
        latest_dates = []
        
        for f in sample_files:
            fp = os.path.join(dir_path, f)
            df = read_parquet(fp)
            if df is not None and not df.empty and "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                latest_dates.append((f.replace(".parquet", ""), df["date"].max()))
        
        if latest_dates:
            overall_max = max(d for _, d in latest_dates)
            overall_min = min(d for _, d in latest_dates)
            logger.info(f"   共 {len(files)} 只股票, 抽样 {len(sample_files)} 只:")
            for sym, dt in latest_dates:
                logger.info(f"      {sym}: 最新日期 {dt.strftime('%Y-%m-%d')}")
            logger.info(f"   📊 抽样最新日期范围: {overall_min.strftime('%Y-%m-%d')} ~ {overall_max.strftime('%Y-%m-%d')}")
        else:
            logger.warning(f"   抽样文件均无有效日期数据")
    except Exception as e:
        logger.error(f"   数据新鲜度检查失败: {e}")


def run_external_script(script_rel_path, step_name):
    """
    调用外部 Python 脚本
    :param script_rel_path: 相对于项目根目录的脚本路径 (如 scripts/analisis/clean_and_check.py)
    :param step_name: 步骤名称
    """
    script_path = os.path.join(project_root, script_rel_path)
    
    logger.info("\n" + "="*60)
    logger.info(f"🚀 正在启动: {step_name} ...")
    logger.info(f"   脚本路径: {script_path}")
    logger.info("="*60)
    
    if not os.path.exists(script_path):
        logger.error(f"❌ 找不到脚本文件: {script_path}")
        return False
        
    try:
        # 使用当前 Python 解释器执行
        cmd = [sys.executable, script_path]
        # cwd 设置为 project_root 确保脚本内部相对路径逻辑正常
        result = subprocess.run(cmd, cwd=project_root)
        
        if result.returncode == 0:
            logger.info(f"✅ {step_name} 执行成功。")
            return True
        else:
            logger.error(f"❌ {step_name} 执行失败，返回码: {result.returncode}")
            return False
    except Exception as e:
        logger.error(f"❌ {step_name} 发生异常: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="增量更新本地数据并运行全流程")
    parser.parse_args()
    
    # === 1. 更新数据 (Download) ===
    try:
        updater = DataUpdater()
        updater.update_calendar()
        updater.update_index()
        updater.update_stocks()
    except Exception as e:
        logger.error(f"数据更新阶段发生严重错误: {e}")
        return
    
    # ✅ 步骤1完成 - 验证原始数据新鲜度
    verify_data_freshness("步骤1: 数据下载", GLOBAL_CONFIG["paths"]["data_raw"])

    # === 2. 清洗数据 (Clean) ===
    # 脚本: scripts/analisis/clean_and_check.py
    if not run_external_script(os.path.join("scripts", "analisis", "clean_and_check.py"), "数据清洗 (Clean)"):
        logger.warning("流程中断：数据清洗失败。")
        return
    
    # ✅ 步骤2完成 - 验证清洗后数据新鲜度
    verify_data_freshness("步骤2: 数据清洗", GLOBAL_CONFIG["paths"]["data_cleaned"])

    # === 3. 构建特征 (Feature Engineering) ===
    # 脚本: scripts/feature_create/rebuild_features.py
    if not run_external_script(os.path.join("scripts", "feature_create", "rebuild_features.py"), "特征工程 (Features)"):
        logger.warning("流程中断：特征构建失败。")
        return
    
    # ✅ 步骤3完成 - 验证特征数据新鲜度（含标签检查）
    concat_file = GLOBAL_CONFIG.get("preprocessing", {}).get("batch", {}).get("concat_file", "all_stocks.parquet")
    verify_data_freshness("步骤3: 特征工程", None, 
                          single_file=os.path.join(GLOBAL_CONFIG["paths"]["data_processed"], concat_file))

    # === 4. 每日推荐 (Recommendation) ===
    # 脚本: scripts/back_test/run_recommendation.py
    # 推荐最近 N 个交易日由 config/main.yaml -> strategy.recommend_history_days 控制
    if not run_external_script(os.path.join("scripts", "back_test", "run_recommendation.py"), "策略推荐 (Recommendation)"):
        logger.warning("流程中断：推荐生成失败。")
        return
    
    # ✅ 步骤4完成 - 验证推荐结果
    picks_dir = os.path.join(GLOBAL_CONFIG["paths"]["reports"], "daily_picks")
    picks_abs = os.path.join(project_root, picks_dir)
    if os.path.exists(picks_abs):
        csv_files = sorted(glob.glob(os.path.join(picks_abs, "picks_*.csv")))
        if csv_files:
            latest_pick = csv_files[-1]
            logger.info(f"\n📅 [步骤4: 策略推荐] 数据新鲜度检查:")
            logger.info(f"   📄 最新推荐文件: {os.path.basename(latest_pick)}")
            try:
                df_pick = pd.read_csv(latest_pick)
                logger.info(f"   推荐股票数: {len(df_pick)}")
                if "symbol" in df_pick.columns:
                    logger.info(f"   推荐列表: {', '.join(df_pick['symbol'].tolist())}")
            except Exception as e:
                logger.warning(f"   读取推荐文件失败: {e}")
    
    logger.info("\n" + "="*60)
    logger.info("🎉🎉🎉 每日全流程任务顺利完成！请查看 reports 目录下的推荐结果。")
    logger.info("="*60)

if __name__ == "__main__":
    main()
