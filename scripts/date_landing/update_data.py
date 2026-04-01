# scripts/update_data.py

import os
import sys
import argparse
import datetime
import glob
import random
import subprocess
import time
from typing import Dict, Tuple

import pandas as pd
from tqdm import tqdm

# 路径适配：从 scripts/date_landing 回到项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_source.datahub import DataHub
from src.utils.config import GLOBAL_CONFIG
from src.utils.io import read_parquet, save_parquet
from src.utils.logger import get_logger

logger = get_logger()


class DataUpdater:
    def __init__(self):
        self.config = GLOBAL_CONFIG
        self.paths = self.config["paths"]
        self.datahub = DataHub()
        self.retry_cfg = self._load_retry_config()
        self._consecutive_fetch_failures = 0

        self.today = datetime.datetime.now().strftime("%Y-%m-%d")

        # 本地交易日历（用于判断是否已经是最新）
        self.calendar_path = os.path.join(self.paths["data_meta"], "trade_calendar.parquet")
        self.trade_dates = []
        self._load_local_calendar()

    def _load_retry_config(self):
        cfg = self.config.get("data", {}).get("retry", {})
        return {
            "max_retries": int(cfg.get("max_retries", 3)),
            "base_sleep": float(cfg.get("base_sleep", 1.0)),
            "max_sleep": float(cfg.get("max_sleep", 8.0)),
            "jitter": float(cfg.get("jitter", 0.5)),
            "reconnect": bool(cfg.get("reconnect", True)),
            "min_interval": float(cfg.get("min_interval", 0.0)),
            "reconnect_on_final_failure": bool(cfg.get("reconnect_on_final_failure", True)),
            "cooldown_after_failures": int(cfg.get("cooldown_after_failures", 5)),
            "cooldown_seconds": float(cfg.get("cooldown_seconds", 6.0)),
        }

    def _is_transient_error(self, err: Exception) -> bool:
        if isinstance(err, (ConnectionError, TimeoutError)):
            return True
        if isinstance(err, OSError):
            winerror = getattr(err, "winerror", None)
            if winerror in (10054, 10053, 10060):
                return True

        msg = str(err).lower()
        transient_keys = [
            "timed out",
            "timeout",
            "connection reset",
            "connection aborted",
            "connection problem",
            "connection refused",
            "remote host",
            "forcibly closed",
            "please login",
            "not login",
            "socket",
            "wsarecv",
            "10054",
            "10053",
            "10060",
            "连接",
            "网络",
            "登录",
        ]
        return any(k in msg for k in transient_keys)

    def _reset_datahub(self, reason: str = None):
        if reason:
            logger.warning(f"重建 DataHub 连接：{reason}")
        self.datahub = DataHub()

    def _fetch_price_with_retry(self, symbol: str, start_date: str, end_date: str):
        max_retries = max(1, self.retry_cfg["max_retries"])
        base_sleep = max(0.0, self.retry_cfg["base_sleep"])
        max_sleep = max(base_sleep, self.retry_cfg["max_sleep"])
        jitter = max(0.0, self.retry_cfg["jitter"])
        reconnect = self.retry_cfg["reconnect"]
        min_interval = max(0.0, self.retry_cfg["min_interval"])
        reconnect_on_final_failure = self.retry_cfg["reconnect_on_final_failure"]
        cooldown_after_failures = max(1, self.retry_cfg["cooldown_after_failures"])
        cooldown_seconds = max(base_sleep, self.retry_cfg["cooldown_seconds"])

        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                df = self.datahub.fetch_price(symbol, start_date=start_date, end_date=end_date)
                self._consecutive_fetch_failures = 0
                if min_interval > 0:
                    time.sleep(min_interval)
                return df
            except Exception as e:
                last_err = e
                if (not self._is_transient_error(e)) or attempt >= max_retries:
                    break

                sleep_s = min(max_sleep, base_sleep * (2 ** (attempt - 1)))
                if jitter:
                    sleep_s += random.random() * jitter
                logger.warning(
                    f"拉取 {symbol} 失败（{attempt}/{max_retries}）：{e}，"
                    f"{sleep_s:.1f}s 后重试"
                )
                time.sleep(sleep_s)
                if reconnect:
                    self._reset_datahub("检测到临时网络异常")

        logger.error(f"拉取 {symbol} 最终失败：{last_err}")
        self._consecutive_fetch_failures += 1

        if reconnect_on_final_failure:
            self._reset_datahub(f"{symbol} 最终失败后强制重连")

        if self._consecutive_fetch_failures >= cooldown_after_failures:
            logger.warning(
                f"已连续失败 {self._consecutive_fetch_failures} 次，"
                f"冷却 {cooldown_seconds:.1f}s 后继续"
            )
            time.sleep(cooldown_seconds)
            self._consecutive_fetch_failures = 0

        return None

    def _get_last_date_cache_path(self) -> str:
        return os.path.join(self.paths["data_meta"], "stock_last_dates_cache.parquet")

    def _load_last_date_cache(self) -> Dict[str, Dict[str, object]]:
        cache_path = self._get_last_date_cache_path()
        if not os.path.exists(cache_path):
            return {}

        try:
            df_cache = read_parquet(cache_path)
            if df_cache is None or df_cache.empty:
                return {}

            required_cols = {"symbol", "last_date", "file_mtime_ns"}
            if not required_cols.issubset(df_cache.columns):
                logger.warning("last-date 缓存字段不匹配，准备重建")
                return {}

            cache: Dict[str, Dict[str, object]] = {}
            for _, row in df_cache.iterrows():
                symbol = str(row["symbol"])
                if not symbol:
                    continue

                last_date = None
                if pd.notna(row["last_date"]) and str(row["last_date"]).strip():
                    last_date = pd.to_datetime(row["last_date"]).strftime("%Y-%m-%d")

                file_mtime_ns = int(row["file_mtime_ns"]) if pd.notna(row["file_mtime_ns"]) else 0
                cache[symbol] = {"last_date": last_date, "file_mtime_ns": file_mtime_ns}

            return cache
        except Exception as e:
            logger.warning(f"读取 last-date 缓存失败，回退为重建：{e}")
            return {}

    def _save_last_date_cache(self, cache: Dict[str, Dict[str, object]]):
        rows = []
        for symbol, info in sorted(cache.items()):
            rows.append(
                {
                    "symbol": symbol,
                    "last_date": info.get("last_date"),
                    "file_mtime_ns": int(info.get("file_mtime_ns", 0)),
                }
            )

        df_cache = pd.DataFrame(rows, columns=["symbol", "last_date", "file_mtime_ns"])
        save_parquet(df_cache, self._get_last_date_cache_path())

    def _extract_last_date_fast(self, file_path: str) -> str:
        try:
            df_date = pd.read_parquet(file_path, columns=["date"])
            if df_date is None or df_date.empty:
                return None
            dt = pd.to_datetime(df_date["date"], errors="coerce").max()
            if pd.isna(dt):
                return None
            return dt.strftime("%Y-%m-%d")
        except Exception:
            # 兼容某些 parquet 引擎不支持列投影
            df_all = read_parquet(file_path)
            return self.get_last_date(df_all)

    def _prepare_stock_last_dates(
        self, existing_files: list, raw_dir: str
    ) -> Tuple[Dict[str, str], Dict[str, Dict[str, object]]]:
        last_dates: Dict[str, str] = {}
        cache = self._load_last_date_cache()
        to_refresh = []

        for file_name in existing_files:
            symbol = file_name.replace(".parquet", "")
            file_path = os.path.join(raw_dir, file_name)
            try:
                mtime_ns = os.stat(file_path).st_mtime_ns
            except OSError:
                continue

            cached = cache.get(symbol)
            if cached and int(cached.get("file_mtime_ns", 0)) == mtime_ns:
                last_dates[symbol] = cached.get("last_date")
            else:
                to_refresh.append((symbol, file_path, mtime_ns))

        if to_refresh:
            logger.info(f"last-date 缓存需要刷新：{len(to_refresh)} 个文件")
            refresh_bar = tqdm(to_refresh, desc="Refresh last-date cache", leave=False)
            for symbol, file_path, mtime_ns in refresh_bar:
                try:
                    last_date = self._extract_last_date_fast(file_path)
                except Exception as e:
                    logger.warning(f"读取 {symbol} 本地最后日期失败：{e}")
                    last_date = None

                cache[symbol] = {"last_date": last_date, "file_mtime_ns": int(mtime_ns)}
                last_dates[symbol] = last_date

            self._save_last_date_cache(cache)
            logger.info("last-date 缓存刷新完成")
        else:
            logger.info("last-date 缓存全部命中")

        return last_dates, cache

    def _load_local_calendar(self):
        """加载本地交易日历，不存在则置空。"""
        if os.path.exists(self.calendar_path):
            df = read_parquet(self.calendar_path)
            self.trade_dates = pd.to_datetime(df["date"]).dt.date.tolist()
            self.trade_dates.sort()
        else:
            self.trade_dates = []

    def get_last_date(self, df: pd.DataFrame) -> str:
        """返回 dataframe 中 date 列的最大日期（YYYY-MM-DD）。"""
        if df is None or df.empty or "date" not in df.columns:
            return None
        return df["date"].max().strftime("%Y-%m-%d")

    def get_next_date(self, date_str: str) -> str:
        """给定 YYYY-MM-DD，返回下一天。"""
        if not date_str:
            return self.config["data"]["start_date"]

        dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        next_dt = dt + datetime.timedelta(days=1)
        return next_dt.strftime("%Y-%m-%d")

    # ==========================================
    # 1. 更新交易日历
    # ==========================================
    def update_calendar(self):
        logger.info(">>> 步骤 1/3：检查并更新交易日历...")

        try:
            start_date = self.config["data"]["start_date"]
            future_date = (datetime.datetime.now() + datetime.timedelta(days=365)).strftime("%Y-%m-%d")

            df_cal = self.datahub.get_trade_calendar(start_date, future_date)
            if not df_cal.empty:
                save_parquet(df_cal, self.calendar_path)
                self._load_local_calendar()
                logger.info(f"交易日历更新完成，最新日期：{self.get_last_date(df_cal)}")
            else:
                logger.warning("交易日历接口返回空数据，跳过更新")
        except Exception as e:
            logger.error(f"更新交易日历失败：{e}")

    # ==========================================
    # 2. 更新指数
    # ==========================================
    def update_index(self):
        index_code = self.config["preprocessing"]["labels"]["index_code"]
        logger.info(f">>> 步骤 2/3：更新基准指数（{index_code}）...")

        file_name = f"index_{index_code.replace('.', '')}.parquet"
        file_path = os.path.join(self.paths["data_raw"], file_name)

        df_local = pd.DataFrame()
        start_fetch_date = self.config["data"]["start_date"]

        if os.path.exists(file_path):
            df_local = read_parquet(file_path)
            last_date = self.get_last_date(df_local)
            if last_date:
                if last_date >= self.today:
                    logger.info(f"指数 {index_code} 已是最新（{last_date}），跳过")
                    return
                start_fetch_date = self.get_next_date(last_date)

        logger.info(f"下载指数增量数据：{start_fetch_date} -> {self.today}")
        df_new = self.datahub.fetch_index_price(index_code, start_date=start_fetch_date, end_date=self.today)

        if not df_new.empty:
            df_new["date"] = pd.to_datetime(df_new["date"])

            if not df_local.empty:
                df_final = pd.concat([df_local, df_new], axis=0)
                df_final = df_final.drop_duplicates(subset=["date"]).sort_values("date")
            else:
                df_final = df_new

            save_parquet(df_final, file_path)
            logger.info(f"指数更新完成，新增 {len(df_new)} 条记录")
        else:
            logger.info("没有新的指数数据，或指数下载失败")

    # ==========================================
    # 3. 更新个股
    # ==========================================
    def update_stocks(self):
        logger.info(">>> 步骤 3/3：增量更新个股数据...")

        meta_path = os.path.join(self.paths["data_meta"], "all_stocks_meta.parquet")
        if not os.path.exists(meta_path):
            logger.error("元数据不存在，请先运行 init_stock_pool.py")
            return

        raw_dir = self.paths["data_raw"]
        existing_files = sorted([f for f in os.listdir(raw_dir) if f.endswith(".parquet") and f[0].isdigit()])

        if not existing_files:
            logger.warning("data/raw 下没有股票文件，请先运行 download_data.py")
            return

        update_count = 0
        skip_count = 0
        error_count = 0

        market_last_date = self.trade_dates[-1] if self.trade_dates else datetime.date.today()

        # 缓存每只股票的本地最后交易日，避免每次重跑都全量扫 parquet
        symbol_last_dates, cache = self._prepare_stock_last_dates(existing_files, raw_dir)
        pbar = tqdm(existing_files, desc="Updating Stocks")

        for file_name in pbar:
            symbol = file_name.replace(".parquet", "")
            file_path = os.path.join(raw_dir, file_name)

            try:
                last_date_str = symbol_last_dates.get(symbol)
                start_date = self.config["data"]["start_date"]

                if last_date_str:
                    try:
                        last_date = datetime.datetime.strptime(last_date_str, "%Y-%m-%d").date()
                    except ValueError:
                        last_date = None

                    if last_date:
                        if last_date >= market_last_date:
                            skip_count += 1
                            pbar.set_postfix({"Upd": update_count, "Skip": skip_count, "Err": error_count})
                            continue
                        start_date = self.get_next_date(last_date_str)

                if start_date > self.today:
                    skip_count += 1
                    pbar.set_postfix({"Upd": update_count, "Skip": skip_count, "Err": error_count})
                    continue

                df_new = self._fetch_price_with_retry(symbol, start_date=start_date, end_date=self.today)

                if df_new is None:
                    error_count += 1
                    pbar.set_postfix({"Upd": update_count, "Skip": skip_count, "Err": error_count})
                    continue

                if df_new.empty:
                    skip_count += 1
                    pbar.set_postfix({"Upd": update_count, "Skip": skip_count, "Err": error_count})
                    continue

                try:
                    df_local = read_parquet(file_path)
                except Exception:
                    df_local = pd.DataFrame()

                if df_local is None or df_local.empty:
                    df_final = df_new
                else:
                    df_final = pd.concat([df_local, df_new], axis=0)
                    df_final = df_final.drop_duplicates(subset=["date"], keep="last")
                    df_final = df_final.sort_values("date").reset_index(drop=True)

                save_parquet(df_final, file_path)
                latest_saved_date = self.get_last_date(df_final)
                symbol_last_dates[symbol] = latest_saved_date

                try:
                    mtime_ns = int(os.stat(file_path).st_mtime_ns)
                except OSError:
                    mtime_ns = 0
                cache[symbol] = {"last_date": latest_saved_date, "file_mtime_ns": mtime_ns}

                update_count += 1
                pbar.set_postfix({"Upd": update_count, "Skip": skip_count, "Err": error_count})

            except Exception as e:
                logger.error(f"更新 {symbol} 失败：{e}")
                error_count += 1
                pbar.set_postfix({"Upd": update_count, "Skip": skip_count, "Err": error_count})

        try:
            self._save_last_date_cache(cache)
        except Exception as e:
            logger.warning(f"保存 last-date 缓存失败：{e}")

        logger.info(
            f"个股更新完成。已更新: {update_count}, 跳过: {skip_count}, 失败: {error_count}"
        )


def verify_data_freshness(step_name, data_dir, file_pattern="*.parquet", single_file=None):
    """
    验证目录或单文件的数据新鲜度。
    :param step_name: 步骤名称
    :param data_dir: 相对项目根目录的数据目录
    :param file_pattern: 文件匹配模式（保留参数，兼容旧调用）
    :param single_file: 相对项目根目录的单文件路径
    """
    _ = file_pattern
    logger.info(f"\n[ {step_name} ] 数据新鲜度检查")

    try:
        if single_file:
            file_path = os.path.join(project_root, single_file)
            if not os.path.exists(file_path):
                logger.warning(f"文件不存在: {single_file}")
                return

            df = read_parquet(file_path)
            if df is not None and not df.empty and "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                latest = df["date"].max()
                earliest = df["date"].min()
                n_dates = df["date"].nunique()

                logger.info(f"文件: {os.path.basename(single_file)}")
                logger.info(
                    f"日期范围: {earliest.strftime('%Y-%m-%d')} ~ {latest.strftime('%Y-%m-%d')} "
                    f"({n_dates} 个交易日)"
                )

                if "label" in df.columns:
                    label_valid = df["label"].notna().sum()
                    label_nan = df["label"].isna().sum()
                    label_latest = df[df["label"].notna()]["date"].max() if label_valid > 0 else None
                    logger.info(f"label 统计: 有效={label_valid:,}, NaN={label_nan:,}")
                    if label_latest is not None:
                        logger.info(f"label 最新有效日期: {label_latest.strftime('%Y-%m-%d')}")
                    else:
                        logger.warning("label 全部为 NaN")

                feat_cols = [c for c in df.columns if c.startswith("feat_")]
                if feat_cols:
                    sample_cols = feat_cols[:3]
                    df_feat = df.dropna(subset=sample_cols)
                    feat_latest = df_feat["date"].max() if not df_feat.empty else None
                    if feat_latest is not None:
                        logger.info(
                            f"特征最新有效日期: {feat_latest.strftime('%Y-%m-%d')} "
                            f"({len(feat_cols)} 个特征列)"
                        )
            else:
                logger.warning(f"文件为空或缺少 date 列: {single_file}")
            return

        dir_path = os.path.join(project_root, data_dir)
        if not os.path.exists(dir_path):
            logger.warning(f"目录不存在: {data_dir}")
            return

        files = [f for f in os.listdir(dir_path) if f.endswith(".parquet") and f[0].isdigit()]
        if not files:
            logger.warning(f"目录下没有股票数据文件: {data_dir}")
            return

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
            logger.info(f"共 {len(files)} 只股票，抽样 {len(sample_files)} 只")
            for sym, dt in latest_dates:
                logger.info(f"  {sym}: 最新日期 {dt.strftime('%Y-%m-%d')}")
            logger.info(
                f"抽样最新日期范围: {overall_min.strftime('%Y-%m-%d')} ~ {overall_max.strftime('%Y-%m-%d')}"
            )
        else:
            logger.warning("抽样文件都没有有效日期数据")

    except Exception as e:
        logger.error(f"数据新鲜度检查失败: {e}")


def run_external_script(script_rel_path, step_name):
    """
    运行外部 Python 脚本。
    :param script_rel_path: 相对项目根目录的脚本路径
    :param step_name: 步骤名称
    """
    script_path = os.path.join(project_root, script_rel_path)

    logger.info("\n" + "=" * 60)
    logger.info(f"正在启动: {step_name}")
    logger.info(f"脚本路径: {script_path}")
    logger.info("=" * 60)

    if not os.path.exists(script_path):
        logger.error(f"找不到脚本文件: {script_path}")
        return False

    try:
        cmd = [sys.executable, script_path]
        result = subprocess.run(cmd, cwd=project_root)

        if result.returncode == 0:
            logger.info(f"{step_name} 执行成功")
            return True

        logger.error(f"{step_name} 执行失败，返回码: {result.returncode}")
        return False

    except Exception as e:
        logger.error(f"{step_name} 发生异常: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="增量更新本地数据并执行每日全流程")
    parser.parse_args()

    # === 1. 更新数据（Download）===
    try:
        updater = DataUpdater()
        updater.update_calendar()
        updater.update_index()
        updater.update_stocks()
    except Exception as e:
        logger.error(f"数据更新阶段发生严重错误: {e}")
        return

    # 步骤 1 完成后：验证原始数据新鲜度
    verify_data_freshness("步骤1: 数据下载", GLOBAL_CONFIG["paths"]["data_raw"])

    # === 2. 清洗数据（Clean）===
    if not run_external_script(os.path.join("scripts", "analisis", "clean_and_check.py"), "数据清洗 (Clean)"):
        logger.warning("流程中断：数据清洗失败")
        return

    # 步骤 2 完成后：验证清洗后数据
    verify_data_freshness("步骤2: 数据清洗", GLOBAL_CONFIG["paths"]["data_cleaned"])

    # === 3. 特征工程（Feature Engineering）===
    if not run_external_script(
        os.path.join("scripts", "feature_create", "rebuild_features.py"),
        "特征工程 (Features)",
    ):
        logger.warning("流程中断：特征构建失败")
        return

    # 步骤 3 完成后：检查处理后大表
    concat_file = GLOBAL_CONFIG.get("preprocessing", {}).get("batch", {}).get("concat_file", "all_stocks.parquet")
    verify_data_freshness(
        "步骤3: 特征工程",
        None,
        single_file=os.path.join(GLOBAL_CONFIG["paths"]["data_processed"], concat_file),
    )

    # === 4. 推荐生成（Recommendation）===
    if not run_external_script(
        os.path.join("scripts", "back_test", "run_recommendation.py"),
        "策略推荐 (Recommendation)",
    ):
        logger.warning("流程中断：推荐生成失败")
        return

    # 步骤 4 完成后：检查推荐文件
    picks_dir = os.path.join(GLOBAL_CONFIG["paths"]["reports"], "daily_picks")
    picks_abs = os.path.join(project_root, picks_dir)
    if os.path.exists(picks_abs):
        csv_files = sorted(glob.glob(os.path.join(picks_abs, "picks_*.csv")))
        if csv_files:
            latest_pick = csv_files[-1]
            logger.info("\n[步骤4: 策略推荐] 数据新鲜度检查")
            logger.info(f"最新推荐文件: {os.path.basename(latest_pick)}")
            try:
                df_pick = pd.read_csv(latest_pick)
                logger.info(f"推荐股票数: {len(df_pick)}")
                if "symbol" in df_pick.columns:
                    logger.info(f"推荐列表: {', '.join(df_pick['symbol'].tolist())}")
            except Exception as e:
                logger.warning(f"读取推荐文件失败: {e}")

    logger.info("\n" + "=" * 60)
    logger.info("每日全流程任务执行完成，请查看 reports 目录输出")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
