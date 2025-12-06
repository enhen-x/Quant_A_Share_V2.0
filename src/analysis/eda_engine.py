# src/analysis/eda_engine.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.utils.config import GLOBAL_CONFIG
from src.utils.logger import get_logger
from src.utils.io import read_parquet, ensure_dir, save_csv

logger = get_logger()

class EDAEngine:
    def __init__(self):
        self.config = GLOBAL_CONFIG
        self.paths = self.config["paths"]
        self.ana_cfg = self.config.get("analysis", {}).get("eda", {})
        
        # 路径初始化
        self.cleaned_dir = self.paths.get("data_cleaned", os.path.join(self.paths["data_root"], "raw_cleaned"))
        self.output_dir = os.path.join(self.paths["project_root"], self.paths.get("reports", "reports"), "eda")
        self.figure_dir = os.path.join(self.paths["project_root"], self.paths.get("figures", "figures"))
        
        ensure_dir(self.output_dir)
        ensure_dir(self.figure_dir)
        
        self._setup_plotting()
        
        self.df_sample = None
        self.df_index = None

    def _setup_plotting(self):
        """配置绘图风格"""
        plot_cfg = self.ana_cfg.get("plot", {})
        try:
            plt.style.use(plot_cfg.get("style", "ggplot"))
        except:
            plt.style.use("ggplot")
        
        # 字体设置 (兼容中英文)
        user_fonts = plot_cfg.get("font_sans_serif", [])
        default_fonts = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
        plt.rcParams['font.sans-serif'] = user_fonts + default_fonts
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.figsize'] = tuple(plot_cfg.get("figure_size", [12, 6]))

    def load_data(self, sample_size=None, random_state=None):
        """加载数据"""
        if sample_size is None:
            sample_size = self.ana_cfg.get("default_sample_size", 200)
        if random_state is None:
            random_state = self.ana_cfg.get("random_seed", 42)

        # 1. 加载指数
        logger.info("正在加载基准指数数据...")
        try:
            index_code = self.config.get("preprocessing", {}).get("labels", {}).get("index_code", "000300.SH")
            index_file = f"index_{index_code.replace('.', '')}.parquet"
            index_path = os.path.join(self.paths["data_raw"], index_file)
            if os.path.exists(index_path):
                self.df_index = read_parquet(index_path)
                self.df_index["date"] = pd.to_datetime(self.df_index["date"])
                self.df_index = self.df_index.sort_values("date").set_index("date")
        except Exception as e:
            logger.warning(f"加载指数失败: {e}")

        # 2. 加载个股
        logger.info(f"正在采样个股 (目标: {sample_size} 只)...")
        if not os.path.exists(self.cleaned_dir):
            logger.error("数据目录不存在")
            return

        files = [f for f in os.listdir(self.cleaned_dir) if f.endswith(".parquet")]
        if not files: return

        if sample_size < len(files):
            np.random.seed(random_state)
            sampled_files = np.random.choice(files, sample_size, replace=False)
        else:
            sampled_files = files

        dfs = []
        for f in tqdm(sampled_files, desc="Loading"):
            try:
                df = read_parquet(os.path.join(self.cleaned_dir, f))
                df["symbol"] = f.replace(".parquet", "")
                df["date"] = pd.to_datetime(df["date"])
                df = df.sort_values("date")
                
                # 预计算常用字段
                df["pct_chg"] = df["close"].pct_change()
                if "turnover" not in df.columns and "volume" in df.columns:
                    # 如果没有换手率，暂时用 log_volume 代替分析
                    df["log_volume"] = np.log1p(df["volume"])
                
                dfs.append(df)
            except:
                pass

        if dfs:
            self.df_sample = pd.concat(dfs, ignore_index=True)
            logger.info(f"加载完成: {len(self.df_sample)} 行")

    def analyze_distributions(self):
        """1. 收益率分布"""
        if self.df_sample is None: return
        logger.info("--- 分析 1/5: 收益率分布 ---")
        
        data = self.df_sample["pct_chg"].dropna()
        # 剔除极端值画图以便看清分布
        plot_data = data[(data > -0.11) & (data < 0.11)]

        plt.figure()
        sns.histplot(plot_data, bins=100, kde=True, stat="density", color="skyblue")
        
        # 绘制阈值线
        cfg_bins = self.config.get("preprocessing", {}).get("labels", {}).get("quantile_bins", [0.2, 0.8])
        thresholds = data.quantile(cfg_bins).values
        for i, th in enumerate(thresholds):
            plt.axvline(th, color='r', linestyle='--', label=f'Q{cfg_bins[i]}: {th:.4f}')
            
        plt.title("Daily Return Distribution (Clipped +/- 11%)")
        plt.legend()
        plt.savefig(os.path.join(self.figure_dir, "dist_returns.png"))
        plt.close()

    def analyze_quality(self):
        """2. 数据质量：停牌率与缺失率"""
        if self.df_sample is None: return
        logger.info("--- 分析 2/5: 数据质量 (停牌/缺失) ---")

        # A. 停牌率 (Zero Volume)
        # 假设 volume=0 代表停牌
        # [Fix]: 显式选择 ["volume"] 列再进行 apply，消除 FutureWarning
        suspension_ratios = self.df_sample.groupby("symbol")["volume"].apply(
            lambda x: (x < 1e-6).mean()
        )
        
        plt.figure()
        sns.histplot(suspension_ratios, bins=50, kde=False, color="orange")
        plt.title("Suspension Ratio Distribution (Per Stock)")
        plt.xlabel("Suspension Ratio (0.1 = 10% days suspended)")
        plt.savefig(os.path.join(self.figure_dir, "dist_suspension.png"))
        plt.close()

        # 输出停牌严重的股票
        bad_stocks = suspension_ratios[suspension_ratios > 0.1]
        if not bad_stocks.empty:
            logger.warning(f"有 {len(bad_stocks)} 只股票停牌率超过 10%，可能需要过滤。")

            
    def analyze_autocorrelation(self):
        """3. 自相关性分析 (Momentum check)"""
        if self.df_sample is None: return
        logger.info("--- 分析 3/5: 收益率自相关性 (Lag-1) ---")
        
        # 计算每只股票 Lag-1 自相关系数
        # corr(t, t-1)
        autocorrs = self.df_sample.groupby("symbol")["pct_chg"].apply(
            lambda x: x.autocorr(lag=1)
        ).dropna()
        
        plt.figure()
        sns.histplot(autocorrs, bins=50, kde=True, color="purple")
        plt.axvline(0, color="black", linestyle="--")
        plt.title("Return Autocorrelation (Lag-1) Distribution")
        plt.xlabel("Autocorrelation Coefficient")
        
        mean_corr = autocorrs.mean()
        plt.text(0.05, plt.ylim()[1]*0.9, f"Mean: {mean_corr:.4f}")
        
        plt.savefig(os.path.join(self.figure_dir, "dist_autocorr.png"))
        plt.close()
        
        logger.info(f"平均自相关系数: {mean_corr:.4f} (正值暗示动量，负值暗示反转)")

    def analyze_turnover(self):
        """4. 流动性分析 (Turnover/Volume)"""
        if self.df_sample is None: return
        logger.info("--- 分析 4/5: 流动性分布 ---")
        
        # 如果有 turnover 字段最好，没有则用 log(amount)
        target_col = "turnover" if "turnover" in self.df_sample.columns else "amount"
        
        if target_col not in self.df_sample.columns:
            logger.warning("未找到 turnover 或 amount 字段，跳过流动性分析。")
            return

        # 取对数分布，因为金额跨度很大
        data = self.df_sample[target_col].replace(0, np.nan).dropna()
        if target_col == "amount":
            data = np.log1p(data)
            xlabel = "Log(Amount)"
        else:
            xlabel = "Turnover Ratio (%)"
            
        plt.figure()
        sns.histplot(data, bins=100, color="green")
        plt.title(f"Liquidity Distribution ({xlabel})")
        plt.xlabel(xlabel)
        plt.savefig(os.path.join(self.figure_dir, "dist_liquidity.png"))
        plt.close()

    def check_alignment(self, n_check=3):
        """5. 对齐检查 (保留原有)"""
        if self.df_sample is None or self.df_index is None: return
        logger.info("--- 分析 5/5: 走势对齐检查 ---")
        
        symbols = self.df_sample["symbol"].unique()
        check_syms = np.random.choice(symbols, min(n_check, len(symbols)), replace=False)
        
        recent_date = self.df_index.index.max() - pd.Timedelta(days=365*2)
        idx_sub = self.df_index[self.df_index.index >= recent_date]
        if idx_sub.empty: return

        plt.figure(figsize=(12, 6))
        # 指数归一化
        plt.plot(idx_sub["close"]/idx_sub["close"].iloc[0], 
                 label="Index", color="black", lw=2, linestyle="--")
        
        for sym in check_syms:
            stock = self.df_sample[(self.df_sample["symbol"]==sym) & (self.df_sample["date"]>=recent_date)]
            if not stock.empty:
                stock = stock.set_index("date").sort_index()
                plt.plot(stock["close"]/stock["close"].iloc[0], label=str(sym), alpha=0.6)
                
        plt.title("Price Alignment Check (Last 2 Years)")
        plt.legend()
        plt.savefig(os.path.join(self.figure_dir, "check_alignment.png"))
        plt.close()

    def run_full_scan(self, sample_size=None):
        """执行所有分析"""
        self.load_data(sample_size=sample_size)
        
        if self.df_sample is not None and not self.df_sample.empty:
            self.analyze_distributions()   # 1. 收益分布
            self.analyze_quality()         # 2. 停牌/质量 (新增)
            self.analyze_autocorrelation() # 3. 自相关 (新增)
            self.analyze_turnover()        # 4. 流动性 (新增)
            self.check_alignment()         # 5. 对齐检查
            
        logger.info(f"EDA 分析全部完成！请查看: {self.figure_dir}")