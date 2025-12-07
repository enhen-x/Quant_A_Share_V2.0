# src/analysis/horizon_analyzer.py

import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.utils.logger import get_logger
from src.utils.config import GLOBAL_CONFIG
from src.utils.io import read_parquet, ensure_dir, save_csv

logger = get_logger()

class TimeHorizonAnalyzer:
    def __init__(self):
        self.config = GLOBAL_CONFIG
        self.paths = self.config["paths"]
        self.processed_dir = self.paths["data_processed"]
        
        # 定义要测试的时间窗口 (Days)
        self.horizons = [1, 2, 3, 5, 10, 20]
        
        # 输出路径
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(self.paths["reports"], "horizon_analysis", self.timestamp)
        self.figure_dir = os.path.join(self.paths["figures"], "horizon_analysis", self.timestamp)
        ensure_dir(self.output_dir)
        ensure_dir(self.figure_dir)
        
        # 绘图风格
        try:
            plt.style.use('ggplot')
        except:
            pass
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False

    def load_data(self):
        """加载数据并保留价格列用于重新计算收益"""
        file_path = os.path.join(self.processed_dir, "all_stocks.parquet")
        if not os.path.exists(file_path):
            logger.error(f"未找到数据: {file_path}")
            return None
        
        logger.info(f"正在加载数据: {file_path} ...")
        # 我们需要 raw price (close, high, low) 和 features
        df = read_parquet(file_path)
        
        # 筛选特征列
        self.feat_cols = [c for c in df.columns if c.startswith("feat_")]
        
        # 检查是否包含计算收益所需的基础列
        if "amount" in df.columns and "volume" in df.columns:
            self.use_vwap = True
        else:
            self.use_vwap = False
            logger.warning("未找到 amount/volume，将使用 Close 计算收益，可能会有偏差。")
            
        return df

    def _calc_dynamic_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        动态计算不同 Horizon 下的收益率
        逻辑需与 labels.py 保持一致 (T+1 Entry, VWAP)
        """
        logger.info(">>> 正在动态生成多周期标签 (1, 2, 3, 5, 10, 20 days)...")
        
        df = df.copy()
        
        # 1. 确保排序 (Groupby shift 依赖顺序)
        df = df.sort_values(["symbol", "date"])
        
        # 2. 确定价格基准并写入临时列
        # [Fix]: 将计算好的价格序列作为一列加入 df，解决 KeyError 问题
        target_col = "close"
        
        if self.use_vwap:
            # 计算 VWAP: Amount / Volume
            # 处理停牌 (volume=0) -> 使用 Close 填充
            # 注意: replace(0, np.nan) 防止除零报错
            vwap_series = df["amount"] / (df["volume"].replace(0, np.nan))
            
            # 创建临时列
            df["_temp_price_base"] = vwap_series.fillna(df["close"])
            target_col = "_temp_price_base"
            
        # 3. 针对每个 horizon 计算
        # [Fix]: 移除了旧的 apply 代码，直接使用 groupby vectorization
        grouped = df.groupby("symbol")[target_col]
        
        for h in tqdm(self.horizons, desc="Calculating Horizons"):
            # T+1 入场 (Entry), T+1+h 出场 (Exit)
            curr_entry = grouped.shift(-1)
            curr_exit = grouped.shift(-(1 + h))
            
            label_col = f"ret_{h}d"
            df[label_col] = curr_exit / curr_entry - 1.0
            
        # 4. 清理临时列
        if "_temp_price_base" in df.columns:
            df = df.drop(columns=["_temp_price_base"])
            
        return df

    def analyze_ic_decay(self, df: pd.DataFrame):
        """计算 IC 衰减并绘图"""
        logger.info(">>> 计算 IC 衰减曲线...")
        
        label_cols = [f"ret_{h}d" for h in self.horizons]
        
        # Dropna: 确保标签和特征都有值
        sample_df = df.dropna(subset=label_cols + self.feat_cols)
        if sample_df.empty:
            logger.error("有效数据为空！")
            return

        # 选出与 5d 收益相关性最强的 Top 10 特征进行绘图
        base_corr = sample_df[self.feat_cols].corrwith(sample_df["ret_5d"])
        top_features = base_corr.abs().sort_values(ascending=False).head(10).index.tolist()
        
        logger.info(f"选取 Top 10 特征进行分析: {top_features}")
        
        results = {}
        for feat in top_features:
            feat_ics = []
            for h_col in label_cols:
                # Spearman RankIC (更稳健)
                ic = sample_df[feat].corr(sample_df[h_col], method="spearman")
                feat_ics.append(ic)
            results[feat] = feat_ics
            
        # --- 绘图 ---
        plt.figure(figsize=(12, 8))
        
        for feat, ics in results.items():
            # 绘制线条，带点
            plt.plot(self.horizons, ics, marker='o', label=feat, linewidth=2, alpha=0.8)
            
        plt.title("Feature IC Decay Analysis (IC vs Time Horizon)")
        plt.xlabel("Horizon (Days)")
        plt.ylabel("Rank IC (Spearman)")
        plt.axhline(0, color='black', linestyle='--', alpha=0.3)
        plt.xticks(self.horizons) # 强制显示 x 轴刻度
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # 图例放外边
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        save_path = os.path.join(self.figure_dir, "ic_decay_curve.png")
        plt.savefig(save_path)
        logger.info(f"IC 衰减曲线已保存: {save_path}")
        
        # 保存数据表
        df_res = pd.DataFrame(results, index=[f"{h}d" for h in self.horizons])
        save_csv(df_res, os.path.join(self.output_dir, "ic_decay_data.csv"))
        print("\n=== IC Decay Data ===")
        print(df_res)

    def run(self):
        df = self.load_data()
        if df is not None:
            # 1. 重新计算多周期标签
            df_multi = self._calc_dynamic_labels(df)
            # 2. 分析 IC 衰减
            self.analyze_ic_decay(df_multi)
            
            logger.info("=" * 40)
            logger.info(f"时间窗口分析完成！")
            logger.info(f"  - 结果: {self.output_dir}")
            logger.info("=" * 40)

if __name__ == "__main__":
    analyzer = TimeHorizonAnalyzer()
    analyzer.run()