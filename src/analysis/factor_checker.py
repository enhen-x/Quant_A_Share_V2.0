# src/analysis/factor_checker.py

import os
import datetime  # [新增]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from src.utils.logger import get_logger
from src.utils.config import GLOBAL_CONFIG
from src.utils.io import read_parquet, ensure_dir, save_csv

logger = get_logger()

class FactorChecker:
    def __init__(self):
        self.config = GLOBAL_CONFIG
        self.paths = self.config["paths"]
        self.processed_dir = self.paths["data_processed"]
        
        # [新增] 生成时间戳
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # [修改] 路径增加时间戳子目录
        # 报告: reports/factor_analysis/20231201_120000/
        self.output_dir = os.path.join(self.paths["reports"], "factor_analysis", self.timestamp)
        # 图表: figures/factors/20231201_120000/
        self.figure_dir = os.path.join(self.paths["figures"], "factors", self.timestamp)
        
        ensure_dir(self.output_dir)
        ensure_dir(self.figure_dir)
        
        logger.info(f"因子分析报告目录: {self.output_dir}")
        logger.info(f"因子分析图表目录: {self.figure_dir}")
        
        # 绘图设置
        self._setup_plotting()

    def _setup_plotting(self):
        plt.style.use('ggplot')
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
        # 默认图片大小
        plt.rcParams['figure.figsize'] = (12, 6)

    def load_data(self):
        """加载处理后的全量数据"""
        file_path = os.path.join(self.processed_dir, "all_stocks.parquet")
        if not os.path.exists(file_path):
            logger.error(f"未找到特征文件: {file_path}，请先运行 rebuild_features.py")
            return None
        
        logger.info(f"正在加载特征矩阵: {file_path} ...")
        df = read_parquet(file_path)
        
        # 识别特征列和标签列
        self.feat_cols = [c for c in df.columns if c.startswith("feat_")]
        self.label_cols = [c for c in df.columns if c.startswith("label")]
        
        # 简单检查
        if not self.feat_cols:
            logger.warning("未检测到 feat_ 开头的特征列！")
        if not self.label_cols:
            logger.warning("未检测到 label 开头的标签列！")
            
        logger.info(f"数据加载成功: {df.shape}")
        logger.info(f"特征数量: {len(self.feat_cols)}, 标签数量: {len(self.label_cols)}")
        return df

    def check_missing(self, df: pd.DataFrame):
        """1. 缺失值检查"""
        logger.info(">>> 1. 执行缺失值检查...")
        
        # 计算缺失率
        missing = df[self.feat_cols + self.label_cols].isnull().mean()
        high_missing = missing[missing > 0.05] # 缺失超过 5%
        
        # 保存缺失报告
        missing_df = missing.reset_index()
        missing_df.columns = ["column", "missing_ratio"]
        missing_df = missing_df.sort_values("missing_ratio", ascending=False)
        save_csv(missing_df, os.path.join(self.output_dir, "missing_ratio.csv"))
        
        if not high_missing.empty:
            logger.warning("以下字段缺失率较高 (>5%):")
            print(high_missing)
        else:
            logger.info("特征与标签完整性良好（无严重缺失）。")

    def check_label_distribution(self, df: pd.DataFrame):
        """2. 标签分布检查"""
        logger.info(">>> 2. 执行标签分布检查...")
        
        stats_list = []
        for label in self.label_cols:
            data = df[label].dropna()
            if data.empty:
                continue
                
            # 统计指标
            stats = data.describe().to_dict()
            stats["skew"] = data.skew()
            stats["kurt"] = data.kurt()
            stats["label_name"] = label
            stats_list.append(stats)
            
            # 绘图 (剔除极值优化显示)
            q_low, q_high = data.quantile([0.01, 0.99])
            plot_data = data[(data >= q_low) & (data <= q_high)]
            
            plt.figure()
            sns.histplot(plot_data, bins=100, kde=True, color="blue")
            plt.title(f"Label Distribution: {label}\n(Clipped 1%-99%)")
            plt.xlabel("Return / Score")
            plt.tight_layout()
            plt.savefig(os.path.join(self.figure_dir, f"dist_{label}.png"))
            plt.close()
            
        # 保存标签统计表
        if stats_list:
            df_stats = pd.DataFrame(stats_list).set_index("label_name")
            save_csv(df_stats, os.path.join(self.output_dir, "label_stats.csv"))

    def check_ic(self, df: pd.DataFrame):
        """3. IC 分析 (特征有效性 / 泄露检测)"""
        logger.info(">>> 3. 执行 IC 分析 (Information Coefficient)...")
        
        if not self.label_cols or not self.feat_cols:
            return
            
        # 默认取第一个 Label 作为目标
        target_label = "label" if "label" in df.columns else self.label_cols[0]
        logger.info(f"当前 IC 分析目标标签: {target_label}")
        
        # 准备数据 (Dropna)
        valid_df = df[[target_label] + self.feat_cols].dropna()
        if valid_df.empty:
            logger.warning("有效数据为空，无法计算 IC。")
            return
        
        ic_list = []
        
        for feat in self.feat_cols:
            # Pearson IC
            ic = valid_df[feat].corr(valid_df[target_label])
            # Rank IC (Spearman)
            rank_ic = valid_df[feat].corr(valid_df[target_label], method="spearman")
            
            ic_list.append({"Feature": feat, "IC": ic, "RankIC": rank_ic})
            
        df_ic = pd.DataFrame(ic_list).sort_values(by="IC", ascending=False)
        
        # 保存 IC 报告
        save_csv(df_ic, os.path.join(self.output_dir, "feature_ic_report.csv"))
        
        # 打印部分信息
        print("\n=== Top 5 Positive IC ===")
        print(df_ic.head(5))
        print("\n=== Top 5 Negative IC ===")
        print(df_ic.tail(5))
        
        # 画图: IC 柱状图 (Top 30)
        plt.figure(figsize=(10, 8))
        df_plot = df_ic.copy()
        df_plot["AbsIC"] = df_plot["IC"].abs()
        df_plot = df_plot.sort_values("AbsIC", ascending=False).head(30)
        
        # [Fix] 增加 hue="Feature" 并设置 legend=False 以消除 FutureWarning
        sns.barplot(x="IC", y="Feature", data=df_plot, hue="Feature", palette="viridis", legend=False)
        plt.title(f"Top 30 Features by IC (Target: {target_label})")
        plt.axvline(0, color="k", linestyle="--")
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_dir, "feature_ic_top30.png"))
        plt.close()
        
        # 泄露预警
        suspicious = df_ic[df_ic["IC"].abs() > 0.8]
        if not suspicious.empty:
            logger.error(f"警告：发现 {len(suspicious)} 个特征 IC > 0.8，疑似未来函数泄露！请检查 feature_ic_report.csv")

    def check_correlation(self, df: pd.DataFrame):
        """4. 特征共线性检查"""
        logger.info(">>> 4. 执行特征相关性检查 (Multicollinearity)...")
        
        if not self.feat_cols:
            return

        # 为了速度和显示效果，采样 10000 行
        if len(df) > 10000:
            sample_df = df[self.feat_cols].sample(10000, random_state=42)
        else:
            sample_df = df[self.feat_cols]
            
        corr_matrix = sample_df.corr()
        
        # 保存相关性矩阵
        save_csv(corr_matrix, os.path.join(self.output_dir, "feature_corr_matrix.csv"))
        
        # 画热力图
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, cmap="coolwarm", center=0, annot=False)
        plt.title("Feature Correlation Matrix (Sampled)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_dir, "feature_correlation.png"))
        plt.close()

    def run(self):
        df = self.load_data()
        if df is not None:
            self.check_missing(df)
            self.check_label_distribution(df)
            self.check_ic(df)
            self.check_correlation(df)
            
            logger.info("=" * 40)
            logger.info(f"特征有效性分析完成！")
            logger.info(f"  - 报告: {self.output_dir}")
            logger.info(f"  - 图表: {self.figure_dir}")
            logger.info("=" * 40)

if __name__ == "__main__":
    checker = FactorChecker()
    checker.run()