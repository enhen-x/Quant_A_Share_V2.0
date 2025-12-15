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
        """3. IC 分析 (特征有效性 / 泄露检测)
        
        改进: 按日期分组计算日度 IC，然后取均值和标准差计算 IC_IR
        """
        logger.info(">>> 3. 执行 IC 分析 (Information Coefficient)...")
        
        if not self.label_cols or not self.feat_cols:
            return
            
        # 默认取第一个 Label 作为目标
        target_label = "label" if "label" in df.columns else self.label_cols[0]
        logger.info(f"当前 IC 分析目标标签: {target_label}")
        
        # 确保有日期列
        if "date" not in df.columns:
            logger.warning("未找到 date 列，使用整体相关性计算")
            self._check_ic_simple(df, target_label)
            return
        
        # 按日期分组计算日度 IC
        logger.info("使用日度分组 IC 计算方法 (更准确)...")
        
        ic_results = []
        
        for feat in self.feat_cols:
            # 计算每日的 IC (Spearman RankIC 更稳健)
            daily_ic = df.groupby("date").apply(
                lambda x: x[feat].corr(x[target_label], method="spearman") 
                if x[feat].notna().sum() > 10 else np.nan
            ).dropna()
            
            if len(daily_ic) < 30:  # 至少30天数据
                continue
            
            # 计算 IC 统计量
            ic_mean = daily_ic.mean()
            ic_std = daily_ic.std()
            ic_ir = ic_mean / ic_std if ic_std > 0 else 0  # IC 信息比率
            ic_positive_ratio = (daily_ic > 0).mean()  # IC 正比例
            
            ic_results.append({
                "Feature": feat,
                "IC_Mean": ic_mean,
                "IC_Std": ic_std,
                "IC_IR": ic_ir,
                "IC_Positive_Ratio": ic_positive_ratio,
                "Days": len(daily_ic)
            })
        
        if not ic_results:
            logger.warning("无法计算日度 IC，回退到整体相关性")
            self._check_ic_simple(df, target_label)
            return
            
        df_ic = pd.DataFrame(ic_results)
        df_ic["AbsIC"] = df_ic["IC_Mean"].abs()
        df_ic = df_ic.sort_values("AbsIC", ascending=False)
        
        # 保存 IC 报告
        save_csv(df_ic, os.path.join(self.output_dir, "feature_ic_report.csv"))
        
        # 打印报告
        print("\n" + "=" * 60)
        print("特征 IC 分析报告 (日度分组)")
        print("=" * 60)
        print(f"{'Feature':<20} {'IC_Mean':>10} {'IC_Std':>10} {'IC_IR':>8} {'正向比例':>8}")
        print("-" * 60)
        for _, row in df_ic.head(15).iterrows():
            print(f"{row['Feature']:<20} {row['IC_Mean']:>10.4f} {row['IC_Std']:>10.4f} {row['IC_IR']:>8.2f} {row['IC_Positive_Ratio']:>8.1%}")
        print("=" * 60)
        
        # 画图: IC 均值柱状图 (Top 30)
        plt.figure(figsize=(12, 8))
        df_plot = df_ic.head(min(30, len(df_ic)))
        
        colors = ["green" if x > 0 else "red" for x in df_plot["IC_Mean"]]
        
        bars = plt.barh(df_plot["Feature"], df_plot["IC_Mean"], color=colors, alpha=0.7)
        
        # 添加 IC_IR 标注
        for i, (_, row) in enumerate(df_plot.iterrows()):
            ir_text = f"IR={row['IC_IR']:.2f}"
            x_pos = row["IC_Mean"] + 0.002 if row["IC_Mean"] > 0 else row["IC_Mean"] - 0.002
            ha = 'left' if row["IC_Mean"] > 0 else 'right'
            plt.text(x_pos, i, ir_text, va='center', ha=ha, fontsize=8, color='gray')
        
        plt.axvline(0, color="k", linestyle="--", linewidth=0.8)
        plt.xlabel("IC Mean (日度 Spearman RankIC 均值)")
        plt.ylabel("Feature")
        plt.title(f"Top Features by IC (Target: {target_label})\nIC_IR = IC_Mean / IC_Std，越大越稳定")
        plt.gca().invert_yaxis()  # 最重要的在上面
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_dir, "feature_ic_top30.png"), dpi=150)
        plt.close()
        
        # 泄露预警
        suspicious = df_ic[df_ic["AbsIC"] > 0.3]
        if not suspicious.empty:
            logger.error(f"警告：发现 {len(suspicious)} 个特征 IC > 0.3，疑似未来函数泄露！")
        
        # IC 因子表现总结
        strong_factors = df_ic[(df_ic["AbsIC"] > 0.02) & (df_ic["IC_IR"].abs() > 0.5)]
        logger.info(f"有效因子数量 (|IC|>0.02 且 |IC_IR|>0.5): {len(strong_factors)}")
    
    def _check_ic_simple(self, df: pd.DataFrame, target_label: str):
        """简单 IC 计算 (整体相关性，作为后备)"""
        valid_df = df[[target_label] + self.feat_cols].dropna()
        if valid_df.empty:
            logger.warning("有效数据为空，无法计算 IC。")
            return
        
        ic_list = []
        for feat in self.feat_cols:
            ic = valid_df[feat].corr(valid_df[target_label])
            rank_ic = valid_df[feat].corr(valid_df[target_label], method="spearman")
            ic_list.append({"Feature": feat, "IC": ic, "RankIC": rank_ic})
            
        df_ic = pd.DataFrame(ic_list).sort_values(by="IC", ascending=False)
        save_csv(df_ic, os.path.join(self.output_dir, "feature_ic_report.csv"))
        
        # 画图
        plt.figure(figsize=(10, 8))
        df_plot = df_ic.copy()
        df_plot["AbsIC"] = df_plot["IC"].abs()
        df_plot = df_plot.sort_values("AbsIC", ascending=False).head(30)
        
        sns.barplot(x="IC", y="Feature", data=df_plot, hue="Feature", palette="viridis", legend=False)
        plt.title(f"Top 30 Features by IC (Target: {target_label})")
        plt.axvline(0, color="k", linestyle="--")
        plt.tight_layout()
        plt.savefig(os.path.join(self.figure_dir, "feature_ic_top30.png"))
        plt.close()

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