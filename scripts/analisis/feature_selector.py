# scripts/analisis/feature_selector.py
"""
特征筛选脚本
基于 IC 分析和模型特征重要性进行特征筛选

使用方法:
    python scripts/analisis/feature_selector.py
"""

import sys
import os
import pandas as pd
import numpy as np

# 路径适配
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.config import GLOBAL_CONFIG
from src.utils.logger import get_logger
from src.utils.io import read_parquet, save_parquet

logger = get_logger()


class FeatureSelector:
    """
    特征筛选器
    
    筛选策略:
    1. 基于 IC 绝对值筛选 (|IC| > threshold)
    2. 移除高度相关的冗余特征 (corr > 0.9)
    3. 结合模型特征重要性进行交叉验证
    """
    
    def __init__(self, ic_threshold: float = 0.01, corr_threshold: float = 0.9):
        """
        Args:
            ic_threshold: IC 绝对值阈值，低于此值的特征将被标记为低效
            corr_threshold: 相关性阈值，高于此值的特征对将被标记为冗余
        """
        self.config = GLOBAL_CONFIG
        self.paths = self.config["paths"]
        self.ic_threshold = ic_threshold
        self.corr_threshold = corr_threshold
        
        # 加载数据路径
        self.data_path = os.path.join(self.paths["data_processed"], "all_stocks.parquet")
        
    def load_data(self) -> pd.DataFrame:
        """加载特征数据"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"数据文件不存在: {self.data_path}")
        
        logger.info(f"加载数据: {self.data_path}")
        df = read_parquet(self.data_path)
        return df
    
    def get_feature_cols(self, df: pd.DataFrame) -> list:
        """获取所有特征列"""
        return [c for c in df.columns if c.startswith("feat_")]
    
    def analyze_ic(self, df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
        """
        计算各特征的 IC (Information Coefficient)
        
        Returns:
            DataFrame with columns: Feature, IC, AbsIC, RankIC
        """
        feat_cols = self.get_feature_cols(df)
        
        if label_col not in df.columns:
            # 尝试找其他标签列
            label_cols = [c for c in df.columns if c.startswith("label")]
            if label_cols:
                label_col = label_cols[0]
            else:
                raise ValueError("未找到标签列")
        
        logger.info(f"计算 IC，目标标签: {label_col}")
        
        # 计算 IC (Pearson 相关系数)
        valid_df = df[feat_cols + [label_col]].dropna()
        
        ic_results = []
        for feat in feat_cols:
            ic = valid_df[feat].corr(valid_df[label_col])
            rank_ic = valid_df[feat].corr(valid_df[label_col], method="spearman")
            ic_results.append({
                "Feature": feat,
                "IC": ic,
                "AbsIC": abs(ic),
                "RankIC": rank_ic
            })
        
        df_ic = pd.DataFrame(ic_results)
        df_ic = df_ic.sort_values("AbsIC", ascending=False)
        
        return df_ic
    
    def analyze_correlation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算特征间相关性矩阵
        """
        feat_cols = self.get_feature_cols(df)
        logger.info(f"计算特征相关性矩阵 ({len(feat_cols)} 个特征)")
        
        corr_matrix = df[feat_cols].corr()
        return corr_matrix
    
    def find_redundant_features(self, corr_matrix: pd.DataFrame, 
                                 feature_ranking: list) -> list:
        """
        找出冗余特征（高度相关且排名较低的）
        
        Args:
            corr_matrix: 相关性矩阵
            feature_ranking: 按重要性排序的特征列表
            
        Returns:
            需要移除的冗余特征列表
        """
        redundant = set()
        checked = set()
        
        for i, feat1 in enumerate(feature_ranking):
            if feat1 in redundant:
                continue
                
            for feat2 in feature_ranking[i+1:]:
                if feat2 in redundant or feat2 in checked:
                    continue
                    
                if feat1 in corr_matrix.columns and feat2 in corr_matrix.columns:
                    corr = abs(corr_matrix.loc[feat1, feat2])
                    if corr > self.corr_threshold:
                        # feat2 排名更低，标记为冗余
                        redundant.add(feat2)
                        logger.info(f"  冗余特征: {feat2} (与 {feat1} 相关性 {corr:.3f})")
            
            checked.add(feat1)
        
        return list(redundant)
    
    def select_features(self, df: pd.DataFrame) -> dict:
        """
        执行特征筛选
        
        Returns:
            {
                "selected": [...],  # 选中的特征
                "removed_low_ic": [...],  # 因 IC 过低被移除
                "removed_redundant": [...],  # 因冗余被移除
            }
        """
        logger.info("=" * 50)
        logger.info("开始特征筛选分析")
        logger.info("=" * 50)
        
        # 1. 计算 IC
        df_ic = self.analyze_ic(df)
        
        # 2. 筛除低 IC 特征
        low_ic_features = df_ic[df_ic["AbsIC"] < self.ic_threshold]["Feature"].tolist()
        high_ic_features = df_ic[df_ic["AbsIC"] >= self.ic_threshold]["Feature"].tolist()
        
        logger.info(f"\n[Step 1] IC 筛选 (阈值: |IC| >= {self.ic_threshold})")
        logger.info(f"  - 有效特征: {len(high_ic_features)} 个")
        logger.info(f"  - 低效特征: {len(low_ic_features)} 个")
        
        if low_ic_features:
            logger.info(f"  - 低效特征列表: {low_ic_features}")
        
        # 3. 计算相关性并移除冗余
        corr_matrix = self.analyze_correlation(df)
        
        # 按 IC 排名作为重要性排序
        feature_ranking = df_ic[df_ic["AbsIC"] >= self.ic_threshold]["Feature"].tolist()
        
        logger.info(f"\n[Step 2] 冗余特征筛选 (阈值: |corr| > {self.corr_threshold})")
        redundant_features = self.find_redundant_features(corr_matrix, feature_ranking)
        logger.info(f"  - 冗余特征: {len(redundant_features)} 个")
        
        # 4. 最终选择
        selected = [f for f in high_ic_features if f not in redundant_features]
        
        logger.info(f"\n[结果] 特征筛选完成")
        logger.info(f"  - 原始特征数: {len(self.get_feature_cols(df))}")
        logger.info(f"  - 筛选后特征数: {len(selected)}")
        logger.info(f"  - 移除低 IC 特征: {len(low_ic_features)}")
        logger.info(f"  - 移除冗余特征: {len(redundant_features)}")
        
        return {
            "selected": selected,
            "removed_low_ic": low_ic_features,
            "removed_redundant": redundant_features,
            "ic_report": df_ic
        }
    
    def save_feature_config(self, selected_features: list, output_path: str = None):
        """
        保存特征选择配置
        """
        if output_path is None:
            output_path = os.path.join(self.paths["data_processed"], "selected_features.txt")
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# 筛选后的特征列表\n")
            f.write(f"# 生成时间: {pd.Timestamp.now()}\n")
            f.write(f"# 特征数量: {len(selected_features)}\n\n")
            for feat in selected_features:
                f.write(f"{feat}\n")
        
        logger.info(f"特征配置已保存: {output_path}")
        return output_path
    
    def run(self) -> dict:
        """
        执行完整的特征筛选流程
        """
        df = self.load_data()
        result = self.select_features(df)
        
        # 保存配置
        config_path = self.save_feature_config(result["selected"])
        result["config_path"] = config_path
        
        # 打印 IC 报告摘要
        print("\n" + "=" * 60)
        print("特征 IC 排名 (Top 20)")
        print("=" * 60)
        print(result["ic_report"].head(20).to_string(index=False))
        
        return result


def main():
    """主函数"""
    print(">>> 开始特征筛选分析...")
    
    try:
        # 可通过参数调整阈值
        selector = FeatureSelector(
            ic_threshold=0.01,      # IC 阈值
            corr_threshold=0.9       # 相关性阈值
        )
        result = selector.run()
        
        print("\n" + "=" * 60)
        print("✅ 特征筛选完成！")
        print("=" * 60)
        print(f"  - 选中特征数: {len(result['selected'])}")
        print(f"  - 配置文件: {result['config_path']}")
        print("\n下一步: 修改训练代码以使用筛选后的特征")
        
    except Exception as e:
        print(f"运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
