# src/preprocessing/neutralization.py

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from src.utils.logger import get_logger

logger = get_logger()

class FeatureNeutralizer:
    def __init__(self, config):
        self.cfg = config.get("preprocessing", {}).get("neutralization", {})
        self.enabled = self.cfg.get("enable", False)
        self.risk_factors = self.cfg.get("risk_factors", ["feat_mcap_log"])
        self.target_features = self.cfg.get("features_to_neutralize", "all")
        
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        执行特征中性化
        :param df: 全量数据 (包含 date, symbol, features)
        :return: 中性化后的 df
        """
        if not self.enabled:
            return df
            
        logger.info(f"=== 开始特征中性化 (Risk Factors: {self.risk_factors}) ===")
        
        # 1. 确定要处理的特征列
        all_cols = df.columns
        if self.target_features == "all":
            # 排除非特征列和风险因子本身
            feat_cols = [
                c for c in all_cols 
                if c.startswith("feat_") and c not in self.risk_factors
            ]
        else:
            feat_cols = [c for c in self.target_features if c in all_cols]
            
        if not feat_cols:
            logger.warning("未找到需要中性化的特征列，跳过。")
            return df

        # 检查风险因子是否存在
        missing_risks = [c for c in self.risk_factors if c not in df.columns]
        if missing_risks:
            logger.error(f"缺失风险因子列: {missing_risks}，无法进行中性化。")
            return df

        # 2. 按日期分组进行横截面回归
        # 这是一个 GroupBy Apply 操作，为了性能，我们显式循环并显示进度条
        dates = df["date"].unique()
        dates = np.sort(dates)
        
        neutralized_dfs = []
        
        # 使用 sklearn 的 LinearRegression
        model = LinearRegression(fit_intercept=True)
        
        for date in tqdm(dates, desc="Neutralizing"):
            # 获取当日切片 (Cross-Section)
            daily_mask = df["date"] == date
            daily_df = df.loc[daily_mask].copy()
            
            if len(daily_df) < 10:  # 样本太少不回归
                neutralized_dfs.append(daily_df)
                continue
                
            # 准备 X (风险因子) 和 Y (目标特征)
            # 需要处理 NaN：如果风险因子或特征有 NaN，通常填 0 或中位数，这里简单 dropna 可能导致行数对不上
            # 建议：在进入此步骤前 pipeline 已做过基础填充。这里做简单填充防崩。
            X = daily_df[self.risk_factors].fillna(0).values
            Y = daily_df[feat_cols].fillna(0).values
            
            # 拟合：Y_pred = beta * X + alpha
            model.fit(X, Y)
            Y_pred = model.predict(X)
            
            # 残差：E = Y - Y_pred
            # 这就是剔除了市值影响后的新特征
            residuals = Y - Y_pred
            
            # 更新 DataFrame
            daily_df[feat_cols] = residuals
            neutralized_dfs.append(daily_df)
            
        # 3. 合并结果
        logger.info("正在合并中性化后的数据...")
        final_df = pd.concat(neutralized_dfs, axis=0)
        
        return final_df