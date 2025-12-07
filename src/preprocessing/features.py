# src/preprocessing/features.py

import pandas as pd
import numpy as np
from src.utils.logger import get_logger

logger = get_logger()

class FeatureGenerator:
    """特征工厂：负责计算各类技术指标"""
    
    def __init__(self, config: dict):
        self.cfg = config.get("preprocessing", {}).get("features", {})
        
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """主入口：输入原始DF，返回带有特征的DF"""
        if df is None or df.empty:
            return df
            
        # 1. 基础特征
        if self.cfg.get("enable_basic", True):
            df = self._add_basic_features(df)
            
        # 2. 均线 (MA)
        if self.cfg.get("enable_ma", True):
            windows = self.cfg.get("ma_windows", [5, 10, 20, 60])
            df = self._add_ma_features(df, windows)

        # 3. MACD
        if self.cfg.get("enable_macd", True):
            df = self._add_macd_features(df)
            
        # 4. RSI
        if self.cfg.get("enable_rsi", True):
            window = self.cfg.get("rsi_window", 14)
            df = self._add_rsi_features(df, window)
            
        # 5. KDJ
        if self.cfg.get("enable_kdj", True):
            window = self.cfg.get("kdj_window", 9)
            df = self._add_kdj_features(df, window)
            
        # 6. Bollinger Bands
        if self.cfg.get("enable_boll", True):
            window = self.cfg.get("boll_window", 20)
            std = self.cfg.get("boll_std", 2)
            df = self._add_boll_features(df, window, std)

        # 7. 量能特征
        if self.cfg.get("enable_volume", True):
            df = self._add_volume_features(df)

        # [新增] 将市值作为特征
        # 注意：需要保证 upstream 传进来的 df 有 amount 和 turnover
        if "amount" in df.columns and "turnover" in df.columns:
            # 简单估算市值 (Amount / Turnover%)
            mcap = df["amount"] / (df["turnover"].replace(0, np.nan) * 0.01)
            # 对市值取 Log，使其分布更均匀
            df["feat_mcap_log"] = np.log1p(mcap)

        return df

    def _add_basic_features(self, df):
        """基础特征: 收益率, 振幅, 换手(如有)"""
        # 对数收益率
        df["feat_log_ret"] = np.log(df["close"] / df["close"].shift(1))
        # 振幅 (High-Low)/Close
        df["feat_amplitude"] = (df["high"] - df["low"]) / df["close"].shift(1)
        return df

    def _add_ma_features(self, df, windows):
        """均线偏离度"""
        for w in windows:
            ma = df["close"].rolling(window=w).mean()
            # 使用偏离度 (Price / MA - 1) 归一化，而不是绝对价格
            df[f"feat_ma_{w}_bias"] = df["close"] / ma - 1.0
        return df

    def _add_macd_features(self, df):
        """MACD (12, 26, 9)"""
        # 注意：使用 EMA 计算
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        macd = (dif - dea) * 2
        
        # 归一化处理：除以收盘价，使其对不同价位的股票可比
        df["feat_macd_dif"] = dif / df["close"]
        df["feat_macd_dea"] = dea / df["close"]
        df["feat_macd"] = macd / df["close"]
        return df

    def _add_rsi_features(self, df, window):
        """RSI 相对强弱指标"""
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        # 0-100 归一化到 0-1
        df[f"feat_rsi_{window}"] = (100 - (100 / (1 + rs))) / 100.0
        return df
    
    def _add_kdj_features(self, df, window):
        """KDJ (未成熟随机值)"""
        low_list = df["low"].rolling(window=window).min()
        high_list = df["high"].rolling(window=window).max()
        
        rsv = (df["close"] - low_list) / (high_list - low_list + 1e-9)
        
        # 简单迭代计算 K, D, J
        # 这里用 EWM 模拟 1/3 权重
        df[f"feat_kdj_k"] = rsv.ewm(alpha=1/3, adjust=False).mean()
        df[f"feat_kdj_d"] = df[f"feat_kdj_k"].ewm(alpha=1/3, adjust=False).mean()
        df[f"feat_kdj_j"] = 3 * df[f"feat_kdj_k"] - 2 * df[f"feat_kdj_d"]
        return df

    def _add_boll_features(self, df, window, std_dev):
        """布林带宽度与位置"""
        ma = df["close"].rolling(window=window).mean()
        std = df["close"].rolling(window=window).std()
        
        upper = ma + std_dev * std
        lower = ma - std_dev * std
        
        # 1. 股价在布林带的位置 (0=下轨, 0.5=中轨, 1=上轨)
        df["feat_boll_pos"] = (df["close"] - lower) / (upper - lower + 1e-9)
        # 2. 布林带宽度 (归一化)
        df["feat_boll_width"] = (upper - lower) / ma
        return df

    def _add_volume_features(self, df):
        """量能特征"""
        # 量比: 今日量 / 5日均量
        ma5_vol = df["volume"].rolling(window=5).mean()
        df["feat_vol_ratio_5"] = df["volume"] / (ma5_vol + 1e-9)
        
        # 换手率 (如果存在)
        if "turnover" in df.columns:
            # 换手率通常已经是百分比，除以 100 归一化或者保持原样
            # 这里做 Log 处理使其分布更正态
            df["feat_turnover_log"] = np.log1p(df["turnover"])

    
            
        return df