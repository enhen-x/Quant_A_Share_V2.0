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

        # === [新增] 8. 变化率/斜率特征 (Slope) ===
        if self.cfg.get("enable_slope", True):
            df = self._add_slope_features(df)

        # === [新增] 9. 波动率特征 (Volatility) ===
        if self.cfg.get("enable_volatility", True):
            window = self.cfg.get("vol_window", 20)
            df = self._add_volatility_features(df, window)

        # === [新增] 10. 量价相关性 (PV Corr) ===
        if self.cfg.get("enable_pv_corr", True):
            window = self.cfg.get("vol_window", 20)
            df = self._add_pv_corr_features(df, window)

        # 市值特征 (保持不变)
        if "amount" in df.columns and "turnover" in df.columns:
            mcap = df["amount"] / (df["turnover"].replace(0, np.nan) * 0.01)
            df["feat_mcap_log"] = np.log1p(mcap)

        return df

    # ... (原有方法 _add_basic_features 到 _add_volume_features 保持不变) ...
    def _add_basic_features(self, df):
        df["feat_log_ret"] = np.log(df["close"] / df["close"].shift(1))
        df["feat_amplitude"] = (df["high"] - df["low"]) / df["close"].shift(1)
        return df

    def _add_ma_features(self, df, windows):
        for w in windows:
            ma = df["close"].rolling(window=w).mean()
            df[f"feat_ma_{w}_bias"] = df["close"] / ma - 1.0
        return df

    def _add_macd_features(self, df):
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        macd = (dif - dea) * 2
        df["feat_macd_dif"] = dif / df["close"]
        # dea 与 dif 高度相关，可以选择性保留或做差分
        # df["feat_macd_dea"] = dea / df["close"] 
        df["feat_macd"] = macd / df["close"]
        return df

    def _add_rsi_features(self, df, window):
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        df[f"feat_rsi_{window}"] = (100 - (100 / (1 + rs))) / 100.0
        return df
    
    def _add_kdj_features(self, df, window):
        low_list = df["low"].rolling(window=window).min()
        high_list = df["high"].rolling(window=window).max()
        rsv = (df["close"] - low_list) / (high_list - low_list + 1e-9)
        df[f"feat_kdj_k"] = rsv.ewm(alpha=1/3, adjust=False).mean()
        df[f"feat_kdj_d"] = df[f"feat_kdj_k"].ewm(alpha=1/3, adjust=False).mean()
        df[f"feat_kdj_j"] = 3 * df[f"feat_kdj_k"] - 2 * df[f"feat_kdj_d"]
        return df

    def _add_boll_features(self, df, window, std_dev):
        ma = df["close"].rolling(window=window).mean()
        std = df["close"].rolling(window=window).std()
        upper = ma + std_dev * std
        lower = ma - std_dev * std
        df["feat_boll_pos"] = (df["close"] - lower) / (upper - lower + 1e-9)
        df["feat_boll_width"] = (upper - lower) / ma
        return df

    def _add_volume_features(self, df):
        ma5_vol = df["volume"].rolling(window=5).mean()
        df["feat_vol_ratio_5"] = df["volume"] / (ma5_vol + 1e-9)
        if "turnover" in df.columns:
            df["feat_turnover_log"] = np.log1p(df["turnover"])
        return df

    # === [新增方法实现] ===

    def _add_slope_features(self, df):
        """
        计算指标的变化率 (Slope/Delta)。
        模型通常无法从单点值判断趋势方向，加入斜率特征可以弥补这一缺陷。
        """
        # 1. RSI 斜率
        if "feat_rsi_14" in df.columns:
            # 3日变动值
            df["feat_rsi_14_slope"] = df["feat_rsi_14"] - df["feat_rsi_14"].shift(3)
            
        # 2. MACD 柱变化 (红绿柱是在变长还是变短)
        if "feat_macd" in df.columns:
            df["feat_macd_slope"] = df["feat_macd"] - df["feat_macd"].shift(1)
            
        # 3. 均线角度 (使用 5日线)
        # 用 atan 计算弧度，归一化到 -1~1
        ma5 = df["close"].rolling(5).mean()
        ma5_slope = (ma5 - ma5.shift(1)) / ma5.shift(1) * 100
        df["feat_ma_5_angle"] = np.arctan(ma5_slope)
        
        return df

    def _add_volatility_features(self, df, window):
        """
        计算波动率特征 (ATR, Std)。
        用于衡量个股风险和活跃度。
        """
        # 1. 归一化 ATR (Average True Range)
        # TR = Max(H-L, |H-Cp|, |L-Cp|)
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window).mean()
        
        # 归一化：ATR / Price (类似变异系数)
        df["feat_atr_norm"] = atr / df["close"]
        
        # 2. 收益率标准差 (历史波动率)
        if "feat_log_ret" in df.columns:
            df[f"feat_std_{window}"] = df["feat_log_ret"].rolling(window).std()
            
        return df

    def _add_pv_corr_features(self, df, window):
        """
        计算量价相关性 (Price-Volume Correlation)。
        corr > 0: 放量上涨或缩量下跌 (趋势确认)
        corr < 0: 放量下跌或缩量上涨 (背离/出货)
        """
        # 使用 pct_change 序列计算相关性
        pct_close = df["close"].pct_change()
        pct_vol = df["volume"].pct_change()
        
        # 计算滚动相关系数
        df[f"feat_pv_corr_{window}"] = pct_close.rolling(window).corr(pct_vol)
        
        # 填充一下初始 NaN，防止被 drop 太多
        df[f"feat_pv_corr_{window}"] = df[f"feat_pv_corr_{window}"].fillna(0)
        
        return df