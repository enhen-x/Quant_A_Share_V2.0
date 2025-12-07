# src/preprocessing/labels.py

import pandas as pd
import numpy as np
from src.utils.logger import get_logger
from src.data_source.datahub import DataHub

logger = get_logger()

class LabelGenerator:
    """
    标签工厂：负责生成训练目标 (Label)
    
    [实战化改进 v2.0]
    1. 引入 VWAP (均价) 代替 Close，减少尾盘脉冲噪声。
    2. 调整收益率计算时序：基于 T+1 日入场，而非 T 日。
    3. 剔除 "一字涨停" (One-word Limit Up) 样本，防止学习到无法交易的虚假利润。
    """
    
    def __init__(self, config: dict):
        self.cfg = config.get("preprocessing", {}).get("labels", {})
        self.horizon = self.cfg.get("horizon", 5) # 预测周期 (持有天数)
        self.return_mode = self.cfg.get("return_mode", "excess_index")
        
        # 新增配置项 (可写入 main.yaml，也可使用默认值)
        # use_vwap: 是否使用均价计算收益 (推荐 True)
        self.use_vwap = self.cfg.get("use_vwap", True) 
        # filter_limit: 是否剔除 T+1 一字涨停无法买入的样本 (推荐 True)
        self.filter_limit = self.cfg.get("filter_limit", True)
        
        # 加载指数数据 (如果需要计算超额收益)
        self.df_index = None
        if self.return_mode == "excess_index":
            self._load_index(config)
            
    def _load_index(self, config):
        """预加载指数数据"""
        index_code = self.cfg.get("index_code", "000300.SH")
        # 直接读取 raw 数据以提高速度
        raw_dir = config["paths"]["data_raw"]
        import os
        from src.utils.io import read_parquet
        
        index_file = f"index_{index_code.replace('.', '')}.parquet"
        path = os.path.join(raw_dir, index_file)
        
        if os.path.exists(path):
            df = read_parquet(path)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").set_index("date")
            
            # --- 指数收益率计算逻辑同步更新 ---
            # 如果个股用 T+1 VWAP，指数也应该尽量匹配该时段。
            # 但指数通常没有 Amount/Volume 数据(或不准)，且指数无法交易，
            # 这里的基准仍然沿用 Close(T+1 -> T+1+N) 的逻辑比较稳妥。
            
            # 这里的 shift(-1) 代表 T+1 日，shift(-(1+horizon)) 代表 T+1+N 日
            s_next = df["close"].shift(-1)
            s_future = df["close"].shift(-(1 + self.horizon))
            
            df["idx_ret_future"] = s_future / s_next - 1
            
            self.df_index = df
            logger.info(f"标签生成器已加载指数数据: {index_code}")
        else:
            logger.warning(f"未找到指数数据 {path}，将降级为绝对收益模式。")
            self.return_mode = "absolute"

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        输入带有 date 的 DF，输出带有 label 列的 DF
        """
        if df is None or df.empty:
            return df
        
        # ----------------------------------------------------------------------
        # 1. 价格基准选择 (VWAP vs Close)
        # ----------------------------------------------------------------------
        if self.use_vwap and "amount" in df.columns and "volume" in df.columns:
            # VWAP = 成交额 / 成交量
            # 处理 volume = 0 (停牌) 的情况，避免除零
            vwap = df["amount"] / (df["volume"] + 1e-8)
            # 极端情况下(如数据缺失)用 Close 填充
            price_series = vwap.where(df["volume"] > 0, df["close"])
        else:
            price_series = df["close"]

        # ----------------------------------------------------------------------
        # 2. 计算未来收益 (基于 T+1 日进场)
        # ----------------------------------------------------------------------
        # 假设：
        # T 日: 我们在收盘后计算出信号
        # T+1 日: 我们按均价(VWAP)或开盘价买入 (Entry)
        # T+1+N 日: 我们按均价卖出 (Exit)
        
        entry_price = price_series.shift(-1)
        exit_price = price_series.shift(-(1 + self.horizon))
        
        df["stock_ret_future"] = (exit_price / entry_price) - 1.0

        # ----------------------------------------------------------------------
        # 3. 核心风控：剔除 "一字涨停" (One-word Limit Up) 样本
        # ----------------------------------------------------------------------
        if self.filter_limit:
            # 判定 T+1 日是否一字板
            # 条件：T+1 High == T+1 Low (全天价格没变过) 且 T+1 Close > T Close (是涨的)
            # 并且涨幅超过一定阈值 (比如 9.5%，排除掉横盘不动的情况)
            
            next_high = df["high"].shift(-1)
            next_low = df["low"].shift(-1)
            next_close = df["close"].shift(-1)
            curr_close = df["close"]
            
            # 判断逻辑：
            # 1. High == Low (一字)
            # 2. (NextClose / CurrClose - 1) > 0.09 (涨停) 
            #    注：科创板是20%，但 >9% 已经足够覆盖主板的一字板风险，且科创板若涨9%全天不动也很难买
            is_limit_up_entry = (next_high == next_low) & (next_close > curr_close * 1.09)
            
            # 将无法买入的日期的 Label 设为 NaN
            # 在 Pipeline 中，这些行会被 dropna() 自动剔除，不进入训练集
            mask_limit = is_limit_up_entry
            if mask_limit.any():
                df.loc[mask_limit, "stock_ret_future"] = np.nan
                # (可选) 打印日志调试
                # logger.debug(f"剔除 {mask_limit.sum()} 个一字涨停无法买入样本")

        # ----------------------------------------------------------------------
        # 4. 生成最终 Label (超额 vs 绝对)
        # ----------------------------------------------------------------------
        label_col = f"label_{self.horizon}d"
        
        if self.return_mode == "excess_index" and self.df_index is not None:
            # 对齐指数收益
            df = df.set_index("date")
            common_idx = df.index.intersection(self.df_index.index)
            
            # 取出指数在 T+1 -> T+1+N 的收益
            idx_ret = self.df_index.loc[common_idx, "idx_ret_future"]
            
            # 计算超额
            df.loc[common_idx, "idx_ret_future"] = idx_ret
            df[label_col] = df["stock_ret_future"] - df["idx_ret_future"]
            
            df = df.reset_index()
        else:
            # 绝对收益
            df[label_col] = df["stock_ret_future"]

        # 统一输出名为 label 的列
        df["label"] = df[label_col]
        
        # 清理中间列 (可选，保留以便 debug)
        # df = df.drop(columns=["stock_ret_future", "idx_ret_future"], errors="ignore")
        
        return df