# scripts/run_recommendation.py

import os
import sys
import pandas as pd
import datetime

# 路径适配
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.config import GLOBAL_CONFIG
from src.utils.logger import get_logger
from src.utils.io import read_parquet
from src.model.xgb_model import XGBModelWrapper
from src.strategy.signal import TopKSignalStrategy

logger = get_logger()

def get_latest_model_path():
    """自动寻找 data/models 下最新的版本目录"""
    models_dir = GLOBAL_CONFIG["paths"]["models"]
    if not os.path.exists(models_dir):
        return None
    
    # 找子目录 (按时间戳命名)
    subdirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    if not subdirs:
        return None
    
    # 按倒序排列，取第一个
    subdirs.sort(reverse=True)
    latest_version = subdirs[0]
    model_path = os.path.join(models_dir, latest_version, "model.json")
    
    if os.path.exists(model_path):
        return latest_version, model_path
    return None, None

def load_latest_data():
    """加载特征数据，并提取出【最新一个交易日】的数据"""
    data_path = os.path.join(GLOBAL_CONFIG["paths"]["data_processed"], "all_stocks.parquet")
    if not os.path.exists(data_path):
        logger.error(f"特征文件不存在: {data_path}，请先运行 rebuild_features.py")
        return None, None

    df = read_parquet(data_path)
    
    # 获取数据中最新的日期
    latest_date = df["date"].max()
    logger.info(f"数据集中最新日期为: {latest_date}")
    
    # 筛选出最新这一天的数据
    df_latest = df[df["date"] == latest_date].copy()
    
    # 提取特征列 (feat_ 开头)
    feat_cols = [c for c in df_latest.columns if c.startswith("feat_")]
    
    return df_latest, feat_cols

def main():
    logger.info("=== 启动每日推荐系统 (Daily Recommendation) ===")

    # 1. 加载模型
    version, model_path = get_latest_model_path()
    if not model_path:
        logger.error("未找到可用模型，请先运行 train_model.py")
        return
    
    logger.info(f"使用模型版本: {version}")
    model = XGBModelWrapper()
    model.load(model_path)
    
    # 2. 加载数据
    df_latest, feat_cols = load_latest_data()
    if df_latest is None or df_latest.empty:
        logger.error("今日无数据，无法推荐。")
        return

    # 3. 执行预测
    logger.info(f"正在对 {len(df_latest)} 只股票进行打分...")
    # 注意：这里直接预测，不需要 Label
    pred_scores = model.predict(df_latest[feat_cols])
    
    # 构造预测结果 DataFrame，格式需满足 signal.py 的要求
    pred_df = df_latest[["date", "symbol"]].copy()
    pred_df["pred_score"] = pred_scores
    
    # 4. 策略筛选 (应用风控：剔除ST、低价、流动性差、得分低)
    strategy = TopKSignalStrategy()
    # signal.py 内部会读取 all_stocks.parquet 再做一次合并过滤，
    # 既然我们已经有了 df_latest，其实可以直接传，但为了复用 signal 逻辑，
    # 我们按标准流程传入 pred_df
    recommend_df = strategy.generate(pred_df)
    
    # 5. 输出结果
    if recommend_df.empty:
        logger.warning("策略筛选后无股票入选 (可能都被风控剔除或分数不足)。")
        return

    # 补充股票名称以便阅读
    meta_path = os.path.join(GLOBAL_CONFIG["paths"]["data_meta"], "all_stocks_meta.parquet")
    if os.path.exists(meta_path):
        df_meta = read_parquet(meta_path)
        recommend_df = pd.merge(recommend_df, df_meta[["symbol", "name"]], on="symbol", how="left")
    
    # 补充最新收盘价和预测分 (从 pred_df 拿回来)
    recommend_df = pd.merge(recommend_df, pred_df[["symbol", "pred_score"]], on="symbol", how="left")
    
    # 格式化输出
    print("\n" + "="*50)
    print(f"🌟 {df_latest['date'].iloc[0].strftime('%Y-%m-%d')} 每日精选推荐 (Top {len(recommend_df)}) 🌟")
    print("="*50)
    
    # 调整列顺序
    cols = ["symbol", "name", "pred_score", "weight"]
    print_df = recommend_df[cols].sort_values("pred_score", ascending=False).reset_index(drop=True)
    
    # 打印表格
    print(print_df.to_markdown(index=True, floatfmt=".4f") if hasattr(print_df, "to_markdown") else print_df)
    
    # 保存推荐列表到本地
    out_dir = os.path.join(GLOBAL_CONFIG["paths"]["reports"], "daily_picks")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_file = os.path.join(out_dir, f"picks_{version}_{df_latest['date'].iloc[0].strftime('%Y%m%d')}.csv")
    print_df.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"\n推荐列表已保存至: {out_file}")
    print("="*50)

if __name__ == "__main__":
    main()