# scripts/run_recommendation.py

import os
import sys
import pandas as pd
import datetime
import glob
import re

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
    """
    智能寻找 data/models 下最新的模型文件
    支持识别普通训练目录和 WF (滚动训练) 目录
    """
    models_dir = GLOBAL_CONFIG["paths"]["models"]
    if not os.path.exists(models_dir):
        return None, None
    
    # 1. 获取所有子目录
    subdirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
    if not subdirs:
        return None, None

    # 2. 辅助函数：解析目录名中的时间戳
    def parse_timestamp(dir_name):
        # 移除前缀 (如 "WF_")
        clean_name = dir_name.replace("WF_", "")
        # 尝试匹配 YYYYMMDD_HHMMSS 格式
        try:
            return datetime.datetime.strptime(clean_name, "%Y%m%d_%H%M%S")
        except ValueError:
            # 如果格式不对，返回一个极小时间，排在最后
            return datetime.datetime.min

    # 3. 按时间倒序排列 (最新的在前)
    subdirs.sort(key=parse_timestamp, reverse=True)
    latest_version = subdirs[0]
    version_dir = os.path.join(models_dir, latest_version)
    
    logger.info(f"锁定最新模型版本目录: {latest_version}")

    # 4. 在目录中寻找最佳模型文件
    # 情况 A: 单次训练的标准模型
    if os.path.exists(os.path.join(version_dir, "model.json")):
        return latest_version, os.path.join(version_dir, "model.json")
    
    # 情况 B: 滚动训练的年度模型 (model_2024.json, model_2025.json ...)
    # 我们需要找到年份最大的那个，因为它包含了最新的市场规律
    wf_models = glob.glob(os.path.join(version_dir, "model_*.json"))
    if wf_models:
        def extract_year(path):
            fname = os.path.basename(path)
            # 提取数字部分
            match = re.search(r"model_(\d+)\.json", fname)
            return int(match.group(1)) if match else 0
        
        # 找年份最大的
        best_model_path = max(wf_models, key=extract_year)
        best_year = extract_year(best_model_path)
        logger.info(f"检测到滚动训练模型集，已自动选择最新年份: model_{best_year}.json")
        return latest_version, best_model_path

    return None, None

def load_latest_data():
    """加载特征数据，并提取出【最新一个交易日】的数据"""
    data_path = os.path.join(GLOBAL_CONFIG["paths"]["data_processed"], "all_stocks.parquet")
    if not os.path.exists(data_path):
        logger.error(f"特征文件不存在: {data_path}，请先运行 rebuild_features.py")
        return None, None

    # 读取数据 (实盘可优化为只读最后的分区)
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

    # 1. 智能加载模型
    version, model_path = get_latest_model_path()
    if not model_path:
        logger.error("未找到可用模型文件，请先运行 run_walkforward.py 或 train_model.py")
        return
    
    logger.info(f"加载模型文件: {model_path}")
    model = XGBModelWrapper()
    model.load(model_path)
    
    # 2. 加载最新行情数据
    df_latest, feat_cols = load_latest_data()
    if df_latest is None or df_latest.empty:
        logger.error("今日无数据，无法推荐。")
        return

    # 3. 执行预测
    logger.info(f"正在对 {len(df_latest)} 只股票进行打分...")
    pred_scores = model.predict(df_latest[feat_cols])
    
    pred_df = df_latest[["date", "symbol"]].copy()
    pred_df["pred_score"] = pred_scores
    
    # =======================================================
    # 4. 策略筛选 (读取推荐专用 Top-K 配置)
    # =======================================================
    
    # 优先读取 recommend_top_k，如果没有则回退到 top_k
    strat_cfg = GLOBAL_CONFIG["strategy"]
    rec_k = strat_cfg.get("recommend_top_k", strat_cfg.get("top_k", 5))
    
    logger.info(f"生成推荐列表长度: {rec_k} (含备选)")
    
    # 实例化策略时传入 top_k
    # 注意：需确保 src/strategy/signal.py 的 __init__ 已支持 top_k 参数
    strategy = TopKSignalStrategy(top_k=rec_k)
    recommend_df = strategy.generate(pred_df)
    
    # 5. 输出结果
    if recommend_df.empty:
        logger.warning("策略筛选后无股票入选 (可能都被风控剔除或分数不足)。")
        logger.info("Top 5 原始预测得分 (未经过滤):")
        print(pred_df.sort_values("pred_score", ascending=False).head(5))
        return

    # 补充股票名称以便阅读
    meta_path = os.path.join(GLOBAL_CONFIG["paths"]["data_meta"], "all_stocks_meta.parquet")
    if os.path.exists(meta_path):
        df_meta = read_parquet(meta_path)
        recommend_df = pd.merge(recommend_df, df_meta[["symbol", "name"]], on="symbol", how="left")
    
    # 补充预测分
    recommend_df = pd.merge(recommend_df, pred_df[["symbol", "pred_score"]], on="symbol", how="left")
    
    # 格式化输出
    print("\n" + "="*60)
    print(f"🌟 {df_latest['date'].iloc[0].strftime('%Y-%m-%d')} 每日精选推荐 (Top {len(recommend_df)}) 🌟")
    print("="*60)
    
    cols = ["symbol", "name", "pred_score", "weight"]
    print_cols = [c for c in cols if c in recommend_df.columns]
    
    print_df = recommend_df[print_cols].sort_values("pred_score", ascending=False).reset_index(drop=True)
    
    # 尝试使用 tabulate 美化输出
    try:
        print(print_df.to_markdown(index=True, floatfmt=".4f"))
    except:
        print(print_df)
    
    # 保存结果
    out_dir = os.path.join(GLOBAL_CONFIG["paths"]["reports"], "daily_picks")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_file = os.path.join(out_dir, f"picks_{version}_{df_latest['date'].iloc[0].strftime('%Y%m%d')}.csv")
    print_df.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"\n[文件] 推荐列表已保存至: {out_file}")
    print("="*60)

if __name__ == "__main__":
    main()