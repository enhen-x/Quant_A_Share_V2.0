# scripts/run_recommendation.py

import os
import sys
import pandas as pd
import datetime
import glob
import re

# 路径适配
current_dir = os.path.dirname(os.path.abspath(__file__))
# 从当前文件位置 (scripts/back_test) 返回两级到项目根目录
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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
    """加载特征数据，并提取出【最近 N 个交易日】的数据，用于预测和平滑。"""
    data_path = os.path.join(GLOBAL_CONFIG["paths"]["data_processed"], "all_stocks.parquet")
    if not os.path.exists(data_path):
        logger.error(f"特征文件不存在: {data_path}，请先运行 rebuild_features.py")
        return None, None, None

    # 读取数据 (实盘可优化为只读最后的分区)
    df = read_parquet(data_path)
    df["date"] = pd.to_datetime(df["date"])
    
    # 提取特征列 (feat_ 开头)
    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    
    # 获取最新的 N 个交易日的数据 (N=3, 与 signal.py 中的 SMOOTH_WINDOW 匹配)
    N_DAYS = 3 
    
    # 1. 获取唯一的日期并排序
    unique_dates = sorted(df["date"].unique(), reverse=True)
    
    if len(unique_dates) < N_DAYS:
        logger.warning(f"总交易日 ({len(unique_dates)}) 少于平滑窗口 ({N_DAYS}天)，使用全部数据。")
        target_dates = unique_dates
    else:
        # 取最近的 N 个交易日
        target_dates = unique_dates[:N_DAYS]
    
    df_slice = df[df["date"].isin(target_dates)].copy()
    
    if df_slice.empty:
        logger.error("数据切片为空，无法推荐。")
        return None, None, None
    
    latest_date = unique_dates[0]
    logger.info(f"数据集中最新日期为: {latest_date.strftime('%Y-%m-%d')}，将加载前 {len(target_dates)} 个交易日的数据。")

    # 返回切片数据、特征列表、最新日期
    return df_slice, feat_cols, latest_date

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
    
    # 2. 加载最新行情数据（最近 N 天）
    df_slice, feat_cols, latest_date = load_latest_data()
    if df_slice is None or df_slice.empty:
        logger.error("无数据切片，无法推荐。")
        return

    # 3. 执行预测
    logger.info(f"正在对 {len(df_slice)} 行数据 ({df_slice['symbol'].nunique()} 只股票) 进行打分...")
    
    # 3.1 特征对齐
    # 优先使用模型记录的特征名 (如果有)
    final_features = feat_cols
    if hasattr(model.model, "feature_names") and model.model.feature_names:
        model_features = model.model.feature_names
        logger.info(f"使用模型内置特征列表: {len(model_features)} 个")
        
        # 检查缺失特征
        missing = [f for f in model_features if f not in df_slice.columns]
        if missing:
            logger.error(f"严重错误：数据中缺少模型所需的特征: {missing}")
            logger.error("这通常是由于特征工程配置 (rebuild_features.py) 与模型训练时的配置不一致导致的。")
            return
            
        final_features = model_features
    else:
        logger.warning(f"模型未记录特征名，将使用所有 {len(final_features)} 个 'feat_' 开头的列。可能会导致 mismatch 错误。")

    # 3.2 预测分数
    try:
        # 确保列顺序与模型一致
        X_pred = df_slice[final_features]
        pred_scores = model.predict(X_pred)
    except Exception as e:
        logger.error(f"预测失败: {e}")
        return
    
    # 构造包含历史预测的 DataFrame (用于策略计算平滑分)
    pred_df = df_slice[["date", "symbol"]].copy()
    pred_df["pred_score"] = pred_scores
    
    # =======================================================
    # 4. 策略筛选 (读取推荐专用 Top-K 配置)
    # =======================================================
    
    strat_cfg = GLOBAL_CONFIG["strategy"]
    rec_k = strat_cfg.get("recommend_top_k", strat_cfg.get("top_k", 5))
    
    logger.info(f"生成推荐列表长度: {rec_k} (含备选)")
    
    # 实例化策略时传入 top_k
    strategy = TopKSignalStrategy(top_k=rec_k)
    
    # **关键：传递包含历史数据的 pred_df，以便 strategy.generate 计算平滑得分**
    # 注意：需确保 src/strategy/signal.py 已修改为返回包含 pos_ratio 的列
    recommend_df = strategy.generate(pred_df)
    
    # 筛选出最新的信号（即今天）
    recommend_df_latest = recommend_df[recommend_df["date"] == latest_date].copy()
    
    # 5. 输出结果
    if recommend_df_latest.empty:
        logger.warning("策略筛选后无股票入选 (可能都被风控剔除或分数不足)。")
        logger.info("Top 5 原始预测得分 (未经过滤):")
        print(pred_df[pred_df["date"] == latest_date].sort_values("pred_score", ascending=False).head(5))
        return

    # === [新增] 获取风控仓位系数 ===
    current_pos_ratio = 1.0
    if "pos_ratio" in recommend_df_latest.columns:
        # 获取当天的风控系数 (所有股票同一天系数相同)
        current_pos_ratio = recommend_df_latest["pos_ratio"].iloc[0]

    # 补充股票名称以便阅读
    meta_path = os.path.join(GLOBAL_CONFIG["paths"]["data_meta"], "all_stocks_meta.parquet")
    if os.path.exists(meta_path):
        df_meta = read_parquet(meta_path)
        recommend_df_latest = pd.merge(recommend_df_latest, df_meta[["symbol", "name"]], on="symbol", how="left")
    
    # 补充原始预测分
    recommend_df_latest = pd.merge(recommend_df_latest, 
                                   pred_df[["date", "symbol", "pred_score"]], 
                                   on=["date", "symbol"], how="left")
    
    # 格式化输出
    print("\n" + "="*70)
    print(f"🌟 {latest_date.strftime('%Y-%m-%d')} 每日精选推荐 (Top {len(recommend_df_latest)}) 🌟")
    
    # === [新增] 显式打印风控状态 ===
    print("-" * 70)
    print(f"🛡️  风控系统建议总仓位: {current_pos_ratio * 100:.0f}%")
    if current_pos_ratio < 1.0:
        if current_pos_ratio == 0.0:
            print("⚠️  [极高风险] 大盘处于熊市阶段，策略建议空仓观望！(列表中股票仅供跟踪研究)")
        else:
            print(f"⚠️  [风险提示] 大盘处于震荡/回调阶段，建议降低仓位至 {current_pos_ratio * 100:.0f}%")
    else:
        print("✅  [积极信号] 市场趋势良好，建议正常仓位操作。")
    print("-" * 70)
    
    # [修改] 输出列中加入 pos_ratio
    cols = ["symbol", "name", "pred_score", "pos_ratio", "weight"]
    print_cols = [c for c in cols if c in recommend_df_latest.columns]
    
    print_df = recommend_df_latest[print_cols].sort_values("pred_score", ascending=False).reset_index(drop=True)
    
    # 尝试使用 tabulate 美化输出
    try:
        # floatfmt 控制小数位数，让 pred_score 和 weight 显示更清晰
        print(print_df.to_markdown(index=True, floatfmt=".4f"))
    except:
        print(print_df)
    
    # 保存结果
    out_dir = os.path.join(GLOBAL_CONFIG["paths"]["reports"], "daily_picks")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # 使用 latest_date 作为文件名日期
    out_file = os.path.join(out_dir, f"picks_{version}_{latest_date.strftime('%Y%m%d')}.csv")
    print_df.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"\n[文件] 推荐列表已保存至: {out_file}")
    print("="*70)

if __name__ == "__main__":
    main()