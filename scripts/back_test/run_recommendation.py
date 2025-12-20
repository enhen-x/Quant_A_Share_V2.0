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
from src.strategy.signal import TopKSignalStrategy

logger = get_logger()

def get_latest_model_path():
    """
    智能寻找 data/models 下最新的模型文件
    支持识别普通训练目录和 WF (滚动训练) 目录
    返回: (version, model_info_dict)
    model_info_dict 格式:
      - 单模型: {"type": "single", "path": "...", "format": "xgb/lgb"}
      - 双头: {"type": "dual_head", "reg_path": "...", "cls_path": "...", "format": "lgb"}
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
        clean_name = dir_name.replace("WF_", "")
        try:
            return datetime.datetime.strptime(clean_name, "%Y%m%d_%H%M%S")
        except ValueError:
            return datetime.datetime.min

    # 3. 按时间倒序排列 (最新的在前)
    subdirs.sort(key=parse_timestamp, reverse=True)
    latest_version = subdirs[0]
    version_dir = os.path.join(models_dir, latest_version)
    
    logger.info(f"锁定最新模型版本目录: {latest_version}")

    # 4. 检测模型类型
    model_info = {}
    
    # 情况 A: 双头模型 (LightGBM joblib 格式)
    # 检查是否存在 model_reg*.joblib 和 model_cls*.joblib
    reg_models = glob.glob(os.path.join(version_dir, "model_reg*.joblib"))
    cls_models = glob.glob(os.path.join(version_dir, "model_cls*.joblib"))
    
    if reg_models and cls_models:
        # 双头模型，找年份最大的
        def extract_year(path):
            fname = os.path.basename(path)
            match = re.search(r"model_(?:reg|cls)_(\d+)\.joblib", fname)
            return int(match.group(1)) if match else 0
        
        best_reg = max(reg_models, key=extract_year)
        best_cls = max(cls_models, key=extract_year)
        best_year = extract_year(best_reg)
        
        logger.info(f"检测到双头模型，已自动选择最新年份: {best_year}")
        model_info = {
            "type": "dual_head",
            "reg_path": best_reg,
            "cls_path": best_cls,
            "format": "lgb"
        }
        return latest_version, model_info
    
    # 情况 B: 单模型 - LightGBM joblib (model_reg.joblib 单独存在)
    single_lgb = os.path.join(version_dir, "model_reg.joblib")
    if os.path.exists(single_lgb):
        model_info = {"type": "single", "path": single_lgb, "format": "lgb"}
        return latest_version, model_info
    
    # 情况 C: 单模型 - XGBoost json
    if os.path.exists(os.path.join(version_dir, "model.json")):
        model_info = {"type": "single", "path": os.path.join(version_dir, "model.json"), "format": "xgb"}
        return latest_version, model_info
    
    # 情况 D: 滚动训练的 XGBoost 年度模型 (model_2024.json ...)
    wf_xgb_models = glob.glob(os.path.join(version_dir, "model_*.json"))
    if wf_xgb_models:
        def extract_year_xgb(path):
            fname = os.path.basename(path)
            match = re.search(r"model_(\d+)\.json", fname)
            return int(match.group(1)) if match else 0
        
        best_model_path = max(wf_xgb_models, key=extract_year_xgb)
        best_year = extract_year_xgb(best_model_path)
        logger.info(f"检测到滚动训练 XGBoost 模型，已自动选择最新年份: model_{best_year}.json")
        model_info = {"type": "single", "path": best_model_path, "format": "xgb"}
        return latest_version, model_info

    return None, None

def load_model(model_info):
    """
    根据 model_info 加载模型
    返回: model 或 (reg_model, cls_model)
    """
    if model_info["type"] == "single":
        if model_info["format"] == "xgb":
            from src.model.xgb_model import XGBModelWrapper
            model = XGBModelWrapper()
            model.load(model_info["path"])
            return model, None
        else:  # lgb
            from src.model.lgb_model import LGBModelWrapper
            model = LGBModelWrapper(task_type="regression")
            model.load(model_info["path"])
            return model, None
    else:  # dual_head
        from src.model.lgb_model import LGBModelWrapper
        reg_model = LGBModelWrapper(task_type="regression")
        reg_model.load(model_info["reg_path"])
        cls_model = LGBModelWrapper(task_type="classification")
        cls_model.load(model_info["cls_path"])
        return reg_model, cls_model

def fuse_predictions(pred_reg, pred_cls, dual_head_cfg):
    """融合双头模型预测结果"""
    import numpy as np
    
    fusion_cfg = dual_head_cfg.get("fusion", {})
    normalize = fusion_cfg.get("normalize", True)
    reg_weight = dual_head_cfg.get("regression", {}).get("weight", 0.6)
    cls_weight = dual_head_cfg.get("classification", {}).get("weight", 0.4)
    
    if normalize:
        def min_max_normalize(arr):
            arr = np.array(arr)
            min_val, max_val = arr.min(), arr.max()
            if max_val - min_val < 1e-9:
                return np.zeros_like(arr)
            return (arr - min_val) / (max_val - min_val)
        pred_reg = min_max_normalize(pred_reg)
        pred_cls = min_max_normalize(pred_cls)
    
    return reg_weight * pred_reg + cls_weight * pred_cls

def load_latest_data():
    """加载特征数据，并提取出【最近 N 个交易日】的数据，用于预测和平滑。"""
    data_path = os.path.join(GLOBAL_CONFIG["paths"]["data_processed"], "all_stocks.parquet")
    if not os.path.exists(data_path):
        logger.error(f"特征文件不存在: {data_path}，请先运行 rebuild_features.py")
        return None, None, None

    df = read_parquet(data_path)
    df["date"] = pd.to_datetime(df["date"])
    
    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    
    N_DAYS = 3 
    unique_dates = sorted(df["date"].unique(), reverse=True)
    
    if len(unique_dates) < N_DAYS:
        logger.warning(f"总交易日 ({len(unique_dates)}) 少于平滑窗口 ({N_DAYS}天)，使用全部数据。")
        target_dates = unique_dates
    else:
        target_dates = unique_dates[:N_DAYS]
    
    df_slice = df[df["date"].isin(target_dates)].copy()
    
    if df_slice.empty:
        logger.error("数据切片为空，无法推荐。")
        return None, None, None
    
    latest_date = unique_dates[0]
    logger.info(f"数据集中最新日期为: {latest_date.strftime('%Y-%m-%d')}，将加载前 {len(target_dates)} 个交易日的数据。")

    return df_slice, feat_cols, latest_date

def main():
    logger.info("=== 启动每日推荐系统 (Daily Recommendation) ===")

    # 读取双头模型配置
    dual_head_cfg = GLOBAL_CONFIG["model"].get("dual_head", {})
    dual_head_enabled = dual_head_cfg.get("enable", False)
    logger.info(f"双头模型配置: {'启用' if dual_head_enabled else '禁用'}")

    # 1. 智能加载模型
    version, model_info = get_latest_model_path()
    if not model_info:
        logger.error("未找到可用模型文件，请先运行 run_walkforward.py 或 train_model.py")
        return
    
    logger.info(f"模型类型: {model_info['type']}, 格式: {model_info['format']}")
    
    reg_model, cls_model = load_model(model_info)
    is_dual_head = model_info["type"] == "dual_head"
    
    # 2. 加载最新行情数据（最近 N 天）
    df_slice, feat_cols, latest_date = load_latest_data()
    if df_slice is None or df_slice.empty:
        logger.error("无数据切片，无法推荐。")
        return

    # 3. 执行预测
    logger.info(f"正在对 {len(df_slice)} 行数据 ({df_slice['symbol'].nunique()} 只股票) 进行打分...")
    
    # 3.1 特征对齐
    final_features = feat_cols
    # 使用模型记录的特征名
    if reg_model and hasattr(reg_model, 'feature_names') and reg_model.feature_names:
        model_features = reg_model.feature_names
        logger.info(f"使用模型内置特征列表: {len(model_features)} 个")
        
        missing = [f for f in model_features if f not in df_slice.columns]
        if missing:
            logger.error(f"严重错误：数据中缺少模型所需的特征: {missing}")
            return
            
        final_features = model_features
    else:
        logger.warning(f"模型未记录特征名，将使用所有 {len(final_features)} 个 'feat_' 开头的列。")

    # 3.2 预测分数
    try:
        X_pred = df_slice[final_features]
        
        if is_dual_head:
            pred_reg = reg_model.predict(X_pred)
            pred_cls = cls_model.predict(X_pred)
            pred_scores = fuse_predictions(pred_reg, pred_cls, dual_head_cfg)
            logger.info(f"双头融合预测完成 (权重: reg={dual_head_cfg.get('regression', {}).get('weight', 0.6)}, cls={dual_head_cfg.get('classification', {}).get('weight', 0.4)})")
        else:
            pred_scores = reg_model.predict(X_pred)
            
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
    
    strategy = TopKSignalStrategy(top_k=rec_k)
    recommend_df = strategy.generate(pred_df)
    recommend_df_latest = recommend_df[recommend_df["date"] == latest_date].copy()
    
    # 5. 输出结果
    if recommend_df_latest.empty:
        logger.warning("策略筛选后无股票入选 (可能都被风控剔除或分数不足)。")
        logger.info("Top 5 原始预测得分 (未经过滤):")
        print(pred_df[pred_df["date"] == latest_date].sort_values("pred_score", ascending=False).head(5))
        return

    current_pos_ratio = 1.0
    if "pos_ratio" in recommend_df_latest.columns:
        current_pos_ratio = recommend_df_latest["pos_ratio"].iloc[0]

    meta_path = os.path.join(GLOBAL_CONFIG["paths"]["data_meta"], "all_stocks_meta.parquet")
    if os.path.exists(meta_path):
        df_meta = read_parquet(meta_path)
        recommend_df_latest = pd.merge(recommend_df_latest, df_meta[["symbol", "name"]], on="symbol", how="left")
    
    recommend_df_latest = pd.merge(recommend_df_latest, 
                                   pred_df[["date", "symbol", "pred_score"]], 
                                   on=["date", "symbol"], how="left")
    
    # 格式化输出
    print("\n" + "="*70)
    print(f"🌟 {latest_date.strftime('%Y-%m-%d')} 每日精选推荐 (Top {len(recommend_df_latest)}) 🌟")
    
    if is_dual_head:
        print(f"📊 使用双头模型 (回归+分类融合)")
    
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
    
    cols = ["symbol", "name", "pred_score", "pos_ratio", "weight"]
    print_cols = [c for c in cols if c in recommend_df_latest.columns]
    
    print_df = recommend_df_latest[print_cols].sort_values("pred_score", ascending=False).reset_index(drop=True)
    
    try:
        print(print_df.to_markdown(index=True, floatfmt=".4f"))
    except:
        print(print_df)
    
    # 保存结果
    out_dir = os.path.join(GLOBAL_CONFIG["paths"]["reports"], "daily_picks")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_file = os.path.join(out_dir, f"picks_{version}_{latest_date.strftime('%Y%m%d')}.csv")
    print_df.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"\n[文件] 推荐列表已保存至: {out_file}")
    print("="*70)

if __name__ == "__main__":
    main()
