# scripts/run_recommendation.py

import os
import sys
import argparse
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

# 设置默认模型版本名（如 "WF_20260122_145035"），为 None 则自动搜索最新模型
DEFAULT_MODEL_VERSION = "WF_20260128_141530"


def get_latest_model_path(models_dir=None):
    """
    智能寻找 data/models 下最新的模型文件
    支持识别普通训练目录和 WF (滚动训练) 目录
    返回: (version, model_info_dict)
    model_info_dict 格式:
      - 单模型: {"type": "single", "path": "...", "format": "xgb/lgb"}
      - 双头: {"type": "dual_head", "reg_path": "...", "cls_path": "...", "format": "lgb"}
    """
    models_dir = models_dir or GLOBAL_CONFIG["paths"]["models"]
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
    
    # 情况 A: 双头模型
    # 支持两种格式: .joblib (LightGBM) 和 .ubj (XGBoost)
    return_models = (glob.glob(os.path.join(version_dir, "model_return*.joblib")) + 
                    glob.glob(os.path.join(version_dir, "model_return*.ubj")))
    risk_models = (glob.glob(os.path.join(version_dir, "model_risk*.joblib")) +
                  glob.glob(os.path.join(version_dir, "model_risk*.ubj")))
    
    # 向后兼容：如果没有找到新版，寻找旧版
    if not return_models:
        return_models = (glob.glob(os.path.join(version_dir, "model_reg*.joblib")) +
                        glob.glob(os.path.join(version_dir, "model_reg*.ubj")))
    if not risk_models:
        risk_models = (glob.glob(os.path.join(version_dir, "model_cls*.joblib")) +
                      glob.glob(os.path.join(version_dir, "model_cls*.ubj")))
    
    if return_models and risk_models:
        # 双头模型，找年份最大的
        def extract_year(path):
            fname = os.path.basename(path)
            # 匹配 model_(return|risk|reg|cls)_YYYY.(joblib|ubj) 或 model_(return|risk|reg|cls).(joblib|ubj)
            match = re.search(r"model_(?:return|risk|reg|cls)(?:_(\d+))?\.(joblib|ubj)", fname)
            return int(match.group(1)) if (match and match.group(1)) else 0
        
        best_return = max(return_models, key=extract_year)
        best_risk = max(risk_models, key=extract_year)
        best_year = extract_year(best_return)
        
        logger.info(f"检测到双头模型 (收益+风险预测)，已自动选择最新年份: {best_year if best_year > 0 else '无年份后缀'}")
        model_info = {
            "type": "dual_head",
            "return_path": best_return,
            "risk_path": best_risk,
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

def detect_model_in_dir(version_dir):
    """
    鎸夌洰褰曡矾寰勬嫻鍨嬫ā鍨? 
    杩斿洖: model_info_dict
    """
    if not os.path.exists(version_dir):
        return None

    model_info = {}
    
    # 鎯呭喌 A: 鍙屽ご妯″瀷
    # 鏀寔涓ょ鏍煎紡: .joblib (LightGBM) 鍜?.ubj (XGBoost)
    return_models = (glob.glob(os.path.join(version_dir, "model_return*.joblib")) + 
                    glob.glob(os.path.join(version_dir, "model_return*.ubj")))
    risk_models = (glob.glob(os.path.join(version_dir, "model_risk*.joblib")) +
                  glob.glob(os.path.join(version_dir, "model_risk*.ubj")))
    
    # 鍚戝悗鍏煎锛氬鏋滄病鏈夋壘鍒版柊鐗堬紝瀵绘壘鏃х増
    if not return_models:
        return_models = (glob.glob(os.path.join(version_dir, "model_reg*.joblib")) +
                        glob.glob(os.path.join(version_dir, "model_reg*.ubj")))
    if not risk_models:
        risk_models = (glob.glob(os.path.join(version_dir, "model_cls*.joblib")) +
                      glob.glob(os.path.join(version_dir, "model_cls*.ubj")))
    
    if return_models and risk_models:
        # 鍙屽ご妯″瀷锛屾壘骞翠唤鏈€澶х殑
        def extract_year(path):
            fname = os.path.basename(path)
            # 鍖归厤 model_(return|risk|reg|cls)_YYYY.(joblib|ubj) 鎴?model_(return|risk|reg|cls).(joblib|ubj)
            match = re.search(r"model_(?:return|risk|reg|cls)(?:_(\d+))?\.(joblib|ubj)", fname)
            return int(match.group(1)) if (match and match.group(1)) else 0
        
        best_return = max(return_models, key=extract_year)
        best_risk = max(risk_models, key=extract_year)
        best_year = extract_year(best_return)
        
        logger.info(f"Detected dual-head model, latest year: {best_year if best_year > 0 else 'unknown'}")
        model_info = {
            "type": "dual_head",
            "return_path": best_return,
            "risk_path": best_risk,
            "format": "lgb"
        }
        return model_info
    
    # 鎯呭喌 B: 鍗曟ā鍨?- LightGBM joblib (model_reg.joblib 鍗曠嫭瀛樺湪)
    single_lgb = os.path.join(version_dir, "model_reg.joblib")
    if os.path.exists(single_lgb):
        model_info = {"type": "single", "path": single_lgb, "format": "lgb"}
        return model_info
    
    # 鎯呭喌 C: 鍗曟ā鍨?- XGBoost json
    if os.path.exists(os.path.join(version_dir, "model.json")):
        model_info = {"type": "single", "path": os.path.join(version_dir, "model.json"), "format": "xgb"}
        return model_info
    
    # 鎯呭喌 D: 婊氬姩璁粌鐨?XGBoost 骞村害妯″瀷 (model_2024.json ...)
    wf_xgb_models = glob.glob(os.path.join(version_dir, "model_*.json"))
    if wf_xgb_models:
        def extract_year_xgb(path):
            fname = os.path.basename(path)
            match = re.search(r"model_(\d+)\.json", fname)
            return int(match.group(1)) if match else 0
        
        best_model_path = max(wf_xgb_models, key=extract_year_xgb)
        best_year = extract_year_xgb(best_model_path)
        logger.info(f"Detected walk-forward XGBoost model, latest year: model_{best_year}.json")
        model_info = {"type": "single", "path": best_model_path, "format": "xgb"}
        return model_info

    return None

def load_model(model_info):
    """
    根据 model_info 加载模型
    返回: model 或 (reg_model, cls_model)
    
    修复：直接检查文件扩展名来判断模型格式（.ubj=XGBoost, .joblib=LightGBM）
    """
    # 兼容 'type' 和 'model_type' 两种键名
    model_type = model_info.get('model_type') or model_info.get('type')
    logger.info(f"模型类型: {model_type}, 配置格式: {model_info['format']}")
    
    if model_type == "single":
        model_path = model_info["path"]
        # 检测实际文件格式（优先使用文件扩展名）
        if model_path.endswith('.ubj') or model_path.endswith('.json'):
            logger.info("检测到XGBoost模型格式")
            from src.model.xgb_model import XGBModelWrapper
            model = XGBModelWrapper()
            model.load(model_path)
            return model, None
        else:  # .joblib
            logger.info("检测到LightGBM模型格式")
            from src.model.lgb_model import LGBModelWrapper
            model = LGBModelWrapper(task_type="regression")
            model.load(model_path)
            return model, None
    else:  # dual_head
        # 兼容新旧路径
        return_path = model_info.get("return_path") or model_info.get("reg_path")
        risk_path = model_info.get("risk_path") or model_info.get("cls_path")
        
        # 检测实际文件格式
        if return_path.endswith('.ubj'):
            logger.info("检测到XGBoost双头模型（.ubj格式）")
            from src.model.xgb_model import XGBModelWrapper
            return_model = XGBModelWrapper()
            risk_model = XGBModelWrapper()
        else:  # .joblib
            logger.info("检测到LightGBM双头模型（.joblib格式）")
            from src.model.lgb_model import LGBModelWrapper
            return_model = LGBModelWrapper(task_type="regression")
            risk_model = LGBModelWrapper(task_type="regression")
        
        return_model.load(return_path)
        risk_model.load(risk_path)
        return return_model, risk_model

def fuse_predictions(pred_df, dual_head_cfg):
    """
    融合双头模型预测结果 (收益+风险) - 与回测逻辑保持一致
    """
    import numpy as np
    from scipy.stats import rankdata
    
    fusion_cfg = dual_head_cfg.get("fusion", {})
    method = fusion_cfg.get("method", "rank_ratio")
    risk_aversion = fusion_cfg.get("risk_aversion", 2.0)
    
    return_weight = dual_head_cfg.get("return_head", {}).get("weight", 0.6)
    risk_weight = dual_head_cfg.get("risk_head", {}).get("weight", 0.4)
    
    has_return = "pred_return" in pred_df.columns
    has_risk = "pred_risk" in pred_df.columns
    has_reg = "pred_reg" in pred_df.columns
    has_cls = "pred_cls" in pred_df.columns

    if has_return and has_risk:
        pred_return = pred_df["pred_return"].values
        pred_risk = pred_df["pred_risk"].values
    elif has_reg and has_cls:
        pred_return = pred_df["pred_reg"].values
        pred_risk = pred_df["pred_cls"].values
    else:
        raise ValueError("缺少 pred_return/pred_risk 或 pred_reg/pred_cls，无法融合")
    
    if method == "rank_ratio":
        n = len(pred_return)
        rank_return = rankdata(pred_return, nan_policy='omit') / n
        rank_risk = rankdata(pred_risk, nan_policy='omit') / n
        epsilon = 0.01
        fused = rank_return / (rank_risk + epsilon)
        fused = (fused - np.nanmin(fused)) / (np.nanmax(fused) - np.nanmin(fused) + 1e-8)
        return fused
    
    if method == "sharpe_like":
        epsilon = np.nanstd(pred_risk) * 0.1 if np.nanstd(pred_risk) > 0 else 0.01
        return pred_return / (pred_risk + epsilon)
    
    if method == "utility":
        return pred_return - risk_aversion * (pred_risk ** 2)
    
    if method == "weighted_average":
        ret_norm = (pred_return - np.nanmin(pred_return)) / (np.nanmax(pred_return) - np.nanmin(pred_return) + 1e-8)
        risk_norm = (pred_risk - np.nanmin(pred_risk)) / (np.nanmax(pred_risk) - np.nanmin(pred_risk) + 1e-8)
        return return_weight * ret_norm - risk_weight * risk_norm
    
    logger.warning(f"未知融合方法: {method}，使用 rank_ratio")
    rank_return = rankdata(pred_return, nan_policy='omit')
    rank_risk = rankdata(pred_risk, nan_policy='omit')
    return rank_return / (rank_risk + 1.0)

def load_latest_data():
    """加载特征数据，并提取出【最近 N 个交易日】的数据，用于预测和平滑。"""
    data_path = os.path.join(GLOBAL_CONFIG["paths"]["data_processed"], "all_stocks.parquet")
    if not os.path.exists(data_path):
        logger.error(f"特征文件不存在: {data_path}，请先运行 rebuild_features.py")
        return None, None, None

    df = read_parquet(data_path)
    df["date"] = pd.to_datetime(df["date"])
    
    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    
    strat_cfg = GLOBAL_CONFIG.get("strategy", {})
    N_DAYS = int(strat_cfg.get("recommend_history_days", 3))
    if N_DAYS <= 0:
        N_DAYS = 1
    if strat_cfg.get("enable_score_smoothing", True):
        N_DAYS = max(N_DAYS, 2)
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

def resolve_feature_columns(df_slice, return_model, risk_model):
    model_features = None
    if return_model is not None and getattr(return_model, "feature_names", None):
        model_features = list(return_model.feature_names)
    if risk_model is not None and getattr(risk_model, "feature_names", None):
        risk_features = list(risk_model.feature_names)
        if model_features is None:
            model_features = risk_features
        elif model_features != risk_features:
            logger.error("收益模型与风险模型特征不一致，请检查训练或模型文件。")
            return None, None

    if model_features:
        return model_features, "model"

    use_feature_selection = GLOBAL_CONFIG.get("model", {}).get("use_feature_selection", False)
    if use_feature_selection:
        selected_path = os.path.join(GLOBAL_CONFIG["paths"]["data_processed"], "selected_features.txt")
        if not os.path.exists(selected_path):
            logger.error(f"已启用特征筛选，但未找到文件: {selected_path}")
            return None, None
        with open(selected_path, "r", encoding="utf-8") as f:
            features = [
                line.strip() for line in f.readlines()
                if line.strip() and not line.strip().startswith("#")
            ]
        if not features:
            logger.error("selected_features.txt 为空，无法对齐特征。")
            return None, None
        return features, "selected_features"

    return [c for c in df_slice.columns if c.startswith("feat_")], "all_feat"

def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run daily recommendation with optional custom models.")
    parser.add_argument("--model-dir", help="Custom model directory path (e.g. data/models/WF_20260122_145035)")
    return parser

def main():
    args = build_arg_parser().parse_args()
    logger.info("=== 启动每日推荐系统 (Daily Recommendation) ===")

    # 读取双头模型配置
    dual_head_cfg = GLOBAL_CONFIG["model"].get("dual_head", {})
    dual_head_enabled = dual_head_cfg.get("enable", False)
    logger.info(f"双头模型配置: {'启用' if dual_head_enabled else '禁用'}")

    # 1. 智能加载模型（可指定模型目录）
    model_version = args.model_dir or DEFAULT_MODEL_VERSION
    model_dir = None
    if model_version:
        if os.path.sep in model_version or (os.path.altsep and os.path.altsep in model_version):
            model_dir = model_version
        else:
            model_dir = os.path.join(GLOBAL_CONFIG["paths"]["models"], model_version)
        version = os.path.basename(os.path.normpath(model_dir)) or "manual_dir"
        model_info = detect_model_in_dir(model_dir)
        if not model_info:
            logger.warning(f"指定模型目录未找到可用模型: {model_dir}，将回退到最新模型。")
            version, model_info = get_latest_model_path()
            model_dir = os.path.join(GLOBAL_CONFIG["paths"]["models"], version)
    else:
        version, model_info = get_latest_model_path()
        model_dir = os.path.join(GLOBAL_CONFIG["paths"]["models"], version)
    if not model_info:
        logger.error("未找到可用模型文件，请先运行 run_walkforward.py 或 train_model.py")
        return
    
    logger.info(f"使用模型版本: {version}, 模型目录: {model_dir}")
    logger.info(f"模型类型: {model_info['type']}, 格式: {model_info['format']}")

    return_model, risk_model = load_model(model_info)
    is_dual_head = model_info["type"] == "dual_head"

    # 2. 加载最新行情数据（最近 N 天）
    df_slice, feat_cols, latest_date = load_latest_data()
    if df_slice is None or df_slice.empty:
        logger.error("无数据切片，无法推荐。")
        return
    date_min = df_slice["date"].min()
    date_max = df_slice["date"].max()
    logger.info(f"预测数据日期范围: {date_min.strftime('%Y-%m-%d')} ~ {date_max.strftime('%Y-%m-%d')}")

    # 3. 执行预测
    logger.info(f"正在对 {len(df_slice)} 行数据 ({df_slice['symbol'].nunique()} 只股票) 进行打分...")
    
    # 3.1 特征对齐
    final_features, feature_source = resolve_feature_columns(df_slice, return_model, risk_model)
    if not final_features:
        return
    logger.info(f"使用特征列表来源: {feature_source}, 列数: {len(final_features)}")

    missing = [f for f in final_features if f not in df_slice.columns]
    if missing:
        logger.error(f"严重错误：数据中缺少模型所需的特征: {missing}")
        return

    # 3.2 预测分数
    try:
        X_pred = df_slice[final_features]
        
        if is_dual_head:
            pred_return = return_model.predict(X_pred)
            pred_risk = risk_model.predict(X_pred)
            
            # 构造临时 DataFrame 用于融合
            temp_df = df_slice[["date", "symbol"]].copy()
            temp_df["pred_return"] = pred_return
            temp_df["pred_risk"] = pred_risk
            
            pred_scores = fuse_predictions(temp_df, dual_head_cfg)
            logger.info(f"双头融合预测完成 (方法: {dual_head_cfg.get('fusion', {}).get('method', 'rank_ratio')})")
        else:
            pred_scores = return_model.predict(X_pred)

            
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
    rec_k = strat_cfg.get("top_k", 5)
    
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
        print(f"📊 使用双头模型 (收益+风险预测融合)")

    
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
