# scripts/model_train/run_inference_update.py
import os
import sys
import yaml
import glob
import pandas as pd
import numpy as np

# 路径适配
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.model.lgb_model import LGBModelWrapper
from src.utils.logger import get_logger
from src.utils.io import read_parquet, ensure_dir

logger = get_logger()

def load_run_config(model_dir):
    """加载模型运行配置"""
    config_path = os.path.join(model_dir, "run_config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}

def find_latest_models(model_dir):
    """查找最新的回归和分类模型（年份最大）"""
    reg_models = glob.glob(os.path.join(model_dir, "model_reg_*.joblib"))
    cls_models = glob.glob(os.path.join(model_dir, "model_cls_*.joblib"))
    
    if not reg_models and not cls_models:
        raise FileNotFoundError(f"在 {model_dir} 未找到任何模型文件")
        
    def get_year(path):
        try:
            return int(os.path.basename(path).split("_")[-1].split(".")[0])
        except:
            return 0

    latest_reg = max(reg_models, key=get_year) if reg_models else None
    latest_cls = max(cls_models, key=get_year) if cls_models else None
    
    logger.info(f"找到最新模型:\n  Reg: {latest_reg}\n  Cls: {latest_cls}")
    return latest_reg, latest_cls

def fuse_predictions(pred_reg, pred_cls, config):
    """融合预测分数 - 逻辑复用自 ModelTrainer"""
    dual_head_cfg = config.get("dual_head", {})
    fusion_cfg = dual_head_cfg.get("fusion", {})
    method = fusion_cfg.get("method", "weighted_average")
    normalize = fusion_cfg.get("normalize", True)
    
    reg_weight = dual_head_cfg.get("regression", {}).get("weight", 0.6)
    cls_weight = dual_head_cfg.get("classification", {}).get("weight", 0.4)
    
    logger.info(f"融合配置: Reg权重={reg_weight}, Cls权重={cls_weight}, 归一化={normalize}")
    
    def min_max_normalize(arr):
        arr = np.array(arr)
        min_val, max_val = np.nanmin(arr), np.nanmax(arr)
        if max_val - min_val > 1e-8:
            return (arr - min_val) / (max_val - min_val)
        return np.zeros_like(arr)
    
    if normalize:
        if pred_reg is not None:
            pred_reg = min_max_normalize(pred_reg)
        if pred_cls is not None:
            pred_cls = min_max_normalize(pred_cls)
            
    if method == "weighted_average":
        if pred_reg is not None and pred_cls is not None:
            fused = reg_weight * pred_reg + cls_weight * pred_cls
        elif pred_reg is not None:
            fused = pred_reg
        else:
            fused = pred_cls
    else:
        # 简化处理，默认加权平均
        if pred_reg is not None and pred_cls is not None:
            fused = reg_weight * pred_reg + cls_weight * pred_cls
        elif pred_reg is not None:
            fused = pred_reg
        else:
            fused = pred_cls
            
    return pred_reg, pred_cls, fused

def main():
    # 1. 确定模型目录 (硬编码为最新已知的目录)
    model_dir = os.path.join(project_root, "data", "models", "WF_20260118_172453")
    if not os.path.exists(model_dir):
        logger.error(f"模型目录不存在: {model_dir}")
        return

    logger.info(f"=== 开始增量推理更新 (Model: {os.path.basename(model_dir)}) ===")

    # 2. 加载现有预测数据，确定最后日期
    pred_path = os.path.join(model_dir, "predictions.parquet")
    if not os.path.exists(pred_path):
        logger.error(f"预测文件不存在: {pred_path}")
        return
        
    old_preds = read_parquet(pred_path)
    old_preds["date"] = pd.to_datetime(old_preds["date"])
    last_date = old_preds["date"].max()
    logger.info(f"现有预测数据截至日期: {last_date.date()}")
    
    # 3. 加载最新的特征数据
    data_path = os.path.join(project_root, "data", "processed", "all_stocks.parquet")
    df = read_parquet(data_path)
    df["date"] = pd.to_datetime(df["date"])
    
    # 筛选新数据 (大于 last_date)
    new_data = df[df["date"] > last_date].copy()
    if new_data.empty:
        logger.info("没有更新的数据需要预测。")
        return
        
    logger.info(f"发现新数据: {len(new_data)} 条，日期范围: {new_data['date'].min().date()} ~ {new_data['date'].max().date()}")
    
    # 4. 准备特征
    feature_cols = [c for c in df.columns if c.startswith("feat_")]
    X_new = new_data[feature_cols]
    
    # 5. 加载模型并预测
    reg_path, cls_path = find_latest_models(model_dir)
    run_config = load_run_config(model_dir)
    
    pred_reg = None
    pred_cls = None
    
    if reg_path:
        logger.info("正在进行回归预测...")
        reg_model = LGBModelWrapper(task_type="regression")
        reg_model.load(reg_path)
        
        # 特征对齐
        if reg_model.feature_names:
            logger.info(f"对齐回归模型特征: {len(reg_model.feature_names)} 个")
            # 检查缺失特征
            missing_cols = [c for c in reg_model.feature_names if c not in X_new.columns]
            if missing_cols:
                logger.warning(f"缺失特征 (自动补0): {missing_cols}")
                for c in missing_cols:
                    X_new[c] = 0
            X_reg = X_new[reg_model.feature_names]
        else:
            X_reg = X_new
            
        pred_reg = reg_model.predict(X_reg)
        
    if cls_path:
        logger.info("正在进行分类预测...")
        cls_model = LGBModelWrapper(task_type="classification")
        cls_model.load(cls_path)
        
        # 特征对齐
        if cls_model.feature_names:
            logger.info(f"对齐分类模型特征: {len(cls_model.feature_names)} 个")
            missing_cols = [c for c in cls_model.feature_names if c not in X_new.columns]
            if missing_cols:
                 # 不需要再警告一遍了，假设两个模型特征一致
                 for c in missing_cols:
                    X_new[c] = 0
            X_cls = X_new[cls_model.feature_names]
        else:
            X_cls = X_new

        pred_cls = cls_model.predict(X_cls)
        
    # 6. 融合预测
    pred_reg_norm, pred_cls_norm, pred_score = fuse_predictions(pred_reg, pred_cls, run_config)
    
    # 7. 构造结果并追加
    # 注意：我们要保留原始的 pred_reg / pred_cls 值用于后续分析吗？
    # 旧数据中 pred_reg 和 pred_cls 可能已经是归一化过的吗？
    # 检查 fuse_predictions 逻辑：它直接修改了 pred_reg 变量指向归一化后的数组。
    # 为了保持一致性，我们应该查看 ModelTrainer 如何保存。
    # ModelTrainer 中：df["pred_reg"] = pred_reg (原始预测值)
    # 然后 fuse_predictions 计算 score 时才做归一化。
    # 所以保存到 parquet 的应该是原始值。
    
    new_data["pred_reg"] = pred_reg if pred_reg is not None else np.nan
    new_data["pred_cls"] = pred_cls if pred_cls is not None else np.nan
    new_data["pred_score"] = pred_score
    
    # 保持列一致
    cols_to_save = list(old_preds.columns)
    # 确保新数据有所有列，缺失补 NaN
    for col in cols_to_save:
        if col not in new_data.columns:
            new_data[col] = np.nan
            
    final_new_preds = new_data[cols_to_save]
    
    # 合并
    updated_preds = pd.concat([old_preds, final_new_preds], axis=0, ignore_index=True)
    updated_preds = updated_preds.sort_values(["date", "symbol"])
    
    # 8. 保存
    # 备份旧文件
    backup_path = pred_path + ".bak"
    import shutil
    shutil.copy2(pred_path, backup_path)
    logger.info(f"已备份原预测文件至: {backup_path}")
    
    updated_preds.to_parquet(pred_path, index=False)
    logger.info(f"已保存更新后的预测文件，共 {len(updated_preds)} 条记录。")
    logger.info(f"最新日期更新至: {updated_preds['date'].max().date()}")

if __name__ == "__main__":
    main()
