import sys
import os
import argparse
import pandas as pd
import numpy as np

# Path handling to ensure src can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
# current_dir is scripts/analisis, so we need to go up two levels
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.model.xgb_model import XGBModelWrapper
from src.model.trainer import ModelTrainer
from src.analysis.model_interpreter import ModelInterpreter
from src.utils.logger import get_logger
from src.utils.config import GLOBAL_CONFIG

logger = get_logger()

def main():
    parser = argparse.ArgumentParser(description="使用 SHAP 解释训练好的模型")
    parser.add_argument("--version", type=str, required=False, help="模型版本 (data/models 下的文件夹名)。如果不提供，默认使用最新版本。")
    parser.add_argument("--sample_size", type=int, default=2000, help="用于 SHAP 计算的样本数量")
    parser.add_argument("--top_k", type=int, default=3, help="绘制依赖图的前 K 个特征数量")
    parser.add_argument("--model_file", type=str, required=False, help="指定具体的模型文件名 (例如 model_2025.json)。如果不指定，将自动搜索。")
    
    args = parser.parse_args()
    
    models_root = GLOBAL_CONFIG["paths"]["models"]
    version = args.version
    
    if not version:
        # 自动检测最新版本
        if not os.path.exists(models_root):
             logger.error(f"未找到模型目录: {models_root}")
             return
        
        # 列出所有子目录
        subdirs = [d for d in os.listdir(models_root) if os.path.isdir(os.path.join(models_root, d))]
        if not subdirs:
            logger.error(f"在 {models_root} 中未找到模型版本")
            return
            
        # 按名称排序 (假设格式为 YYYYMMDD_HHMMSS)
        subdirs.sort(reverse=True)
        version = subdirs[0]
        logger.info(f"自动检测到最新模型版本: {version}")
    else:
        logger.info(f"使用指定模型版本: {version}")
    
    # 1. 加载配置和数据
    # 复用 ModelTrainer 处理数据加载和特征选择逻辑
    try:
        logger.info("正在初始化 ModelTrainer 以加载数据配置...")
        trainer = ModelTrainer()
        df, features, label = trainer.load_data()
    except Exception as e:
        logger.error(f"加载数据失败: {e}")
        return

    # 2. 加载模型
    model_dir = os.path.join(models_root, version)
    
    if not os.path.exists(model_dir):
        logger.error(f"未找到模型目录: {model_dir}")
        return
        
    model_wrapper = XGBModelWrapper()
    target_model_path = None
    
    if args.model_file:
        # 用户指定了文件名
        target_model_path = os.path.join(model_dir, args.model_file)
    else:
        # 自动搜索策略
        # 1. 优先找 model.json (标准单次训练)
        if os.path.exists(os.path.join(model_dir, "model.json")):
            target_model_path = os.path.join(model_dir, "model.json")
        # 2. 其次找 model.pkl
        elif os.path.exists(os.path.join(model_dir, "model.pkl")):
            target_model_path = os.path.join(model_dir, "model.pkl")
        else:
            # 3. 搜索 model_*.json (Walk-Forward 模式)
            import glob
            wf_models = glob.glob(os.path.join(model_dir, "model_*.json"))
            if wf_models:
                # 排序 (通常是 model_2019.json, ... model_2025.json)，取最后一个(最新年份)
                wf_models.sort()
                target_model_path = wf_models[-1]
                logger.info(f"检测到 Walk-Forward 目录，自动选择最新模型: {os.path.basename(target_model_path)}")
            
    if target_model_path and os.path.exists(target_model_path):
        logger.info(f"正在从 {target_model_path} 加载模型")
        model_wrapper.load(target_model_path)
    else:
        logger.error(f"在 {model_dir} 中未找到可用模型文件 (checked model.json, model.pkl, model_*.json)")
        return
        
    # 3. 准备解释用的数据
    # 优先使用模型记录的特征名 (如果有)
    model_features = None
    if hasattr(model_wrapper.model, "feature_names") and model_wrapper.model.feature_names:
        model_features = model_wrapper.model.feature_names
        logger.info(f"使用模型内置特征列表: {len(model_features)} 个")
    else:
        model_features = features
        logger.warning(f"模型未记录特征名，使用当前配置特征: {len(model_features)} 个")

    # 检查特征是否存在于数据中
    missing_features = [f for f in model_features if f not in df.columns]
    if missing_features:
        logger.error(f"数据中缺少以下模型特征: {missing_features}")
        return

    # 使用模型对应的特征
    X = df[model_features]
    
    if args.sample_size < len(X):
        logger.info(f"正在从 {len(X)} 条总数据中采样 {args.sample_size} 条")
        X_sample = X.sample(n=args.sample_size, random_state=42)
    else:
        X_sample = X
        
    # 4. 运行解释器
    logger.info("正在初始化模型解释器 (ModelInterpreter)...")
    # 传入底层 booster/model 对象
    interpreter = ModelInterpreter(model_wrapper.model, data=X_sample)
    
    logger.info("正在计算 SHAP 值...")
    interpreter.compute_shap_values(X_sample)
    
    # 5. 生成图表
    output_dir = os.path.join(GLOBAL_CONFIG["paths"].get("figures", "figures"), "interpretation", version)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"正在保存图表到 {output_dir}")
    
    # 摘要图
    interpreter.plot_summary(save_path=os.path.join(output_dir, "shap_summary_beeswarm.png"), plot_type='dot')
    interpreter.plot_summary(save_path=os.path.join(output_dir, "shap_summary_bar.png"), plot_type='bar')
    
    # Top 特征依赖图
    # 使用内置重要性查找 Top 特征
    importance = model_wrapper.get_feature_importance(importance_type='weight')
    if not importance:
         # 如果无法获取重要性，回退使用前几个特征
         top_features = features[:args.top_k]
    else:
        # 按重要性排序
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        top_features = [x[0] for x in sorted_importance[:args.top_k]]
        
    for feat in top_features:
        if feat in X_sample.columns:
            logger.info(f"正在生成 {feat} 的依赖图")
            interpreter.plot_dependence(feat, save_path=os.path.join(output_dir, f"shap_dependence_{feat}.png"))
            
    logger.info("模型解释分析已完成。")

if __name__ == "__main__":
    main()
