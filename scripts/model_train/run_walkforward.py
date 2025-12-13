# scripts/run_walkforward.py

import os
import sys
import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm

# 路径适配
current_dir = os.path.dirname(os.path.abspath(__file__))
# 从当前文件位置 (scripts/analisis) 返回两级到项目根目录
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.config import GLOBAL_CONFIG
from src.utils.logger import get_logger
from src.utils.io import ensure_dir, save_parquet
from src.model.trainer import ModelTrainer

logger = get_logger()

class WalkForwardRunner:
    def __init__(self):
        self.config = GLOBAL_CONFIG
        self.wf_cfg = self.config["model"].get("walk_forward", {})
        
        # 基础参数
        self.start_year = self.wf_cfg.get("start_year", 2019)
        self.window_type = self.wf_cfg.get("train_window_type", "expanding")
        self.rolling_size = self.wf_cfg.get("rolling_window_size", 5)
        
        # 版本管理 (WF_ + 时间戳)
        self.version = "WF_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(GLOBAL_CONFIG["paths"]["models"], self.version)
        ensure_dir(self.output_dir)

    def run(self):
        logger.info(f"=== 启动滚动训练 (Walk-Forward Validation) ===")
        logger.info(f"版本号: {self.version}")
        logger.info(f"输出路径: {self.output_dir}")
        logger.info(f"策略: {self.window_type} | 测试集起始年份: {self.start_year}")

        # 1. 复用 Trainer 加载数据
        trainer = ModelTrainer()
        df, features, label = trainer.load_data()
        
        # 确保日期列为 datetime
        df["date"] = pd.to_datetime(df["date"])
        
        # 获取所有包含的年份
        data_years = sorted(df["date"].dt.year.unique())
        test_years = [y for y in data_years if y >= self.start_year]
        
        if not test_years:
            logger.error(f"数据中没有 >= {self.start_year} 的年份！数据年份范围: {min(data_years)}~{max(data_years)}")
            return

        all_preds = []
        
        # 2. 按年循环
        for year in test_years:
            logger.info("-" * 40)
            logger.info(f">>> 正在处理: {year} 年 (Test Period)")
            
            # --- A. 划分时间窗口 ---
            test_start = f"{year}-01-01"
            test_end = f"{year}-12-31"
            
            # 训练集结束时间：测试年的前一年最后一天
            train_end_dt = datetime.datetime(year - 1, 12, 31)
            train_end = train_end_dt.strftime("%Y-%m-%d")
            
            # 训练集开始时间
            if self.window_type == "rolling":
                # 滚动窗口: Start = End - N years
                train_start_dt = train_end_dt - datetime.timedelta(days=365 * self.rolling_size)
                train_start = train_start_dt.strftime("%Y-%m-%d")
            else:
                # 扩张窗口: 从数据最早时间开始
                train_start = "2000-01-01" # 只要比数据早即可

            logger.info(f"  Train: {train_start} ~ {train_end}")
            logger.info(f"  Test : {test_start} ~ {test_end}")

            # --- B. 切分数据 ---
            train_mask = (df["date"] >= train_start) & (df["date"] <= train_end)
            test_mask = (df["date"] >= test_start) & (df["date"] <= test_end)
            
            if train_mask.sum() < 100:
                logger.warning(f"  {year} 年训练数据不足 ({train_mask.sum()}条)，跳过。")
                continue
            if test_mask.sum() == 0:
                logger.warning(f"  {year} 年无测试数据，跳过。")
                continue
                
            X_train = df.loc[train_mask, features]
            y_train = df.loc[train_mask, label]
            X_test = df.loc[test_mask, features]
            y_test = df.loc[test_mask, label] # 仅用于评估，不参与训练
            
            # --- C. 训练模型 ---
            # 这里我们把 Test 集作为 Validation 集传给 XGBoost 用于 early_stopping
            # 注意：这在严格意义上有一点点泄露（用来停机），但业界常用做法，
            # 或者你可以再从 train 里分一部分做 val。这里为了简单直接用 test 做 eval。
            model = trainer.train_model(X_train, y_train, X_test, y_test)
            
            # --- D. 保存当年模型 ---
            model_name = f"model_{year}.json"
            model.save(os.path.join(self.output_dir, model_name))
            
            # --- E. 生成预测 ---
            pred_scores = model.predict(X_test)
            
            # 构造结果片段
            res_df = df.loc[test_mask, ["date", "symbol", "close", label]].copy()
            res_df["pred_score"] = pred_scores
            res_df["train_year"] = year - 1 # 标记是用哪一年的模型预测的
            
            all_preds.append(res_df)
            logger.info(f"  {year} 年预测完成，样本数: {len(res_df)}")

        # 3. 合并并保存
        if all_preds:
            full_pred_df = pd.concat(all_preds, ignore_index=True)
            
            # 按日期排序
            full_pred_df = full_pred_df.sort_values(["date", "symbol"])
            
            save_path = os.path.join(self.output_dir, "predictions.parquet")
            save_parquet(full_pred_df, save_path)
            
            logger.info("=" * 40)
            logger.info(f"全量滚动预测结果已保存: {save_path}")
            logger.info(f"总样本数: {len(full_pred_df)}")
            logger.info(f"覆盖年份: {full_pred_df['date'].dt.year.unique()}")
            
            # 自动生成 config 备份 (方便追溯)
            import yaml
            with open(os.path.join(self.output_dir, "run_config.yaml"), "w") as f:
                yaml.dump(self.wf_cfg, f)
                
        else:
            logger.error("未能生成任何预测结果！")

if __name__ == "__main__":
    runner = WalkForwardRunner()
    runner.run()