# scripts/test_dual_head.py
"""
双头模型功能测试脚本

测试内容:
1. 配置加载与双头开关识别
2. 分类标签生成 (label_cls)
3. LightGBM 模型包装器 (回归 + 分类)
4. 预测融合逻辑
5. 完整训练流程 (使用小数据集)
"""

import sys
import os

# 确保可以导入 src 模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
from datetime import datetime

# ========================================
# 测试结果统计
# ========================================
test_results = []

def log_test(name: str, passed: bool, message: str = ""):
    """记录测试结果"""
    status = "✅ PASS" if passed else "❌ FAIL"
    test_results.append((name, passed, message))
    print(f"{status}: {name}")
    if message and not passed:
        print(f"       └─ {message}")

def print_summary():
    """打印测试摘要"""
    passed = sum(1 for _, p, _ in test_results if p)
    total = len(test_results)
    print("\n" + "=" * 50)
    print(f"测试摘要: {passed}/{total} 通过")
    print("=" * 50)
    for name, p, msg in test_results:
        status = "✓" if p else "✗"
        print(f"  [{status}] {name}")

# ========================================
# 测试 1: 配置加载
# ========================================
def test_config_loading():
    """测试配置文件中双头模型配置的加载"""
    try:
        from src.utils.config import GLOBAL_CONFIG
        
        model_cfg = GLOBAL_CONFIG.get("model", {})
        dual_head_cfg = model_cfg.get("dual_head", {})
        
        # 检查必要的配置项
        assert "enable" in dual_head_cfg, "缺少 'enable' 配置"
        assert "model_type" in dual_head_cfg, "缺少 'model_type' 配置"
        assert "regression" in dual_head_cfg, "缺少 'regression' 配置"
        assert "classification" in dual_head_cfg, "缺少 'classification' 配置"
        assert "fusion" in dual_head_cfg, "缺少 'fusion' 配置"
        
        print(f"  - 双头开关: {'启用' if dual_head_cfg['enable'] else '禁用'}")
        print(f"  - 模型类型: {dual_head_cfg['model_type']}")
        print(f"  - 回归权重: {dual_head_cfg['regression'].get('weight', 0.6)}")
        print(f"  - 分类权重: {dual_head_cfg['classification'].get('weight', 0.4)}")
        
        log_test("配置加载", True)
        return dual_head_cfg
    except Exception as e:
        log_test("配置加载", False, str(e))
        return None

# ========================================
# 测试 2: LightGBM 模型包装器
# ========================================
def test_lgb_model_wrapper():
    """测试 LGBModelWrapper 的初始化和参数构建"""
    try:
        from src.model.lgb_model import LGBModelWrapper
        
        # 测试回归模型
        reg_model = LGBModelWrapper(task_type="regression")
        assert reg_model.task_type == "regression"
        assert reg_model.params.get("objective") == "regression"
        assert reg_model.params.get("metric") == "rmse"
        print("  - 回归模型初始化: ✓")
        
        # 测试分类模型
        cls_model = LGBModelWrapper(task_type="classification")
        assert cls_model.task_type == "classification"
        assert cls_model.params.get("objective") == "binary"
        assert cls_model.params.get("metric") == "auc"
        print("  - 分类模型初始化: ✓")
        
        log_test("LGBModelWrapper 初始化", True)
        return True
    except Exception as e:
        log_test("LGBModelWrapper 初始化", False, str(e))
        return False

# ========================================
# 测试 3: 模型训练与预测 (小数据集)
# ========================================
def test_model_training():
    """使用合成数据测试模型训练"""
    try:
        from src.model.lgb_model import LGBModelWrapper
        
        # 生成合成数据
        np.random.seed(42)
        n_samples = 1000
        n_features = 10
        
        X = np.random.randn(n_samples, n_features)
        y_reg = X[:, 0] * 0.5 + X[:, 1] * 0.3 + np.random.randn(n_samples) * 0.1
        y_cls = (y_reg > np.median(y_reg)).astype(int)
        
        # 划分训练/验证集
        split = int(n_samples * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train_reg, y_val_reg = y_reg[:split], y_reg[split:]
        y_train_cls, y_val_cls = y_cls[:split], y_cls[split:]
        
        feature_names = [f"feat_{i}" for i in range(n_features)]
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
        X_val_df = pd.DataFrame(X_val, columns=feature_names)
        
        # 测试回归模型训练
        print("  - 训练回归模型...")
        reg_model = LGBModelWrapper(task_type="regression")
        reg_model.train(X_train_df, y_train_reg, X_val_df, y_val_reg, 
                        feature_names=feature_names, early_stopping_rounds=10)
        pred_reg = reg_model.predict(X_val_df)
        assert len(pred_reg) == len(y_val_reg), "回归预测长度不匹配"
        print(f"    预测范围: [{pred_reg.min():.4f}, {pred_reg.max():.4f}]")
        
        # 测试分类模型训练
        print("  - 训练分类模型...")
        cls_model = LGBModelWrapper(task_type="classification")
        cls_model.train(X_train_df, y_train_cls, X_val_df, y_val_cls, 
                        feature_names=feature_names, early_stopping_rounds=10)
        pred_cls = cls_model.predict(X_val_df)
        assert len(pred_cls) == len(y_val_cls), "分类预测长度不匹配"
        assert pred_cls.min() >= 0 and pred_cls.max() <= 1, "分类概率应在 [0, 1]"
        print(f"    预测概率范围: [{pred_cls.min():.4f}, {pred_cls.max():.4f}]")
        
        # 测试特征重要性
        importance = reg_model.get_feature_importance()
        assert len(importance) > 0, "特征重要性为空"
        print(f"    特征重要性: {len(importance)} 个特征")
        
        log_test("模型训练与预测", True)
        return reg_model, cls_model, pred_reg, pred_cls
    except Exception as e:
        import traceback
        traceback.print_exc()
        log_test("模型训练与预测", False, str(e))
        return None, None, None, None

# ========================================
# 测试 4: 预测融合逻辑
# ========================================
def test_prediction_fusion(pred_reg, pred_cls):
    """测试预测融合逻辑"""
    try:
        from src.model.trainer import ModelTrainer
        
        if pred_reg is None or pred_cls is None:
            log_test("预测融合", False, "缺少预测结果")
            return
        
        trainer = ModelTrainer()
        
        # 测试加权平均融合
        fused = trainer.fuse_predictions(pred_reg, pred_cls)
        assert len(fused) == len(pred_reg), "融合结果长度不匹配"
        print(f"  - 融合结果范围: [{fused.min():.4f}, {fused.max():.4f}]")
        
        # 验证权重应用
        reg_weight = trainer.dual_head_cfg.get("regression", {}).get("weight", 0.6)
        cls_weight = trainer.dual_head_cfg.get("classification", {}).get("weight", 0.4)
        print(f"  - 使用权重: 回归={reg_weight}, 分类={cls_weight}")
        
        # 对比融合方法
        trainer.dual_head_cfg["fusion"]["method"] = "multiplicative"
        fused_mult = trainer.fuse_predictions(pred_reg, pred_cls)
        print(f"  - 乘法融合范围: [{fused_mult.min():.4f}, {fused_mult.max():.4f}]")
        
        log_test("预测融合", True)
    except Exception as e:
        log_test("预测融合", False, str(e))

# ========================================
# 测试 5: 分类标签生成
# ========================================
def test_classification_label_generation():
    """测试分类标签生成"""
    try:
        from src.utils.config import GLOBAL_CONFIG
        from src.preprocessing.labels import LabelGenerator
        
        # 创建临时配置，启用双头模型
        test_config = GLOBAL_CONFIG.copy()
        test_config["model"] = test_config.get("model", {}).copy()
        test_config["model"]["dual_head"] = {
            "enable": True,
            "classification": {
                "enable": True,
                "label_mode": "absolute",
                "threshold": 0.0
            }
        }
        
        # 生成合成数据
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        df = pd.DataFrame({
            "date": dates,
            "symbol": "000001.SZ",
            "open": 10 + np.random.randn(100) * 0.5,
            "high": 10.5 + np.random.randn(100) * 0.5,
            "low": 9.5 + np.random.randn(100) * 0.5,
            "close": 10 + np.random.randn(100) * 0.5,
            "volume": np.random.randint(1000000, 5000000, 100),
            "amount": np.random.randint(10000000, 50000000, 100)
        })
        
        # 生成标签
        label_gen = LabelGenerator(test_config)
        df_with_labels = label_gen.run(df)
        
        # 检查分类标签
        assert "label_cls" in df_with_labels.columns, "缺少 label_cls 列"
        label_cls = df_with_labels["label_cls"].dropna()
        assert set(label_cls.unique()).issubset({0, 1}), "label_cls 应只包含 0 和 1"
        
        pos_ratio = (label_cls == 1).mean()
        print(f"  - 正样本比例: {pos_ratio:.1%}")
        print(f"  - 标签样本数: {len(label_cls)}")
        
        log_test("分类标签生成", True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        log_test("分类标签生成", False, str(e))

# ========================================
# 测试 6: 模型保存与加载
# ========================================
def test_model_save_load(model):
    """测试模型保存和加载"""
    import tempfile
    try:
        if model is None:
            log_test("模型保存/加载", False, "没有可用的模型")
            return
        
        from src.model.lgb_model import LGBModelWrapper
        
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            temp_path = f.name
        
        try:
            # 保存模型
            model.save(temp_path)
            assert os.path.exists(temp_path), "保存失败：文件不存在"
            print(f"  - 模型已保存: {temp_path}")
            
            # 加载模型
            loaded_model = LGBModelWrapper()
            loaded_model.load(temp_path)
            
            assert loaded_model.model is not None, "加载失败：模型为空"
            assert loaded_model.task_type == model.task_type, "加载失败：任务类型不匹配"
            print(f"  - 模型已加载, 类型: {loaded_model.task_type}")
            
            log_test("模型保存/加载", True)
        finally:
            # 清理
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        log_test("模型保存/加载", False, str(e))

# ========================================
# 测试 7: 检查真实数据中的分类标签
# ========================================
def test_real_data_labels():
    """检查已处理数据中是否存在分类标签"""
    try:
        from src.utils.io import read_parquet
        from src.utils.config import GLOBAL_CONFIG
        
        data_path = os.path.join(GLOBAL_CONFIG["paths"]["data_processed"], "all_stocks.parquet")
        
        if not os.path.exists(data_path):
            log_test("真实数据检查", False, f"数据文件不存在: {data_path}")
            return
        
        df = read_parquet(data_path)
        
        has_label = "label" in df.columns
        has_label_cls = "label_cls" in df.columns
        
        print(f"  - 数据行数: {len(df):,}")
        print(f"  - label 列: {'✓' if has_label else '✗'}")
        print(f"  - label_cls 列: {'✓' if has_label_cls else '✗'}")
        
        if has_label_cls:
            pos_ratio = (df["label_cls"] == 1).mean()
            print(f"  - 分类标签正样本比例: {pos_ratio:.1%}")
            log_test("真实数据检查", True)
        else:
            log_test("真实数据检查", False, 
                    "未找到 label_cls 列。请启用双头模型后重新运行 pipeline")
            
    except Exception as e:
        log_test("真实数据检查", False, str(e))

# ========================================
# 主函数
# ========================================
def main():
    print("=" * 60)
    print("双头模型功能测试")
    print("=" * 60)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 运行测试
    print("\n[1] 测试配置加载")
    print("-" * 40)
    dual_head_cfg = test_config_loading()
    
    print("\n[2] 测试 LGBModelWrapper 初始化")
    print("-" * 40)
    test_lgb_model_wrapper()
    
    print("\n[3] 测试模型训练与预测")
    print("-" * 40)
    reg_model, cls_model, pred_reg, pred_cls = test_model_training()
    
    print("\n[4] 测试预测融合")
    print("-" * 40)
    test_prediction_fusion(pred_reg, pred_cls)
    
    print("\n[5] 测试分类标签生成")
    print("-" * 40)
    test_classification_label_generation()
    
    print("\n[6] 测试模型保存/加载")
    print("-" * 40)
    test_model_save_load(reg_model)
    
    print("\n[7] 检查真实数据")
    print("-" * 40)
    test_real_data_labels()
    
    # 打印摘要
    print_summary()
    
    # 返回测试结果
    all_passed = all(p for _, p, _ in test_results)
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
