# 双头模型系统设计文档 (Dual-Head Model)

## 1. 概述

双头模型系统是一种集成学习方法，结合**回归模型**（预测收益率）和**分类模型**（预测涨跌方向）的优势，通过加权融合提高选股胜率。

> **技术选型**：使用 **LightGBM** 作为基础模型，支持 GPU 加速。

### 1.1 设计目标

| 目标 | 说明 |
|------|------|
| 提高胜率 | 分类头帮助过滤假信号，减少选中下跌股的概率 |
| 保持收益弹性 | 回归头负责排序，确保选中涨幅最大的股票 |
| 灵活配置 | 通过权重调节，适应不同市场风格 |

### 1.2 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                        特征工程 (共用特征)                    │
│                    (data/processed/)                        │
└───────────────────────────┬─────────────────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        ▼                                       ▼
┌───────────────────┐                   ┌───────────────────┐
│   回归头 (Reg)     │                   │   分类头 (Cls)     │
│ ─────────────────  │                   │ ─────────────────  │
│ 模型: LightGBM     │                   │ 模型: LightGBM     │
│ 标签: 超额收益      │                   │ 标签: 涨跌(0/1)    │
│ 目标: regression   │                   │ 目标: binary       │
│ 输出: pred_reg     │                   │ 输出: pred_cls     │
└─────────┬─────────┘                   └─────────┬─────────┘
          │                                       │
          └───────────────┬───────────────────────┘
                          ▼
              ┌───────────────────────┐
              │     融合层 (Fusion)    │
              │  ───────────────────  │
              │ pred_score =          │
              │   α × norm(pred_reg)  │
              │ + β × norm(pred_cls)  │
              └───────────┬───────────┘
                          ▼
              ┌───────────────────────┐
              │     信号生成          │
              │   TopKSignalStrategy  │
              └───────────────────────┘
```

---

## 2. 配置说明

### 2.1 主配置 (`config/main.yaml`)

```yaml
model:
  dual_head:
    enable: true                    # 是否启用双头系统
    model_type: "lightgbm"          # 模型类型: lightgbm / xgboost
    
    # 回归头配置
    regression:
      enable: true
      label_mode: "excess_index"    # excess_index (超额) / absolute (绝对)
      weight: 0.6                   # 融合权重 α
    
    # 分类头配置
    classification:
      enable: true
      label_mode: "absolute"        # absolute / excess_index (默认绝对涨跌)
      threshold: 0.0                # >threshold 为涨 (label=1)，≤threshold 为跌 (label=0)
      weight: 0.4                   # 融合权重 β
    
    # 融合策略
    fusion:
      method: "weighted_average"    # 融合方法
      normalize: true               # 融合前是否归一化
```

### 2.2 配置参数详解

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dual_head.enable` | bool | false | 是否启用双头系统 |
| `dual_head.model_type` | str | "lightgbm" | 模型类型 |
| `regression.label_mode` | str | "excess_index" | 回归标签模式 |
| `regression.weight` | float | 0.6 | 回归模型在融合中的权重 |
| `classification.label_mode` | str | "absolute" | 分类标签模式 |
| `classification.threshold` | float | 0.0 | 涨跌分类阈值 (需运行阈值分析确定) |
| `classification.weight` | float | 0.4 | 分类模型在融合中的权重 |
| `fusion.method` | str | "weighted_average" | 融合方法 |
| `fusion.normalize` | bool | true | 是否归一化后融合 |

---

## 3. 标签生成

### 3.1 回归标签 (`label`)

沿用现有逻辑，计算 N 日超额收益：

```
label = (Exit_Price / Entry_Price - 1) - Index_Return
```

### 3.2 分类标签 (`label_cls`)

新增二分类标签生成：

```python
# 绝对涨跌模式
raw_ret = (Exit_Price / Entry_Price - 1)

# 或 超额涨跌模式  
raw_ret = label  # 使用已计算的超额收益

# 生成 0/1 标签
label_cls = 1 if raw_ret > threshold else 0
```

### 3.3 标签分布建议

理想的分类标签分布应接近平衡：

| 类别 | 占比 | 说明 |
|------|------|------|
| 0 (跌) | 45%-55% | 下跌或涨幅不达阈值 |
| 1 (涨) | 45%-55% | 上涨超过阈值 |

如果分布严重不平衡，可以调整 `threshold` 参数。

---

## 4. 模型训练

### 4.1 回归模型

| 配置项 | 值 |
|--------|-----|
| 目标函数 | `reg:squarederror` |
| 评估指标 | `rmse` |
| 输出 | 连续值 (预测收益率) |

### 4.2 分类模型

| 配置项 | 值 |
|--------|-----|
| 目标函数 | `binary:logistic` |
| 评估指标 | `auc` |
| 输出 | 概率值 [0, 1] |

### 4.3 模型文件

训练完成后，`data/models/{version}/` 目录包含：

```
data/models/20241220_120000/
├── model_reg.json       # 回归模型
├── model_cls.json       # 分类模型
├── predictions.parquet  # 融合预测结果
└── ...
```

---

## 5. 预测融合

### 5.1 归一化

融合前对两个模型的输出进行归一化：

```python
def normalize(series):
    """Min-Max 归一化到 [0, 1]"""
    return (series - series.min()) / (series.max() - series.min() + 1e-8)
```

### 5.2 加权平均

```python
pred_score = α × norm(pred_reg) + β × norm(pred_cls)
```

其中：
- `α` = `regression.weight` (默认 0.6)
- `β` = `classification.weight` (默认 0.4)
- 约束: `α + β = 1`

### 5.3 其他融合方法（可选）

| 方法 | 公式 | 特点 |
|------|------|------|
| 加权平均 | `α×reg + β×cls` | 简单稳定 |
| 乘法融合 | `reg × cls` | 更严格过滤 |
| 投票制 | `rank_reg + rank_cls` | 秩次融合 |

---

## 6. 使用场景

### 6.1 推荐权重配置

| 市场风格 | α (回归) | β (分类) | 说明 |
|----------|----------|----------|------|
| 牛市进攻 | 0.7 | 0.3 | 更看重收益弹性 |
| 震荡市 | 0.5 | 0.5 | 平衡收益与胜率 |
| 熊市防守 | 0.3 | 0.7 | 更看重方向正确 |

### 6.2 单头回退

如需禁用双头系统：

```yaml
model:
  dual_head:
    enable: false
```

系统将回退到原有的单回归模型逻辑。

---

## 7. 后续优化方向

1. **动态权重调整**：根据市场环境自动调节 α/β
2. **多头扩展**：增加波动率预测头
3. **特征隔离**：为两个头使用不同的特征子集
4. **模型蒸馏**：用双头预测训练单一模型

---

## 8. 相关文件

| 文件 | 说明 |
|------|------|
| [config/main.yaml](../config/main.yaml) | 主配置文件 |
| [src/preprocessing/labels.py](../src/preprocessing/labels.py) | 标签生成器 |
| [src/model/trainer.py](../src/model/trainer.py) | 模型训练器 |
| [src/model/xgb_model.py](../src/model/xgb_model.py) | XGBoost 包装器 |
| [src/strategy/signal.py](../src/strategy/signal.py) | 信号策略 |
