# Quant_A_Share_V2.0

<p align="center">
  <img src="architecture/test_result/backtest_result/daily_comparison.png" width="700" alt="滚动周期收益">
</p>

## 🎯 项目简介

**Quant_A_Share_V2.0** 是一个**基于机器学习的 A 股量化决策系统**，为个人投资者打造从数据获取到实盘交易的完整闭环。项目强调工程化、透明化和可验证性，致力于解决散户在量化投资中面临的"数据乱、回测假、落地难"三大痛点。

### 💡 核心特性

| 特性 | 说明 |
|------|------|
| 🔄 **工业级数据流水线** | 多源数据获取（Baostock/Akshare）、自动清洗、质量检测、断点续传 |
| 🤖 **双头模型架构** | 收益预测 + 风险预测，融合更稳健的选股信号（支持XGBoost/LightGBM） |
| 📊 **严谨验证体系** | EDA分析、IC检测、压力测试、蒙特卡洛模拟、过拟合检测 |
| 🎯 **智能选股策略** | Top-K选股、动态仓位管理、风险过滤、VWAP计价 |
| 📈 **Walk-Forward验证** | 滚动训练、扩张窗口、避免未来函数 |
| 🔍 **模型可解释性** | SHAP分析、特征重要性、因子归因 |
| 📊 **TensorBoard监控** | 实时训练监控、损失曲线、特征重要性可视化 |
| 🚀 **实盘交易对接** | 雪球组合自动交易，打通研究到交易的最后一公里 |

---

## 📈 项目流程

<p align="center">
  <img src="architecture/data_clean/QUANT_A_SHARE_V2.png" width="800" alt="项目流程图">
</p>

---

## 🛠️ 技术栈

### 核心依赖

| 类别 | 技术 | 版本 | 说明 |
|------|------|------|------|
| **编程语言** | Python | ≥3.10 | 核心开发语言 |
| **数据处理** | Pandas | ≥2.0 | 数据分析与处理 |
| | NumPy | ≥1.23 | 数值计算 |
| | PyArrow | ≥14.0 | Parquet高效读写 |
| **数据源** | Baostock | ≥0.8.8 | 主要数据源（免费、稳定） |
| | Akshare | ≥1.11.80 | 备用数据源（功能丰富） |
| **机器学习** | XGBoost | ≥2.0.3 | 核心模型（支持GPU加速） |
| | Scikit-learn | ≥1.2 | 数据预处理、评估指标 |
| | SHAP | ≥0.40.0 | 模型可解释性分析 |
| **训练监控** | TensorBoard | ≥2.15 | 可视化训练过程 |
| | TensorboardX | ≥2.6 | TensorBoard日志记录 |
| **可视化** | Matplotlib | ≥3.7 | 基础绘图 |
| | Seaborn | ≥0.12 | 统计图表 |
| **实盘交易** | Easytrader | ≥0.23.0 | 券商交易接口 |
| | Schedule | ≥1.1.0 | 定时任务调度 |
| **配置管理** | PyYAML | ≥6.0 | YAML配置文件解析 |

### 可选依赖

- **GPU加速**: CuPy (根据CUDA版本选择)
- **多模型融合**: LightGBM, CatBoost (未来计划)
- **并行计算**: Joblib (未来计划)

---

## 🚀 快速开始

### 1. 环境安装

```bash
# 建议使用 Python 3.10+
pip install -r requirements.txt
```

### 2. 数据获取流水线

```bash
# Step 0: 初始化股票池与交易日历
python scripts/date_landing/init_stock_pool.py

# Step 1: 下载原始行情数据 (支持断点续传)
python scripts/date_landing/download_data.py

# Step 2: 数据清洗与质检
python scripts/analisis/clean_and_check.py

# Step 3: 构建特征与标签
python scripts/feature_create/rebuild_features.py
```

### 3. 模型训练与回测

```bash
# Step 4: 滚动训练 (Walk-Forward，推荐)
python scripts/model_train/run_walkforward.py

# Step 5: 策略回测
python scripts/back_test/run_backtest.py

# 生成每日推荐
python scripts/back_test/run_recommendation.py
```

### 4. 实盘交易 (可选)

```bash
# 模拟测试
python scripts/live/run_auto_trading.py

# 实盘运行
python scripts/live/run_auto_trading.py --real
```

---

## 📁 项目结构

```
Quant_A_Share_V2.0/
├── config/                    # 配置文件
│   └── main.yaml             # 主配置文件（路径、数据源、模型、策略等）
├── data/                      # 数据目录
│   ├── raw/                  # 原始行情数据
│   ├── raw_cleaned/          # 清洗后的数据
│   ├── processed/            # 特征工程后的数据
│   ├── meta/                 # 元数据（股票池、交易日历等）
│   └── models/               # 模型文件与预测结果
├── src/                       # 核心源代码
│   ├── data_source/          # 数据源模块
│   │   ├── base.py          # 数据源抽象基类
│   │   ├── baostock_source.py  # Baostock数据源
│   │   ├── akshare_source.py   # Akshare数据源
│   │   └── datahub.py       # 统一数据接口
│   ├── preprocessing/        # 数据预处理
│   │   ├── features.py      # 特征工程（MA/MACD/RSI/KDJ/BOLL等）
│   │   ├── labels.py        # 标签生成（VWAP计价、超额收益）
│   │   ├── neutralization.py # 特征中性化
│   │   └── pipeline.py      # 数据处理流水线
│   ├── model/                # 模型训练
│   │   ├── xgb_model.py     # XGBoost模型封装
│   │   ├── lgb_model.py     # LightGBM模型封装
│   │   ├── trainer.py       # 训练器（支持双头模型）
│   │   └── training_monitor.py # TensorBoard监控
│   ├── strategy/             # 交易策略
│   │   └── signal.py        # 信号生成（Top-K选股、风险过滤）
│   ├── backtest/             # 回测引擎
│   │   └── backtester.py    # 回测主逻辑
│   ├── analysis/             # 分析工具
│   │   ├── eda_engine.py    # 探索性数据分析
│   │   ├── factor_checker.py # 因子分析
│   │   ├── model_interpreter.py # 模型解释（SHAP）
│   │   └── horizon_analyzer.py  # 时间窗口分析
│   ├── live/                 # 实盘交易
│   │   ├── xueqiu_broker.py # 雪球交易接口
│   │   ├── trading_scheduler.py # 交易调度器
│   │   └── trade_recorder.py    # 交易记录
│   └── utils/                # 工具函数
│       ├── config.py        # 配置加载
│       ├── logger.py        # 日志管理
│       └── io.py            # 文件读写
├── scripts/                   # 可执行脚本
│   ├── date_landing/         # 数据下载
│   ├── analisis/             # 数据分析
│   ├── feature_create/       # 特征构建
│   ├── model_train/          # 模型训练
│   ├── back_test/            # 回测脚本
│   ├── live/                 # 实盘脚本
│   └── tools/                # 辅助工具
├── docs/                      # 文档
│   ├── EDA_GUIDE.md         # EDA分析指南
│   ├── FACTOR_ANALYSIS_REPORT.md # 因子分析报告
│   ├── HORIZON_ANALYSIS_REPORT.md # 时间窗口分析
│   └── XUEQIU_GUIDE.md      # 雪球交易指南
├── architecture/              # 架构文档与测试结果
├── logs/                      # 日志文件
├── requirements.txt           # Python依赖
├── architecture.md            # 架构说明文档
└── README.md                  # 项目说明
```

---

## 🔧 核心模块详解

### 1️⃣ 数据源模块 (src/data_source/)

**设计理念**: 统一接口，数据源可替换

- **DataHub**: 统一数据入口，屏蔽底层数据源差异
- **Baostock**: 主要数据源，免费稳定，支持日线/分钟线
- **Akshare**: 备用数据源，功能更丰富，支持财务数据

**核心功能**:
- 多源数据获取与自动切换
- 断点续传，避免重复下载
- 数据质量检测（缺失值、异常值、停牌检测）

### 2️⃣ 预处理模块 (src/preprocessing/)



**特征工程** (features.py):
- **技术指标**: MA(5/10/20/60)、MACD、RSI、KDJ、BOLL
- **量价特征**: 成交量、换手率、量价相关性
- **波动率**: ATR、标准差、下行波动率
- **动量因子**: 短期/中期/长期动量、斜率因子

**标签生成** (labels.py):
- **VWAP计价**: 使用成交量加权平均价，更接近实盘
- **超额收益**: 相对指数（沪深300/中证500）的超额收益
- **一字板过滤**: 自动过滤无法交易的涨跌停板
- **标签平滑**: 3日均值平滑，降低噪音
- **风险标签**: 下行波动率、最大回撤等风险指标

**特征中性化** (neutralization.py):
- 剔除市值、行业等风险因子的影响
- 提取纯alpha信号

### 3️⃣ 模型训练模块 (src/model/)

**双头模型架构**:
- **收益预测头**: 预测未来N日收益率
- **风险预测头**: 预测未来N日下行波动率
- **融合策略**: Sharpe式比率、效用函数、加权平均

**训练特性**:
- **Walk-Forward验证**: 滚动训练，避免未来函数
- **GPU加速**: XGBoost支持CUDA加速
- **早停机制**: 防止过拟合
- **TensorBoard监控**: 实时查看训练曲线
- **特征筛选**: 基于IC和相关性自动筛选特征

### 4️⃣ 策略模块 (src/strategy/)

**选股逻辑**:
- **Top-K选股**: 选择预测得分最高的K只股票
- **风险过滤**: 排除ST、新股、涨跌停、低流动性股票
- **动态仓位**: 根据市场趋势（MA20/MA60）调整仓位
- **分仓轮动**: 每日轮换1/N仓位，平滑资金曲线

<p align="center">
  <img src="architecture/test_result/signals/20260119_173534/score_distribution.png" width="400" alt="预测得分分布">
  <img src="architecture/test_result/signals/20260119_173534/ic_by_month.png" width="400" alt="月度IC值">
</p>

**交易执行**:
- **VWAP买入**: 使用T+1日均价买入
- **成本扣除**: 考虑佣金、印花税、滑点
- **持仓管理**: 自动调仓、止盈止损

### 5️⃣ 回测模块 (src/backtest/)

**回测功能**:
- 完整的资金曲线模拟
- 多维度绩效指标（年化收益、夏普比率、最大回撤）
- 分年度统计分析
- 持仓分析与换手率统计

**验证体系**:
- **压力测试**: 熊市生存能力、成本敏感性
- **蒙特卡洛模拟**: 评估策略稳健性
- **过拟合检测**: 随机标签对比测试

<p align="center">
  <img src="architecture/test_result/overfit_test/overfit_test_comparison.png" width="450" alt="过拟合检测对比">
  <img src="architecture/test_result/overfit_test/random_equity_curve.png" width="450" alt="随机信号资金曲线">
</p>

### 6️⃣ 分析模块 (src/analysis/)

**EDA分析** (eda_engine.py):
- 数据质量检测（缺失值、异常值、停牌率）
- 收益率分布分析
- 流动性分析
- 自相关性检测

**因子分析** (factor_checker.py):
- IC值计算（信息系数）
- 特征相关性分析
- 特征重要性排序

**时间窗口分析** (horizon_analyzer.py):
- IC衰减曲线分析
- 最优预测周期研究

<p align="center">
  <img src="architecture/test_result/horizon_analysis/20251208_012101/ic_decay_curve.png" width="600" alt="IC衰减曲线">
</p>

**模型解释** (model_interpreter.py):
- SHAP值分析
- 特征贡献度可视化
- 依赖关系图

### 7️⃣ 实盘交易模块 (src/live/)

**雪球组合交易**:
- 自动登录与Cookie管理
- 持仓查询与调仓
- 交易记录与监控
- 定时任务调度

---

## ⚙️ 配置说明

项目使用单一配置文件 `config/main.yaml`，包含以下主要配置模块：

### 核心配置项

| 配置模块 | 说明 | 关键参数 |
|---------|------|----------|
| **paths** | 路径配置 | data_root, models, logs |
| **data** | 数据源配置 | source (baostock/akshare), start_date, end_date |
| **preprocessing** | 预处理配置 | 特征开关、标签类型、过滤规则 |
| **model** | 模型配置 | XGBoost参数、双头模型、Walk-Forward设置 |
| **strategy** | 策略配置 | top_k, 仓位管理、换仓模式 |
| **backtest** | 回测配置 | 交易成本、回测窗口 |

### 重要参数说明

**模型参数** (model.params):
```yaml
n_estimators: 3000        # 迭代次数
max_depth: 5              # 树深度（降低过拟合）
learning_rate: 0.01       # 学习率
reg_alpha: 0.1            # L1正则化
reg_lambda: 1.0           # L2正则化
```

**双头模型** (model.dual_head):
```yaml
enable: true              # 启用双头模型
fusion.method: "sharpe_like"  # 融合方法
return_head.weight: 0.6   # 收益权重
risk_head.weight: 0.4     # 风险权重
```

**策略参数** (strategy):
```yaml
top_k: 5                  # 持仓股票数
entry_price: "vwap"       # 买入价格模式
rebalance_mode: "periodic" # 换仓模式
```

---

## 📊 分析工具

### EDA分析
```bash
python scripts/analisis/run_eda.py
```
生成数据质量报告，包括：缺失值统计、收益率分布、流动性分析、异常值检测

<p align="center">
  <img src="architecture/test_result/eda/20251207_175729/dist_returns.png" width="400" alt="收益率分布">
  <img src="architecture/test_result/eda/20251207_175729/dist_liquidity.png" width="400" alt="流动性分析">
</p>

### 因子分析
```bash
python scripts/analisis/check_features.py
```
输出：IC值排名（Top 30特征）、特征相关性热力图、标签分布图

<p align="center">
  <img src="architecture/test_result/factors/20251208_012005/feature_ic_top30.png" width="400" alt="特征IC值排名">
  <img src="architecture/test_result/factors/20251208_012005/feature_correlation.png" width="400" alt="特征相关性">
</p>
<p align="center">
  <img src="architecture/test_result/factors/20251208_012005/dist_label.png" width="400" alt="标签分布">
</p>

### 模型解释
```bash
python scripts/analisis/explain_model.py
```
生成SHAP分析报告：特征重要性排序、SHAP值分布图、特征依赖关系图

<p align="center">
  <img src="architecture/test_result/interpretation/shap_summary_bar.png" width="400" alt="SHAP特征重要性">
  <img src="architecture/test_result/interpretation/shap_summary_beeswarm.png" width="400" alt="SHAP值分布">
</p>
<p align="center">
  <img src="architecture/test_result/interpretation/shap_dependence_feat_macd.png" width="400" alt="MACD依赖关系">
  <img src="architecture/test_result/interpretation/shap_dependence_feat_boll_pos.png" width="400" alt="BOLL依赖关系">
</p>

### 压力测试
```bash
python scripts/analisis/check_stress_test.py
```
测试场景：熊市生存测试（2018贸易战、2020疫情等）、成本敏感性测试、流动性冲击测试

<p align="center">
  <img src="architecture/test_result/stress_test/cost_sensitivity_comparison.png" width="600" alt="成本敏感性测试">
</p>
<p align="center">
  <img src="architecture/test_result/stress_test/crisis_2020_Covid-19/equity_curve.png" width="400" alt="2020疫情压力测试">
  <img src="architecture/test_result/stress_test/crisis_2018_Trade_War/equity_curve.png" width="400" alt="2018贸易战压力测试">
</p>

### 蒙特卡洛模拟
```bash
python scripts/analisis/check_monte_carlo.py
```
评估策略稳健性：1000次随机模拟、收益率分布、风险指标统计

<p align="center">
  <img src="architecture/test_result/monte_carlo/monte_carlo_distribution.png" width="400" alt="蒙特卡洛收益分布">
  <img src="architecture/test_result/monte_carlo/noise_sensitivity.png" width="400" alt="噪音敏感性测试">
</p>

---

## 🎯 使用场景

### 场景1: 研究模式（离线分析）

```bash
# 1. 更新数据（按周/月）
python scripts/date_landing/update_data.py
python scripts/feature_create/rebuild_features.py

# 2. 训练新模型
python scripts/model_train/run_walkforward.py

# 3. 回测验证
python scripts/back_test/run_backtest.py

# 4. 分析结果
python scripts/analisis/explain_model.py
python scripts/analisis/check_stress_test.py
```

### 场景2: 日常推荐模式

```bash
# 生成每日推荐
python scripts/back_test/run_recommendation.py
```
- 输出文件: `data/recommendations/latest.csv`
- 包含: 股票代码、预测得分、风险评级

**推荐股票特征分析**:

<p align="center">
  <img src="architecture/test_result/signals/20260119_173534/price_distribution.png" width="400" alt="价格分布">
  <img src="architecture/test_result/signals/20260119_173534/mcap_distribution.png" width="400" alt="市值分布">
</p>
<p align="center">
  <img src="architecture/test_result/signals/20260119_173534/turnover_distribution.png" width="400" alt="换手率分布">
  <img src="architecture/test_result/signals/20260119_173534/volatility_distribution.png" width="400" alt="波动率分布">
</p>
<p align="center">
  <img src="architecture/test_result/signals/20260119_173534/short_term_momentum.png" width="400" alt="短期动量分析">
  <img src="architecture/test_result/signals/20260119_173534/reversal_analysis.png" width="400" alt="反转效应分析">
</p>

### 场景3: 实盘交易模式（进阶）

```bash
# 1. 配置雪球Cookie
python scripts/live/check_xq_cookie.py

# 2. 模拟测试
python scripts/live/run_auto_trading.py

# 3. 实盘运行
python scripts/live/run_auto_trading.py --real
```

---

## ❓ 常见问题 (FAQ)

<details>
<summary><b>Q1: 如何选择数据源？</b></summary>

默认使用Baostock（免费、稳定）。如需更多数据，可在 `config/main.yaml` 中切换：
```yaml
data:
  source: "akshare"  # 或 "baostock"
```
</details>

<details>
<summary><b>Q2: 训练时间太长怎么办？</b></summary>

1. 启用GPU加速（需安装CUDA）
2. 减少 `n_estimators`（如改为1000）
3. 减小训练数据范围（修改 `start_date`）
</details>

<details>
<summary><b>Q3: 如何调整持仓数量？</b></summary>

修改 `config/main.yaml` 中的 `strategy.top_k`：
```yaml
strategy:
  top_k: 10  # 持仓10只股票
```
</details>

<details>
<summary><b>Q4: 回测收益不理想怎么办？</b></summary>

1. 检查特征IC值（运行因子分析）
2. 调整模型参数（降低学习率、增加正则化）
3. 优化选股逻辑（提高 `min_pred` 阈值）
4. 启用动态仓位管理
</details>

<details>
<summary><b>Q5: 如何避免过拟合？</b></summary>

1. 使用Walk-Forward验证（已内置）
2. 增加正则化参数（`reg_alpha`, `reg_lambda`）
3. 运行过拟合检测：`python scripts/analisis/check_overfit.py`
4. 进行压力测试和蒙特卡洛模拟
</details>

<details>
<summary><b>Q6: 实盘交易安全吗？</b></summary>

1. 建议先在模拟盘测试
2. 设置合理的仓位上限
3. 定期检查交易记录
4. 使用雪球组合（非实际券商账户）
</details>

---

## 📈 回测结果展示

### 资金曲线对比

<p align="center">
  <img src="architecture/test_result/backtest_result/equity_curve.png" width="700" alt="策略资金曲线">
</p>

### 滚动周期收益分析

<p align="center">
  <img src="architecture/test_result/backtest_result/daily_comparison.png" width="700" alt="滚动周期收益分析">
</p>

### 随机周期分析

<p align="center">
  <img src="architecture/test_result/backtest_result/random_analysis.png" width="700" alt="随机周期分析">
</p>

---

## ✅ 已完成功能

| 功能 | 状态 | 说明 |
|------|:----:|------|
| 数据下载与清洗 | ✅ | 多源支持、断点续传、智能质检 |
| 特征工程 | ✅ | MA/MACD/RSI/KDJ/BOLL/量价/波动率等 |
| 标签生成 | ✅ | VWAP计价、超额收益、一字板过滤、风险标签 |
| 特征中性化 | ✅ | 市值/行业中性化 |
| XGBoost 训练 | ✅ | GPU加速、早停机制 |
| 双头模型 | ✅ | 收益预测 + 风险预测 |
| Walk-Forward 滚动训练 | ✅ | 按年滚动、无未来函数 |
| TensorBoard 训练监控 | ✅ | 损失曲线、特征重要性可视化 |
| 特征筛选 | ✅ | 基于 IC 和相关性自动筛选 |
| 正则化优化 | ✅ | L1/L2 正则化、参数调优 |
| 策略回测 | ✅ | 分仓轮动、成本扣除 |
| 压力测试 | ✅ | 成本敏感性、熊市生存测试 |
| 蒙特卡洛模拟 | ✅ | 策略稳健性评估 |
| 过拟合检测 | ✅ | 随机标签对比测试 |
| SHAP可解释性 | ✅ | 特征贡献度分析 |
| 每日推荐 | ✅ | 自动选股、涨停过滤 |
| 雪球实盘交易 | ✅ | 自动调仓、交易记录 |

---

## 🗺️ 未来计划 (Roadmap)

### P1 (近期优先)
- [ ] 交互式回测报告：生成 HTML 格式报告
- [ ] 模型版本对比工具
- [ ] 因子挖掘框架

### P2 (中期计划)
- [ ] 多模型融合：XGBoost + LightGBM + CatBoost
- [ ] 行业轮动策略
- [ ] 更多数据源支持（Tushare等）

### P3 (远期愿景)
- [ ] 深度学习模型：LSTM / Transformer
- [ ] 券商API对接（非雪球）
- [ ] Web Dashboard：图形化管理平台

---

## 📝 更新日志

| 日期 | 版本 | 更新内容 |
|------|------|----------|
| 2025-12-15 | V2.1 | 添加 TensorBoard 监控、特征筛选、正则化优化、双头模型 |
| 2025-12-14 | V2.0 | Walk-Forward 滚动训练、压力测试、蒙特卡洛模拟 |
| 2025-12-07 | V1.5 | 策略回测、每日推荐、SHAP分析 |
| 2025-12-01 | V1.0 | 初始版本，数据下载与特征工程 |

---

## 📚 相关文档

- [架构说明](architecture.md) - 详细的系统架构设计
- [EDA分析指南](docs/EDA_GUIDE.md) - 探索性数据分析教程
- [因子分析报告](docs/FACTOR_ANALYSIS_REPORT.md) - 特征有效性分析
- [时间窗口分析](docs/HORIZON_ANALYSIS_REPORT.md) - 预测周期研究
- [雪球交易指南](docs/XUEQIU_GUIDE.md) - 实盘交易配置说明

---

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源协议。

---

## 🙏 致谢

- [Baostock](http://baostock.com/) - 免费的A股数据接口
- [Akshare](https://akshare.xyz/) - 丰富的金融数据源
- [XGBoost](https://xgboost.ai/) - 高效的梯度提升框架
- [SHAP](https://shap.readthedocs.io/) - 模型可解释性工具

---

<p align="center">
  <b>⭐ 如果这个项目对你有帮助，请给一个 Star！⭐</b>
</p>
