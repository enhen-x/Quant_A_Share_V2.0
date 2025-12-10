# Quant_A_Share_V2.0


# 🎯 项目目标 (Project Goals)

**Quant_A_Share_V2.0** 致力于为个人投资者打造一个**基于机器学习的 A 股量化决策系统**。本项目不追求复杂的黑盒魔法，而是强调工程化、透明化和可验证性，旨在解决散户在量化投资中面临的“数据乱、回测假、落地难”三大痛点。

核心目标包括：

1.  **构建工业级数据流水线**：实现从数据下载、清洗、质检到特征工程的全自动化闭环，确保输入模型的每一条数据都干净、可靠。
2.  **机器学习驱动选股**：摒弃传统的线性因子叠加，利用 XGBoost 等机器学习模型挖掘非线性市场规律，支持**三相屏障 (Triple Barrier)** 等机构级标签构建方法。
3.  **严谨的策略验证体系**：提供**探索性数据分析 (EDA)**、**因子有效性检测 (IC)** 和**防未来函数回测**等多维度的评估工具，拒绝“过拟合”的虚假回测。
4.  **实战导向的推荐能力**：不仅能跑回测，更能每日生成可执行的**精选股票推荐列表**，打通从研究到交易的“最后一公里”。


项目流程图如下

![项目流程图](architecture\data_clean\QUANT_A_SHARE_V2.png)
---

# 🧪 目前进度 (Current Progress)

截至目前 (V2.0)，项目已完成**全流程闭环**的开发，核心模块均已就绪：

### ✅ 1. 数据基础设施 (Data Infrastructure)
* **多源数据获取**：支持 Baostock / AkShare 双源切换，实现全市场股票列表、交易日历及日线行情的**断点续传**与**增量更新**。
* **智能清洗质检**：内置自动清洗脚本，自动识别并剔除**高停牌率**、**流动性枯竭 (僵尸股)** 及**数据严重缺失**的标的，输出质量体检报告。

### ✅ 2. 深度分析引擎 (Analysis Engine)
* **全维 EDA 分析**：支持停牌率分布、收益率自相关性、价格走势对齐等多维度的数据“体检”。
* **因子评价体系**：实现了 IC (Information Coefficient) 分析、因子衰减 (Decay) 曲线及多重共线性检测，量化评估特征有效性。
* **时间窗口探测**：独创的时间敏感性分析，辅助决策模型预测的最佳周期 (Horizon)。

### ✅ 3. 特征与模型 (Feature & Model)
* **特征工程**：内置 MA, MACD, RSI, KDJ, Bollinger, 量比等经典技术指标库，支持向量化批量计算。
* **高级标签生成**：支持 **三相屏障法 (Triple Barrier Method)**，结合 **VWAP (成交均价)** 计算真实收益，并自动剔除一字涨停样本。
* **模型训练**：集成了 **XGBoost** 树模型，支持自动划分训练/验证集、模型保存与版本管理。

### ✅ 4. 策略与回测 (Strategy & Backtest)
* **Top-K 选股策略**：基于模型预测分，结合价格、市值、板块等风控规则进行每日精选。
* **向量化回测引擎**：实现了支持**分仓轮动**的高效回测框架，严格执行 T+1 交易逻辑，内置交易成本扣除，输出资金曲线与夏普比率。
* **每日推荐系统**：提供一键式脚本，基于最新行情生成**每日潜力股名单 (Daily Picks)**。  

---

## 📌 项目结构


```text
## 📌 项目结构 (Project Structure)

Quant_A_SHARE_V2.0/
├── config/
│   └── main.yaml               # [核心] 全局配置文件
│                               # 包含：路径、数据源、风控参数(市值/价格)、模型参数(滚动训练配置)、
│                               # 策略参数(Top-K/动态仓位/择时)、回测配置(成本/轮动模式)。
│
├── docs/                       # [文档] 分析指南
│   ├── EDA_GUIDE.md            # 探索性数据分析 (EDA) 结果解读指南
│   ├── FACTOR_ANALYSIS_REPORT.md # 因子有效性与 IC 分析解读指南
│   └── HORIZON_ANALYSIS_REPORT.md # 预测周期 (Horizon) 敏感性分析解读指南
│
├── scripts/                    # [入口] 命令行执行脚本
│   ├── init_stock_pool.py      # [Step 0] 初始化股票池与交易日历 (基于 AkShare)
│   ├── download_data.py        # [Step 1] 批量下载/断点续传行情数据
│   ├── auto_run.py             # [Step 1+] 全自动挂机下载（含防封控冷却重启逻辑）
│   ├── update_data.py          # [Daily] 增量更新数据（日历/指数/个股）
│   ├── clean_and_check.py      # [Step 2] 数据清洗与质检（去重、停牌过滤、僵尸股剔除）
│   ├── run_eda.py              # [Analysis] 运行全维度探索性分析 (EDA)
│   ├── rebuild_features.py     # [Step 3] 特征工程流水线 (Features + Labels + Filtering)
│   ├── check_features.py       # [Analysis] 因子有效性检查 (IC 分析/多重共线性/未来函数检测)
│   ├── check_time_horizon.py   # [Analysis] 最佳持仓周期分析 (IC Decay)
│   ├── run_walkforward.py      # [Step 4 - New] 滚动训练 (Walk-Forward Validation)
│   │                           # 模拟真实时间流逝，每年重新训练模型，生成无未来函数的预测集。
│   ├── train_model.py          # [Step 4 - Legacy] 单次模型训练 (仅用于快速测试)
│   ├── run_backtest.py         # [Step 5] 策略回测
│   │                           # 支持分仓轮动(Periodic)、动态仓位、严谨成交(剔除涨停/高开)。
│   ├── check_stress_test.py    # [Test - New] 策略压力测试
│   │                           # 测试策略在不同交易成本及历史极端熊市(如2024微盘股危机)下的生存能力。
│   └── run_recommendation.py   # [App] 每日推荐 (Daily Picks)
│                               # 智能寻找最新模型，生成含备选的 Top-K 股票池，自动过滤今日涨停股。
│
├── src/                        # [源码] 核心逻辑库
│   ├── analysis/               # 分析引擎模块
│   │   ├── eda_engine.py       # EDA 绘图与统计核心
│   │   ├── factor_checker.py   # 因子质量分析器 (IC/RankIC/分布)
│   │   └── horizon_analyzer.py # 多周期 IC 衰减分析器
│   │
│   ├── backtest/               # 回测模块
│   │   └── backtester.py       # 向量化回测引擎
│   │                           # 特性：支持 Periodic/Rolling 轮动、动态成本覆盖、开盘涨停废单逻辑。
│   │
│   ├── data_source/            # 数据源适配层 (Facade模式)
│   │   ├── base.py             # 接口基类
│   │   ├── datahub.py          # 数据统一调度入口
│   │   ├── akshare_source.py   # AkShare 接口实现
│   │   └── baostock_source.py  # Baostock 接口实现
│   │
│   ├── model/                  # 机器学习模块
│   │   ├── trainer.py          # 训练流程管理器 (支持单次及滚动训练调用)
│   │   └── xgb_model.py        # XGBoost 模型封装 (支持 GPU/Hist 模式)
│   │
│   ├── preprocessing/          # 预处理模块
│   │   ├── pipeline.py         # 特征工程总流水线 (含 Row-level 过滤)
│   │   ├── features.py         # 特征计算工厂 (MA, MACD, RSI, KDJ, BOLL, Vol, Cap...)
│   │   └── labels.py           # 标签生成工厂 (三相屏障/VWAP/超额收益/一字板剔除)
│   │
│   ├── strategy/               # 策略模块
│   │   └── signal.py           # 信号生成器
│   │                           # 特性：TopK 排序 + 双均线动态仓位(Position Control) + 涨停过滤 + 市值风控。
│   │
│   └── utils/                  # 通用工具库
│       ├── config.py           # 全局配置加载器 (单例)
│       ├── logger.py           # 日志管理
│       └── io.py               # 文件读写封装
│
├── data/ (自动生成目录)
│   ├── raw/                    # 原始行情数据
│   ├── raw_cleaned/            # 清洗后的标准行情
│   ├── processed/              # 最终特征矩阵 (all_stocks.parquet)
│   ├── meta/                   # 股票列表与交易日历
│   ├── models/                 # 模型仓库
│   │   ├── WF_YYYYMMDD.../     # 滚动训练生成的年度模型集与全量预测表
│   │   └── YYYYMMDD.../        # 单次训练的模型存档
│   └── index/                  # 指数数据
│
├── figures/ (自动生成)          # 分析图表 (.png)
├── reports/ (自动生成)          # 分析报告 & 每日推荐 (.csv)
├── logs/                       # 运行日志
│
├── architecture.md             # 架构设计说明
├── README.md                   # 项目说明文档
└── requirements.txt            # 项目依赖
```

---
## 🚀 Quick Start / 使用说明
#  1. 安装依赖

项目基于 Python 3.10+（建议使用 Conda 虚拟环境）：

```bash
pip install -r requirements.txt
```

---

# 2. 数据获取与更新 (Data Pipeline)

本项目采用了更稳定的分步下载策略，支持断点续传和自动挂机，配置位于 `config/main.yaml`。

#### 2.1 第一步：初始化股票池 

首先获取全市场股票名单（代码、名称）及全历史交易日历，并保存为元数据。


```bash
python scripts/init_stock_pool.py
```

**作用**：从数据源拉取最新 A 股列表与交易日历，生成 `data/meta/all_stocks_meta.parquet、data\meta\trade_calendar.parquet`
**注意**：这是后续下载的基础，首次运行或需要更新新股列表时必须执行。

2.2 第二步：批量下载行情

根据配置文件中的过滤规则（剔除 ST、科创板等），批量下载日线行情与指数数据。

```bash
# 默认模式：跳过已存在的数据（断点续传）
python scripts/download_data.py

# 强制模式：覆盖重新下载所有数据
python scripts/download_data.py --force
```

**数据源**：目前默认使用 Baostock (可配置)。
**输出**：数据将保存至 `data/raw/{symbol}.parquet`。
**过滤**：下载前会自动应用 `config/main.yaml` 中 `stock_pool` 的过滤规则（如是否包含创业板、剔除 ST 等）。


#### 2.3 （可选）全自动挂机下载

如果需要全量下载历史数据，且担心网络不稳定或接口风控，可以使用挂机脚本：

```bash
python scripts/auto_run.py
```

**功能**：自动运行下载任务。
**机制**：若遇到网络错误或反爬限制，脚本会自动休眠 5 分钟（300秒）等待 IP 冷却，然后自动重启继续下载，直到所有任务完成。

#### 2.4 数据清洗与质检 (Data Cleaning) [重要]

原始行情数据（Raw Data）可能包含重复行、价格异常（如 0 元）或隐性缺失。本步骤将执行标准化清洗，并生成质量报告。

```bash
python scripts/clean_and_check.py
```
**输入**：`data/raw/ (原始数据) + data/meta/trade_calendar.parquet` (本地交易日历)。

**配置**：读取 `config/main.yaml` 中的 `preprocessing.quality` 参数。

**处理逻辑**：

 1. 行级清洗：去重、剔除价格为 0 或 NaN 的异常行。

 2. 标的级筛选（自动剔除）：

- 高停牌率：剔除历史停牌时间占比 > 10% 的股票（阈值可配）。

- 低流动性：剔除日均换手率 < 1% 的“僵尸股”（阈值可配）。

- 严重缺失：剔除数据缺失率过高的标的。

**输出**：

1. 清洗后数据：保存至 `data/raw_cleaned/{symbol}.parquet`（仅包含状态为 OK 的股票）。
2. 质量报告：生成 `data/raw_cleaned/data_quality_report.csv`。请务必查看此报告，确认哪些股票被标记为 `REJECT` 及其原因（如 `HIGH_SUSPENSION` 或 `LOW_LIQUIDITY`）。


#### 2.5 数据探索性分析 (EDA) [新增]

在清洗完成后，强烈建议运行 EDA 模块，分析数据的分布特征、停牌情况和动量效应，以便为特征工程提供依据。

```bash
# 默认模式：随机采样 200 只股票进行分析
python scripts/run_eda.py

# 自定义模式：采样 500 只股票
python scripts/run_eda.py --sample 500
```

**输出产物：可视化图表 (figures/)：**

- `check_alignment.png`: 检查个股与指数走势是否对齐（验证数据源质量）。

- `dist_returns.png`: 日收益率分布图（辅助确定 Label 阈值）。

- `dist_suspension.png`: 个股停牌率分布（识别垃圾股）。

- `dist_autocorr.png`: 收益率自相关性分布（判断动量/反转特征）。

- `dist_liquidity.png`: 流动性/换手率分布。

**数据报告 (reports/eda/)：**

- `otential_anomalies.csv`: 疑似未复权或异常波动的股票列表。

- `return_distribution_stats.csv`: 收益率分位数统计。

### 💡 总结：接下来的操作

你现在可以按照这个逻辑顺序执行一次全流程，验证整个数据链路是否通畅：

1.  `python scripts/init_stock_pool.py` (获取列表+日历)
2.  `python scripts/download_data.py` (下载原始数据)
3.  `python scripts/clean_and_check.py` (清洗数据 -> 生成 raw_cleaned)
4.  `python scripts/rebuild_features.py` (读取 raw_cleaned -> 生成 features)

这样你的量化系统就拥有了一个坚实、干净的数据基础。

---
# 3. 构建特征（features）与标签（labels）

执行完整的特征工程流水线，将清洗后的行情数据 (`data/raw_cleaned`) 转化为可用于机器学习训练的特征矩阵 (`data/processed`)。

**运行命令**：
```bash
python scripts/rebuild_features.py
```
**核心逻辑与配置**： 流水线行为由 `config/main.yaml` 中的 `preprocessing` 模块控制，主要包含以下三个阶段：

**3.1 特征工程 (Feature Generation)**

3.1 特征工程 (Feature Generation)
支持各类经典技术指标的批量计算与归一化处理：

* **趋势类**：MA (均线偏离度), MACD (差离值), Bollinger Bands (布林带位置/带宽)。

* **动量/反转**：RSI (相对强弱), KDJ (随机指标), Amplitude (振幅)。

* **量能**：Volume Ratio (量比), Turnover (换手率, log变换)。

* **配置项**：可在 `config/main.yaml` 中开关特定因子或调整窗口参数（如 `ma_windows`, `rsi_window`）。

**3.2 标签生成 (Label Generation) - v2.0 升级**

构建了更贴近实战的训练目标（Label）：

* **时序严谨**：基于 **T日信号 -> T+1日均价买入 -> T+1+N日均价卖出** 的逻辑计算收益，避免“用了未来数据”或“收盘无法成交”的偏差。

* **VWAP 计价**：优先使用 **成交均价 (VWAP)** 代替收盘价计算收益率，减少尾盘脉冲对标签的噪声干扰。

* **超额收益 (Excess Return)**：自动扣除基准指数（如沪深300）同期收益，训练模型寻找跑赢大盘的 Alpha。

* **一字板过滤**：自动剔除 **T+1日一字涨停** 的样本（Label设为 NaN），防止模型学习到“买不进去的虚假暴利”。

* **多周期支持**：根据 `horizon` 参数生成不同持有期（如 5日、10日）的标签。

**3.3 数据合并 (Batch Processing)**

* **单文件输出**：每只股票的特征存放在 `data/processed/{symbol}.parquet`。

* **全量合并**：脚本最后会自动将全市场所有股票数据合并为一张大表 `data/processed/all_stocks.parquet`，这是后续模型训练的直接输入。

### 更新说明：

1.  **明确数据源**：指出了输入源改为清洗后的 `data/raw_cleaned`，而非原始的 `data/raw`。
2.  **细化特征描述**：根据 `features.py` 的代码，列举了具体的指标类型（MA, MACD, RSI, KDJ, BOLL 等）。
3.  **强调 Label 逻辑**：这是 v2.0 的核心改进，根据 `labels.py` 补充了 VWAP、T+1 时序、超额收益和一字板过滤的说明，这些是量化实战中非常关键的细节。
4.  **产出物说明**：明确了最终产出是 `all_stocks.parquet`，方便后续步骤引用。

--- 
# 4. 数据分析与检查 (Analysis & Checks)

为确保数据质量与因子有效性，本项目内置了完整的分析模块 `src/analysis`，并提供了三份配套的分析报告指南（Docs）。

### 4.1 探索性数据分析 (EDA)
在特征工程之前，对清洗后的数据进行全维度的质量扫描，确保没有混入“脏数据”。

- **执行脚本**：

```bash
python scripts/run_eda.py
```

* **核心检查点：**

 * **数据完整性**：检查个股停牌率分布，识别长期停牌的垃圾股。

 * **流动性分布**：检查成交额/换手率分布，识别流动性枯竭的“僵尸股”。

 * **数据对齐**：随机抽样个股与大盘走势对比，验证复权处理是否正确。

 * **市场特征**：统计收益率分布（尖峰肥尾）与自相关性。

* 📘 **详细指南**：请阅读 `docs/EDA_GUIDE.md` 以获取图表解读说明。

### 4.2 因子有效性分析 (Factor Analysis)
在特征工程之后，评估计算出的因子（Features）对标签（Labels）的预测能力，并进行安全性检查。

* **执行脚本**：

```Bash
python scripts/check_features.py
```

* **核心检查点**：

 * **IC 分析 (Information Coefficient)**：计算每个特征与未来收益的相关系数 (RankIC)，判断因子是否有效。

 * **未来函数检测**：如果某因子 IC > 0.8，脚本会发出警告，提示可能存在未来数据泄露。

 * **多重共线性**：生成相关性热力图，辅助剔除高度冗余的特征。

 * **标签分布**：检查训练目标（Label）是否符合正态分布，是否需要对数变换。

* 📘 **详细指南**：请阅读 `docs/FACTOR_ANALYSIS_REPORT.md`。

### 4.3 时间窗口敏感性分析 (Horizon Analysis)

回答“模型预测几天后的收益最准？”这一核心问题，指导 `config/main.yaml` 中 `horizon` 参数的设置。

* **执行脚本**：

```Bash
python scripts/check_time_horizon.py
```

* 核心检查点：

 * **IC 衰减曲线 (Decay Curve)**：绘制 Top 因子在 1d, 3d, 5d, 10d, 20d 不同预测周期下的表现。

 * **最佳周期判定**：

  * **快速衰减型**：适合做超短线（High Frequency）。

  * **发散增强型**：适合做波段趋势（Trend Following）。

* 📘 **详细指南**：请阅读 `docs/HORIZON_ANALYSIS_REPORT.md`。

📊 **产出物位置**： 所有分析生成的**可视化图表**均保存在 `figures/` 目录下（按时间戳归档）,**统计数据表**保存在 `reports/` 目录下。


### 4.4 策略信号诊断 (Signal Diagnosis) [新增]

用于评估策略信号的**稳健性、成本敏感性**和**风险暴露**，对策略进行深度剖析

**执行脚本：**

```bash
python scripts/signal_diagnosis.py
```

**核心检查点：**
 * **风险暴露**：分析选股组合的**中位价格、换手率、历史波动率**及**动量**分布，识别是否存在潜在的风格风险.

 * **成本敏感性**：测试策略在不同交易成本（如 1‰, 2‰, 5‰）下的**夏普比率**衰减，评估策略的实战盈利空间.

 * **危机压力测试**：在历史极端熊市（如 2018 年、2022 年、2024 年微盘股危机）时间窗口下运行回测，评估策略的**最大回撤**与生存能力.

 * **信号质量**：分析**月度 IC 稳定性** (IC IR) 和**平均换仓率**，评估信号的长期有效性与稳定性.

📊 **产出物位置**： 所有分析生成的**可视化图表**均保存在 `figures/` 目录下（按时间戳归档）,**统计数据表**保存在 `reports/` 目录下.

### 内容来源说明：
1.  **EDA 部分**：依据 `scripts/run_eda.py` 和 `docs/EDA_GUIDE.md`，涵盖了停牌率、流动性、对齐检查等功能。
2.  **Factor Analysis 部分**：依据 `scripts/check_features.py` 和 `docs/FACTOR_ANALYSIS_REPORT.md`，涵盖了 IC 分析、未来函数检测和共线性检查。
3.  **Horizon Analysis 部分**：依据 `scripts/check_time_horizon.py` 和 `HORIZON_ANALYSIS_REPORT.md`

---
# 5. 训练模型 (Model Training)

执行以下命令，基于生成的特征矩阵训练预测模型：

```bash
python scripts/train_model.py
```
**核心逻辑**：

 1. **读取数据**：自动加载 `data/processed/all_stocks.parquet` 全量特征数据。

 2. **时序切分**：根据 `config/main.yaml` 中的 `train_val_split_date`（如 2019-01-01）将数据划分为训练集和验证集，严防未来信息泄露。

 3. **模型训练**：使用配置中的超参数训练 **XGBoost** 模型。

 4. **生成预测**：对全量数据（训练集+验证集）生成预测分数，方便后续分析历史拟合情况。

 5. **版本归档**：自动生成带时间戳的版本目录（如 `20231201_100000`），保存模型产物。

**配置说明 (`config/main.yaml`)**：

- `model.label_col`: 目标标签列名（默认 `label`，对应 feature engineering 阶段生成的 `label_5d` 等）。

- `model.params`: XGBoost 核心参数（`learning_rate`, `max_depth`, `n_estimators` 等）。

- `model.train_val_split_date`: 训练/验证集切分日期。

**输出产物 (`data/models/{version}/`)**：

- `model.json`: 训练好的模型文件（可用于后续加载和推理）。

- `predictions.parquet`: 包含 `date`, `symbol`, `close`, `label`, `pred_score` 的预测结果表，是下一步回测的直接输入。

---


# 6. 回测 (Backtesting) - 策略验证

本项目内置了基于**向量化（Vectorized）的高效回测引擎，支持模拟真实的分仓轮动**交易模式。

执行命令：
```bash
python scripts/run_backtest.py
```

该脚本会自动寻找 `data/models/` 下最新的模型版本，加载其生成的预测分数 (`predictions.parquet`)，并执行以下核心流程：

**6.1 核心机制**:

1. **策略信号生成** (`TopKSignalStrategy`)

- **风控过滤**：在生成信号前，自动应用严格的过滤规则（剔除 ST/退市、流动性枯竭、低价股、科创板/创业板等），配置与 `config/main.yaml` 中的 filter 模块一致。

- **Top-K 选股**：根据模型预测分数（Score），每日选取排名前 K 的股票（默认 `top_k=5`）。

2. **分仓轮动机制 (Signal Expansion)**

- **滚动持仓**：如果 `horizon=5`（持有 5 天），回测引擎不会在第 5 天一次性全卖，而是采用每日换仓 1/N 的模式。

- **逻辑**：T 日产生的信号，会在未来 T+1 至 T+5 日内持续生效。资金被平分为 5 份，每天仅对其中的 1/5 进行调仓（卖出 5 天前买入的，买入当天新入选的）。

- **优势**：这种模式能平滑资金曲线，减少单一日期市场波动的影响，更贴近量化基金的实盘操作。

3. **严谨的收益计算**

- **次日成交 (Next-Day Execution)**：严格遵守 **T 日预测 -> T+1 日开盘/均价成交** 的时序，使用 T+1 日的涨跌幅 (`next_pct_chg`) 计算收益，杜绝“未来函数”。

- **成本估算**：内置交易成本扣除逻辑（默认双边滑点+佣金+印花税 ≈ 千分之 1.5），在分仓模式下按每日实际换手率计算扣费。

**6.2 回测产物**
回测结果将保存在 `data/models/{version}/backtest_result/` 目录下：

- `equity_curve.png`：资金曲线图。

 - **上图 (Linear)**：展示策略净值 vs 基准指数（沪深300）的线性走势。

 - **下图 (Log)**：对数坐标图，用于观察早期的超额收益情况。

- `backtest_equity.csv`：每日净值数据（策略净值、基准净值），可用于进一步分析。

- *控制台输出*：包含 年化收益率 (Annual Return)、夏普比率 (Sharpe Ratio)、最大回撤 (Max Drawdown) 等关键指标。

---

# 📌 未来计划 (Roadmap)

我们计划在后续版本中逐步引入以下高级特性，进一步提升系统的实战能力：

* **P1 (近期): 可视化增强**
    * 增加回测结果的交互式图表 (HTML 报告)。
    * 提供特征重要性 (Feature Importance) 与 SHAP 值分析，提升模型可解释性。
* **P2 (中期): 进阶回测体系**
    * 引入 **Walk-Forward Analysis (滚动回测)**，评估策略在时间推移下的稳定性。
    * 引入 **Monte Carlo (蒙特卡洛模拟)**，评估策略在极端市场环境下的风险边界。
* **P3 (远期): 深度学习与实盘**
    * 接入 LSTM / Transformer 等时序深度学习模型。
    * 开发 **Live Engine**，对接实盘/模拟盘 API，实现信号自动下单。
    * 构建 Web Dashboard，实现从命令行工具到图形化平台的升级。

---
# 操作流程图


---

# 📄 License

MIT License

# ✨ 作者

enhen-x
专注于 A 股量化、ML、工程化研究。欢迎 Star 与交流。



