# Quant_A_Share_V2.0


# 🎯 项目目标 (Project Goals)

**Quant_A_Share_V2.0** 致力于为个人投资者打造一个**基于机器学习的 A 股量化决策系统**。本项目不追求复杂的黑盒魔法，而是强调工程化、透明化和可验证性，旨在解决散户在量化投资中面临的“数据乱、回测假、落地难”三大痛点。

项目回测曲线图如下：

<p align="center">
  <h3 align="center">资金曲线图</h3>
  <img src="architecture\test_result\backtest_result\equity_curve.png" width="600" alt="资金曲线图">
 </p>
 
核心目标包括：

1.  **构建工业级数据流水线**：实现从数据下载、清洗、质检到特征工程的全自动化闭环，确保输入模型的每一条数据都干净、可靠。
2.  **机器学习驱动选股**：摒弃传统的线性因子叠加，利用 XGBoost 等机器学习模型挖掘非线性市场规律，支持**三相屏障 (Triple Barrier)** 等机构级标签构建方法。
3.  **严谨的策略验证体系**：提供**探索性数据分析 (EDA)**、**因子有效性检测 (IC)** 和**防未来函数回测**等多维度的评估工具，拒绝“过拟合”的虚假回测。
4.  **实战导向的推荐能力**：不仅能跑回测，更能每日生成可执行的**精选股票推荐列表**，打通从研究到交易的“最后一公里”。


项目流程图如下:

<p align="center">
  <h3 align="center">项目整体流程图</h3>
  <img src="architecture\data_clean\QUANT_A_SHARE_V2.png" width="800" alt="项目流程图">
</p>


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
* **特征中性化**：支持市值中性化 (Market Cap Neutralization)，通过横截面回归剔除风格因子影响，获取纯净 Alpha。
* **高级标签生成**：支持 **三相屏障法 (Triple Barrier Method)**，结合 **VWAP (成交均价)** 计算真实收益，并自动剔除一字涨停样本。
* **双头模型 (Dual-Head Model) 🆕**：同时训练 **回归头** (预测收益率) 和 **分类头** (预测涨跌方向)，通过加权融合获得更稳健的预测信号。
* **模型训练**：支持 **XGBoost** 和 **LightGBM** 双引擎，自动划分训练/验证集、模型保存与版本管理。

### ✅ 4. 策略与回测 (Strategy & Backtest)
* **Top-K 选股策略**：基于模型预测分，结合价格、市值、板块等风控规则进行每日精选。
* **向量化回测引擎**：实现了支持**分仓轮动**的高效回测框架，严格执行 T+1 交易逻辑，内置交易成本扣除，输出资金曲线与夏普比率。
* **每日推荐系统**：提供一键式脚本，基于最新行情生成**每日潜力股名单 (Daily Picks)**。
* **蒙特卡洛模拟 🆕**：通过 Bootstrap 重采样、权重扰动、噪音注入等方法评估策略稳健性和置信区间。  

---

## 📌 项目结构


```text
Quant_A_SHARE_V2.0/
├── config/
│   └── main.yaml               # [核心] 全局配置文件
│                               # 包含：路径、数据源、特征中性化、风控参数(市值/价格)、模型参数(滚动训练配置)、
│                               # 策略参数(Top-K/动态仓位/择时)、回测配置(成本/轮动模式)。
│
├── docs/                       # [文档] 分析指南
│   ├── EDA_GUIDE.md            # 探索性数据分析 (EDA) 结果解读指南
│   ├── FACTOR_ANALYSIS_REPORT.md # 因子有效性与 IC 分析解读指南
│   ├── HORIZON_ANALYSIS_REPORT.md # 预测周期 (Horizon) 敏感性分析解读指南
│   └── XUEQIU_GUIDE.md         # 雪球实盘交易指南
│
├── scripts/                    # [入口] 命令行执行脚本
│   ├── analisis/               # [分析] 数据清洗与分析诊断
│   │   ├── clean_and_check.py      # [Step 2] 数据清洗与质检（去重、停牌过滤、僵尸股剔除）
│   │   ├── run_eda.py              # [Analysis] 运行全维度探索性分析 (EDA)
│   │   ├── check_features.py       # [Analysis] 因子有效性检查 (IC 分析/多重共线性/未来函数检测)
│   │   ├── check_stress_test.py    # [Test] 策略压力测试 (成本敏感性/历史熊市生存测试)
│   │   ├── check_time_horizon.py   # [Analysis] 最佳持仓周期分析 (IC Decay)
│   │   ├── check_monte_carlo.py    # [New] 蒙特卡洛模拟分析 (置信区间/稳健性评估)
│   │   ├── check_overfit.py        # [New] 过拟合检测 (噪音敏感性测试)
│   │   ├── analyze_threshold.py    # [New] 分类阈值优化分析 (双头模型)
│   │   ├── explain_model.py        # [Analysis] 模型可解释性分析 (SHAP)
│   │   ├── feature_selector.py     # [Tools] 特征筛选工具
│   │   └── signal_diagnosis.py     # [Diagnosis] 策略信号诊断
│   │
│   ├── date_landing/           # [数据] 数据下载与更新
│   │   ├── init_stock_pool.py      # [Step 0] 初始化股票池与交易日历
│   │   ├── download_data.py        # [Step 1] 批量下载/断点续传行情数据
│   │   ├── auto_run.py             # [Step 1+] 全自动挂机下载（含防封控冷却重启逻辑）
│   │   └── update_data.py          # [Daily] 增量更新数据（日历/指数/个股）
│   │
│   ├── feature_create/         # [特征] 特征工程
│   │   └── rebuild_features.py     # [Step 3] 特征工程流水线 (Features + Labels + Filtering)
│   │
│   ├── model_train/            # [模型] 模型训练
│   │   ├── run_walkforward.py      # [Step 4] 滚动训练 (Walk-Forward) - 推荐
│   │   └── train_model.py          # [Step 4] 单次模型训练 (Legacy)
│   │
│   ├── back_test/              # [回测] 回测与推荐
│   │   ├── run_backtest.py         # [Step 5] 策略回测 (分仓轮动/动态成本)
│   │   └── run_recommendation.py   # [App] 每日推荐 (Daily Picks)
│   │
│   ├── live/                   # [实盘] 自动交易
│   │   ├── run_auto_trading.py     # [Live] 雪球实盘/模拟自动交易脚本
│   │   └── run_weekly_rebalance.py # [New] 周度全仓换仓脚本
│   │
│   └── tools/                  # [工具] 辅助脚本
│       └── start_tensorboard.py    # 启动 TensorBoard 训练监控
│
├── src/                        # [源码] 核心逻辑库
│   ├── analysis/               # 分析引擎模块
│   │   ├── eda_engine.py       # EDA 绘图与统计核心
│   │   ├── factor_checker.py   # 因子质量分析器 (IC/RankIC/分布)
│   │   ├── horizon_analyzer.py # 多周期 IC 衰减分析器
│   │   └── model_interpreter.py # [New] 模型解释器 (SHAP)
│   │
│   ├── backtest/               # 回测模块
│   │   └── backtester.py       # 向量化回测引擎
│   │
│   ├── data_source/            # 数据源适配层 (Facade模式)
│   │
│   ├── model/                  # 机器学习模块
│   │   ├── trainer.py          # 训练流程管理器 (支持双头模型)
│   │   ├── xgb_model.py        # XGBoost 模型封装
│   │   ├── lgb_model.py        # [New] LightGBM 模型封装 (双头模型专用)
│   │   └── training_monitor.py # TensorBoard 训练监控模块
│   │
│   ├── preprocessing/          # 预处理模块
│   │   ├── pipeline.py         # 特征工程总流水线
│   │   ├── features.py         # 特征计算工厂
│   │   ├── labels.py           # 标签生成工厂
│   │   └── neutralization.py   # [New] 特征中性化 (市值中性化)
│   │
│   ├── strategy/               # 策略模块
│   │   └── signal.py           # 信号生成器
│   │
│   ├── live/                   # [New] 实盘交易模块
│   │   ├── config.py           # 实盘配置管理
│   │   ├── trade_recorder.py   # 交易记录与报表生成
│   │   ├── trading_scheduler.py # 调度与资金管理
│   │   └── xueqiu_broker.py    # 雪球接口封装
│   │
│   └── utils/                  # 通用工具库
│       ├── config.py           # 全局配置加载器
│       ├── logger.py           # 日志管理
│       └── io.py               # 文件读写封装
│
├── data/ (自动生成目录)
│   ├── raw/                    # 原始行情数据
│   ├── raw_cleaned/            # 清洗后的标准行情
│   ├── processed/              # 最终特征矩阵
│   ├── meta/                   # 股票列表与交易日历
│   ├── models/                 # 模型仓库
│   │   ├── WF_YYYYMMDD.../     # 滚动训练生成的年度模型集
│   │   └── YYYYMMDD.../        # 单次训练的模型存档
│   ├── index/                  # 指数数据
│   └── live_trading/           # [New] 实盘交易配置与记录
│
├── figures/ (自动生成)          # 分析图表 (.png)
├── reports/ (自动生成)          # 分析报告 & 每日推荐 (.csv)
├── logs/                       # 运行日志
│   └── tensorboard/            # TensorBoard 训练监控日志
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


#### 2.5 数据探索性分析 (EDA) 

在清洗完成后，强烈建议运行 EDA 模块，分析数据的分布特征、停牌情况和动量效应，以便为特征工程提供依据。

```bash
# 默认模式：随机采样 200 只股票进行分析
python scripts/run_eda.py

# 自定义模式：采样 500 只股票
python scripts/run_eda.py --sample 500
```

**输出产物：可视化图表 (figures/)：**

- `check_alignment.png`: 检查个股与指数走势是否对齐（验证数据源质量）。

<p align="center">
  <h3 align="center">检查数据是否对齐</h3>
  <img src="architecture\test_result\eda\20251207_175729\check_alignment.png" width="600" alt="检查数据是否对齐">
</p>


- `dist_returns.png`: 日收益率分布图（辅助确定 Label 阈值）。

<p align="center">
  <h3 align="center">收益率自相关性分布</h3>
  <img src="architecture\test_result\eda\20251207_175729\dist_returns.png" width="600" alt="收益率自相关性分布">
</p>

- `dist_suspension.png`: 个股停牌率分布（识别垃圾股）。

<p align="center">
  <h3 align="center">个股停牌率分布</h3>
  <img src="architecture\test_result\eda\20251207_175729\dist_suspension.png" width="600" alt="个股停牌率分布">
</p>

- `dist_autocorr.png`: 收益率自相关性分布（判断动量/反转特征）。

<p align="center">
  <h3 align="center">日收益率分布图</h3>
  <img src="architecture\test_result\eda\20251207_175729\dist_autocorr.png" width="600" alt="日收益率分布图">
</p>

- `dist_liquidity.png`: 流动性/换手率分布。

<p align="center">
  <h3 align="center">日收益率分布图</h3>
  <img src="architecture\test_result\eda\20251207_175729\dist_liquidity.png" width="600" alt="日收益率分布图">
</p>

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

**3.4 特征中性化 (Feature Neutralization) - 🆕**

为了构建更稳健的策略，避免模型仅利用“小市值效应”等风格因子获利，V2.0 引入了**特征中性化**模块：

* **核心逻辑**：对每一天（Cross-Section）的数据，以特征为目标变量 $Y$，风险因子（如对数流通市值 `feat_mcap_log`）为自变量 $X$ 进行线性回归，取残差 $E = Y - (\beta X + \alpha)$ 作为去风格后的新特征。
* **灵活配置**：
    * 支持在 `config/main.yaml` 中配置 `risk_factors`（默认剔除市值影响）。
    * 可选择对全部特征或指定特征列表进行中性化。
* **实现代码**：参见 `src/preprocessing/neutralization.py`。

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

* **核心检查点(与原始数据eda分析保持一致)：**

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

 <p align="center">
  <h3 align="center">IC 分析 (Information Coefficient)</h3>
  <img src="architecture\test_result\factors\20251208_012005\feature_ic_top30.png" width="600" alt="IC 分析 (Information Coefficient)">
 </p>

 * **未来函数检测**：如果某因子 IC > 0.8，脚本会发出警告，提示可能存在未来数据泄露。

 * **多重共线性**：生成相关性热力图，辅助剔除高度冗余的特征。

 <p align="center">
  <h3 align="center">多重共线性分析</h3>
  <img src="architecture\test_result\factors\20251208_012005\feature_correlation.png" width="600" alt="多重共线性分析">
 </p>

 * **标签分布**：检查训练目标（Label）是否符合正态分布，是否需要对数变换。

  <p align="center">
  <h3 align="center">标签分布分析</h3>
  <img src="architecture\test_result\factors\20251208_012005\dist_label.png" width="600" alt="标签分布分析">
 </p>

* 📘 **详细指南**：请阅读 `docs/FACTOR_ANALYSIS_REPORT.md`。

### 4.3 时间窗口敏感性分析 (Horizon Analysis)

回答“模型预测几天后的收益最准？”这一核心问题，指导 `config/main.yaml` 中 `horizon` 参数的设置。

* **执行脚本**：

```Bash
python scripts/check_time_horizon.py
```

* 核心检查点：

 * **IC 衰减曲线 (Decay Curve)**：绘制 Top 因子在 1d, 3d, 5d, 10d, 20d 不同预测周期下的表现。

 <p align="center">
  <h3 align="center">IC 衰减曲线 (Decay Curve)</h3>
  <img src="architecture\test_result\horizon_analysis\20251208_012101\ic_decay_curve.png" width="600" alt="IC 衰减曲线 (Decay Curve)">
 </p>

 * **最佳周期判定**：

  * **快速衰减型**：适合做超短线（High Frequency）。

  * **发散增强型**：适合做波段趋势（Trend Following）。

* 📘 **详细指南**：请阅读 `docs/HORIZON_ANALYSIS_REPORT.md`。

📊 **产出物位置**： 所有分析生成的**可视化图表**均保存在 `figures/` 目录下（按时间戳归档）,**统计数据表**保存在 `reports/` 目录下。


### 4.4 策略信号诊断 (Signal Diagnosis)

用于评估策略信号的**稳健性、成本敏感性**和**风险暴露**，对策略进行深度剖析

**执行脚本：**

```bash
python scripts/signal_diagnosis.py
```

**核心检查点：**
 * **风险暴露**：分析选股组合的**中位价格、换手率、历史波动率**及**动量**分布，识别是否存在潜在的风格风险.

 <p align="center">
  <h3 align="center">中位价格</h3>
  <img src="architecture\test_result\signals\20251209_193138\price_distribution.png" width="600" alt="中位价格">
 </p>

 <p align="center">
  <h3 align="center">换仓率</h3>
  <img src="architecture\test_result\signals\20251209_193138\turnover_rate.png" width="600" alt="换手率">
 </p>

 <p align="center">
  <h3 align="center">历史波动率</h3>
  <img src="architecture\test_result\signals\20251209_193138\volatility_distribution.png" width="600" alt="历史波动率">
 </p>

 <p align="center">
  <h3 align="center">动量</h3>
  <img src="architecture\test_result\signals\20251209_193138\momentum_distribution.png" width="600" alt="动量">
 </p>

 * **成本敏感性**：测试策略在不同交易成本（如 1‰, 2‰, 5‰）下的**夏普比率**衰减，评估策略的实战盈利空间.

 <p align="center">
  <h3 align="center">不同交易成本下收益</h3>
  <img src="architecture\test_result\stress_test\cost_sensitivity_comparison.png" width="600" alt="不同交易成本下收益">
 </p>

 * **危机压力测试**：在历史极端熊市（如 **2020年疫情**、**2021年核心资产破裂**、**2022年加息熊市**、**2023-24年微盘股/流动性危机**）时间窗口下运行回测，评估策略的**最大回撤**与生存能力.

 <p align="center">
  <h3 align="center">2020年疫情爆发 (V型反转)</h3>
  <img src="architecture\test_result\stress_test\crisis_2020_Covid-19\equity_curve.png" width="600" alt="2020年疫情">
 </p>

 <p align="center">
  <h3 align="center">2021年核心资产泡沫破裂</h3>
  <img src="architecture\test_result\stress_test\crisis_2021_Bubble_Burst\equity_curve.png" width="600" alt="2021年泡沫破裂">
 </p>

 <p align="center">
  <h3 align="center">2022年美联储加息熊市</h3>
  <img src="architecture\test_result\stress_test\crisis_2022_Fed_Hike\equity_curve.png" width="600" alt="2022年加息熊市">
 </p>

 <p align="center">
  <h3 align="center">2023-2024年漫长阴跌与微盘股危机</h3>
  <img src="architecture\test_result\stress_test\crisis_2023-2024_Bear\equity_curve.png" width="600" alt="2023-24年阴跌">
 </p>

 * **信号质量**：分析**月度 IC 稳定性** (IC IR) 和**平均换仓率**，评估信号的长期有效性与稳定性.

 <p align="center">
  <h3 align="center">月度 IC 稳定性</h3>
  <img src="architecture\test_result\signals\20251209_193138\ic_by_month.png" width="600" alt="月度 IC 稳定性">
 </p>

 <p align="center">
  <h3 align="center">换手率</h3>
  <img src="architecture\test_result\signals\20251209_193138\turnover_distribution.png" width="600" alt="换手率">
 </p>

📊 **产出物位置**： 所有分析生成的**可视化图表**均保存在 `figures/` 目录下（按时间戳归档）,**统计数据表**保存在 `reports/` 目录下.

### 内容来源说明：
1.  **EDA 部分**：依据 `scripts/run_eda.py` 和 `docs/EDA_GUIDE.md`，涵盖了停牌率、流动性、对齐检查等功能。
2.  **Factor Analysis 部分**：依据 `scripts/check_features.py` 和 `docs/FACTOR_ANALYSIS_REPORT.md`，涵盖了 IC 分析、未来函数检测和共线性检查。
3.  **Horizon Analysis 部分**：依据 `scripts/check_time_horizon.py` 和 `HORIZON_ANALYSIS_REPORT.md`

---
# 5. 训练模型 (Model Training)

本项目支持两种训练模式：**单次训练**（快速测试）和 **滚动训练 (Walk-Forward)**（推荐生产使用）。

**配置说明 (`config/main.yaml`)**：

| 参数 | 说明 |
|------|------|
| `model.label_col` | 目标标签列名 |
| `model.params` | XGBoost 超参数 |
| `model.dual_head.enable` | 是否启用双头模型 🆕 |
| `model.dual_head.regression.weight` | 回归头融合权重 (默认 0.6) |
| `model.dual_head.classification.weight` | 分类头融合权重 (默认 0.4) |
| `model.early_stopping_rounds` | 早停耐心 |
| `model.enable_tensorboard` | 是否启用监控 |
| `model.use_feature_selection` | 是否启用特征筛选 |

**输出产物 (`data/models/{version}/`)**：

- `model.json` / `model_YYYY.json`: 模型文件
- `predictions.parquet`: 预测结果表

### 5.1 滚动训练 (推荐)

模拟真实时间流逝，每年重新训练模型，生成无未来函数污染的预测集：

```bash
python scripts/model_train/run_walkforward.py
```

**核心机制**：
- 按年滚动：例如用 2010-2018 年数据训练，预测 2019 年；再用 2010-2019 年数据训练，预测 2020 年...
- 支持 **扩张窗口** (Expanding) 或 **滚动窗口** (Rolling) 两种模式
- 每年自动保存独立模型，最终合并为全量预测表
- **双头模型支持 🆕**：启用后同时训练回归模型 (`model_reg_YYYY.joblib`) 和分类模型 (`model_cls_YYYY.joblib`)，预测结果自动融合

### 5.2 单次训练 (快速测试)

```bash
python scripts/model_train/train_model.py
```

### 5.3 训练监控 (TensorBoard) 🆕

训练过程自动记录到 TensorBoard，支持可视化监控：

```bash
# 启动 TensorBoard
python scripts/tools/start_tensorboard.py
```

访问 http://localhost:6006 可查看：
- 📈 **训练/验证损失曲线** - 判断是否过拟合
- 🎯 **特征重要性排名** - 了解模型关注点
- ⚙️ **超参数配置** - 方便对比实验

### 5.4 特征筛选

自动分析并筛选有效特征，移除低 IC 和冗余特征：

```bash
python scripts/analisis/feature_selector.py
```

启用方法：在 `config/main.yaml` 中设置 `use_feature_selection: true`



### 5.5 模型可解释性 (SHAP) 🆕

使用 SHAP (SHapley Additive exPlanations) 深入分析模型决策逻辑，了解哪些因子最重要，以及它们如何影响预测结果：

```bash
# 自动检测最新模型并分析
python scripts/analisis/explain_model.py

# 指定模型版本
python scripts/analisis/explain_model.py --version 20251214_175849
```

**输出图表 (`figures/interpretation/{version}/`)**：
- **Summary Plot**: 特征重要性总览，展示因子数值大小与 SHAP 值的正负关系。
- **Dependence Plot**: 单个特征的依赖图，展示特征值与预测贡献的非线性关系。

### 5.6 双头模型 (Dual-Head Model) 🆕

双头模型同时训练两个预测目标，融合后获得更稳健的选股信号：

| 模型头 | 任务类型 | 预测目标 | 优势 |
|--------|----------|----------|------|
| **回归头** | Regression | 预测未来收益率 | 捕捉收益弹性 |
| **分类头** | Classification | 预测涨跌方向 (0/1) | 提高方向准确率 |

**融合公式**：
```
pred_score = α × normalize(pred_reg) + β × normalize(pred_cls)
```

**配置示例 (`config/main.yaml`)**：
```yaml
model:
  dual_head:
    enable: true
    model_type: "lightgbm"
    regression:
      weight: 0.6
    classification:
      weight: 0.4
      threshold: 0.0
    fusion:
      method: "weighted_average"
      normalize: true
```

### 5.7 过拟合检测 🆕

通过噪音注入测试评估模型是否存在过拟合：

```bash
python scripts/analisis/check_overfit.py
```

**检测逻辑**：
- 向预测分数添加 5%、10%、20% 噪音
- 对比噪音前后的收益衰减
- **健康模型**：收益平缓下降
- **过拟合模型**：收益急剧崩溃

### 5.8 蒙特卡洛模拟分析 🆕

通过多种随机模拟方法评估策略的稳健性和置信区间：

```bash
python scripts/analisis/check_monte_carlo.py
```

**模拟方法**：

| 方法 | 描述 | 用途 |
|------|------|------|
| Bootstrap 重采样 | 有放回抽样历史信号 | 评估收益置信区间 |
| 权重扰动 | 随机扰动 α/β 融合权重 | 评估权重敏感性 |
| 噪音注入 | 向预测分添加噪音 | 评估模型抗干扰力 |
| 时间窗口采样 | 随机采样时间区间 | 评估时间稳定性 |

**输出报告 (`reports/monte_carlo/`)**：
- `monte_carlo_distribution.png`: 收益/夏普/回撤分布图
- `noise_sensitivity.png`: 噪音敏感性曲线
- `weight_sensitivity.png`: 权重敏感性散点图
- `monte_carlo_report.txt`: 统计汇总与置信区间

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
 
 <p align="center">
  <h3 align="center">资金曲线图</h3>
  <img src="architecture\test_result\backtest_result\equity_curve.png" width="600" alt="资金曲线图">
 </p>

 - **上图 (Linear)**：展示策略净值 vs 基准指数（沪深300）的线性走势。

 - **下图 (Log)**：对数坐标图，用于观察早期的超额收益情况。

- `backtest_equity.csv`：每日净值数据（策略净值、基准净值），可用于进一步分析。

- *控制台输出*：包含 年化收益率 (Annual Return)、夏普比率 (Sharpe Ratio)、最大回撤 (Max Drawdown) 等关键指标。

**6.3 信号诊断分析**

运行信号诊断脚本可深入分析选股特征：

```bash
python scripts/analisis/signal_diagnosis.py
```

**关键指标解读：**

| 指标 | 说明 |
|------|------|
| **动量 (Momentum)** | 选股组合在被选中之前的历史涨跌幅 |
| **波动率 (Volatility)** | 选股组合的日收益标准差 |

**策略风格判断：**

| 动量特征 | 策略风格 | 说明 |
|----------|----------|------|
| 动量为正（>市场） | 动量/趋势追踪 | 追涨，选择上升股 |
| 动量为负（<市场） | 反转/均值回归 | 抄底，选择超跌股 |

**波动率分析：**

- **高波动组合**：收益潜力大，但风险也高；可能偏好小盘股、超跌股
- **低波动组合**：稳健但收益有限；可能偏好蓝筹、稳定股

> **示例解读**：若动量=-15%（市场0%），波动率=4%（市场2.7%），说明模型倾向选择近期暴跌的高波动股票，呈现典型的**反转策略**特征。

---

# 7. 实盘自动交易 (Live Trading) 🆕

本项目已集成 **雪球 (Xueqiu)** 组合交易接口，支持从每日推荐到自动下单的全流程自动化。

### 7.1 环境准备

1. **安装依赖**：
   ```bash
   pip install easytrader schedule
   ```

2. **获取雪球 Cookies**：
   - 登录 [雪球网页版](https://xueqiu.com)。
   - 按 F12 打开开发者工具，刷新页面。
   - 在 Network 面板找到任意请求，复制 `Cookie` 字段的值。

3. **配置文件**：
   创建 `data/live_trading/config.txt`，格式如下：
   ```text
   # 雪球配置
   [xueqiu]
   portfolio_code = ZHxxxxxx  # 你的雪球组合代码
   cookies = 你的雪球Cookies... # 必须包含 xq_a_token

   # 交易配置
   [trading]
   initial_capital = 1000000  # 初始资金
   max_stocks_per_day = 5     # 每日最大买入股票数
   hold_days = 5              # 持仓天数
   daily_budget_fraction = 5 # 每日资金分配比例 (1/5)
   ```

### 7.2 核心功能

*   **智能调度**：自动读取最新的 `Daily Picks` 推荐列表。
*   **资金管理**：基于滚动资金池（Rolling Budget）计算每日可用资金，等权分配。
*   **双重去重**：结合 **雪球真实持仓** 和 **本地交易记录**，防止重复买入。
*   **自动调仓**：调用雪球 `adjust_weight` 接口，一键完成多只股票的买入/调仓。
*   **交易记录**：自动记录买入明细（股数、价格、时间），生成可视化 CSV 报表。

### 7.3 运行方式

**模拟测试 (Dry Run)**：
验证逻辑，不实际下单。
```bash
python scripts/live/run_auto_trading.py
```

**实盘运行 (Live)**：
实际连接账户并执行交易指令。
```bash
python scripts/live/run_auto_trading.py --real
```

---

# 📌 已完成功能 & 未来计划

### ✅ 已完成 (V2.0)

| 功能 | 状态 | 说明 |
|------|------|------|
| Walk-Forward 滚动训练 | ✅ | 按年滚动，无未来函数 |
| TensorBoard 训练监控 | ✅ | 损失曲线、特征重要性可视化 |
| 特征筛选 | ✅ | 基于 IC 和相关性自动筛选 |
| 正则化优化 | ✅ | L1/L2 正则化、参数调优 |
| 压力测试 | ✅ | 成本敏感性、熊市生存测试 |
| 模型可解释性 (SHAP) | ✅ | 特征贡献度分析、依赖图 |
| 实盘自动交易 (Live Trading) | ✅ | 对接雪球组合，全自动下单 |
| **双头模型 (Dual-Head)** 🆕 | ✅ | 回归+分类融合，提升信号稳健性 |
| **蒙特卡洛模拟** 🆕 | ✅ | Bootstrap/权重扰动/噪音注入，评估置信区间 |
| **过拟合检测** 🆕 | ✅ | 噪音敏感性测试，诊断模型健康度 |
| **周度全仓换仓** 🆕 | ✅ | 支持周度调仓模式 |

### 🚀 未来计划

* **P1 (近期): 策略增强与风险评估**
    * 多模型融合 (XGBoost + LightGBM + CatBoost) 提升预测稳定性
    * 增加回测结果的交互式 HTML 报告，提供更丰富的图表交互
    * 动态融合权重调整：根据市场环境自动调节 α/β

* **P2 (中期): 深度学习生态**
    * 接入 LSTM / Transformer / TCN 等时序深度学习模型
    * 探索 Graph Neural Networks (GNN) 挖掘产业链关系

* **P3 (远期): 平台化与智能化**
    * 构建 Web Dashboard 管理平台 (监控、回测、实盘一体化)
    * 引入 LLM (大语言模型) 辅助研报分析与情绪因子挖掘

---

# 📄 License

MIT License

# ✨ 作者

enhen-x
专注于 A 股量化、ML、工程化研究。欢迎 Star 与交流。



