# Quant_A_Share_V2.0

Quant_A_Share_V2.0 是一个面向 **A 股股票量化研究** 的完整工程框架，覆盖数据获取、预处理、特征工程、标签生成、模型训练、预测、策略、回测以及可视化等全过程。

项目目标：

- 构建统一的数据流水线  
- 支持多数据源（Akshare / Baostock / DataHub）  
- 训练跨股票的机器学习模型（如 XGBoost）  
- 自动生成交易信号、构建组合并进行回测  
- 支持多种验证方式（多样化回测、Monte Carlo、Walk-forward）  
- 为未来引入深度学习、RL、实时交易做准备  

---

## 📌 项目结构

```text
## 📌 项目结构

```text
Quant_A_SHARE_V2.0/
├── config/
│   └── main.yaml               # [变更] 核心配置文件（路径、数据源、预处理、模型参数等统一管理）
│
├── docs/                       # [新增] 文档与分析指南
│   ├── EDA_GUIDE.md            # 探索性数据分析说明
│   ├── FACTOR_ANALYSIS_REPORT.md # 因子有效性分析说明
│   └── HORIZON_ANALYSIS_REPORT.md # 时间窗口敏感性分析说明
│
├── scripts/                    # 命令行脚本（入口）
│   ├── init_stock_pool.py      # [Step 1] 初始化股票列表与交易日历
│   ├── download_data.py        # [Step 2] 批量下载/断点续传行情数据
│   ├── auto_run.py             # [Step 2+] 自动挂机下载（含防封控冷却逻辑）
│   ├── update_data.py          # [Daily] 增量更新数据（日历/指数/个股）
│   ├── clean_and_check.py      # [Step 3] 数据清洗与质量质检
│   ├── run_eda.py              # [Analysis] 运行全维度探索性分析 (EDA)
│   ├── rebuild_features.py     # [Step 4] 构建特征工程与标签 (Feature Engineering)
│   ├── check_features.py       # [Analysis] 检查特征有效性 (IC/自相关/共线性)
│   └── check_time_horizon.py   # [Analysis] 检查预测周期 (IC Decay)
│
├── src/                        # 核心源代码库
│   ├── analysis/               # [新增] 分析引擎模块
│   │   ├── eda_engine.py       # EDA 绘图与统计核心
│   │   ├── factor_checker.py   # 因子质量与IC分析器
│   │   └── horizon_analyzer.py # 多周期 IC 衰减分析器
│   │
│   ├── data_source/            # 数据源适配层
│   │   ├── base.py             # 接口基类
│   │   ├── datahub.py          # 数据统一调度入口 (DataHub)
│   │   ├── baostock_source.py  # Baostock 接口实现
│   │   └── akshare_source.py   # AkShare 接口实现
│   │
│   ├── preprocessing/          # 预处理模块
│   │   ├── pipeline.py         # 特征工程总流水线
│   │   ├── features.py         # 特征计算工厂 (MA, MACD, RSI...)
│   │   └── labels.py           # 标签生成工厂 (VWAP, 涨停过滤...)
│   │
│   └── utils/                  # 通用工具库
│       ├── config.py           # 全局配置加载器 (单例模式)
│       ├── logger.py           # 日志管理
│       └── io.py               # 文件读写封装
│
├── data/ (自动生成目录)
│   ├── raw/                    # 原始行情 parquet
│   ├── raw_cleaned/            # 清洗后的标准行情
│   ├── processed/              # 最终特征矩阵 (如 all_stocks.parquet)
│   ├── meta/                   # 股票元数据 & 交易日历
│   └── index/                  # 指数数据
│
├── figures/ (自动生成)          # 存放 EDA/Factor 分析生成的图表 (.png)
├── reports/ (自动生成)          # 存放分析生成的统计数据 (.csv)
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


### 内容来源说明：
1.  **EDA 部分**：依据 `scripts/run_eda.py` 和 `docs/EDA_GUIDE.md`，涵盖了停牌率、流动性、对齐检查等功能。
2.  **Factor Analysis 部分**：依据 `scripts/check_features.py` 和 `docs/FACTOR_ANALYSIS_REPORT.md`，涵盖了 IC 分析、未来函数检测和共线性检查。
3.  **Horizon Analysis 部分**：依据 `scripts/check_time_horizon.py` 和 `docs/HORIZON_A

---
# 5. 训练模型（支持多版本）

使用 XGBoost 对多只股票合并样本进行训练，自动完成训练集 / 验证集划分，并保存模型与评估指标：

```bash
python scripts/train_model.py
```

训练器会自动：
-读取最新 processed 数据
-进行版本管理（例如 h5_excess_index_regression）
-保存模型、评估指标和基础回测结果


# 6. 生成预测信号

基于最新特征数据和训练好的模型，生成逐日逐股的超额收益打分 / 概率：

```bash
python scripts/run_predict.py
```

输出示例：

```text
data/models/{version}/
  └── predictions.parquet   # 包含 ts_code / trade_date / score 等字段
```


# 7. 回测（单策略 / 多策略）

对生成的信号和组合进行历史回测：

```bash
python scripts/run_backtest.py
```

项目支持：
- 单一策略回测
- 多策略组合回测
- Monte Carlo 多样化回测
- Walk-forward 滚动回测
回测结果（净值曲线、收益指标等）会保存到：

```text
data/models/{version}/backtest_results.parquet
logs/                          # 详细日志（含配置与统计信息）
```

📊 数据质量检查

对当前数据集进行质量审计：日期缺口、停牌区间、异常波动、异常成交量等。

```bash
python scripts/check_dataset_quality.py
```

会生成：
- 每只股票的数据质量报告
- 停牌天数、异常涨跌幅、异常成交量统计
- 是否需要剔除的“问题样本”参考
输出示例：

```text
data/dataset_quality_report.csv
```


🧪 当前已完成的能力（项目进展）
- √ 全市场 A 股数据下载与更新
- √ 自动构建价格矩阵
- √ 清洗与质量检查
- √ **数据探索性分析 (EDA) 与可视化** <-- 新增这一项
- √ 特征工程 + 标签生成
- ...


📌 Roadmap（下一步计划）

- 增强可视化模块（净值曲线、收益分布、IC 热力图等）
- 完善多样化回测（Monte Carlo / Walk-Forward）
- 增加特征重要性与模型可解释性（如 SHAP）
- 引入强化学习策略模块
- 接入实盘账户进行模拟交易（live engine）


# 📄 License

MIT License

# ✨ 作者

enhen-x
专注于 A 股量化、ML/RL、工程化研究。欢迎 Star 与交流。



