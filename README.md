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
Quant_A_SHARE_V2.0/
├── config/                     # 配置文件（路径、特征、模型、策略）
│   ├── paths.yaml
│   ├── data_config.yaml
│   ├── model_config.yaml
│   ├── preprocessing.yaml
│   └── strategy_config.yaml
│
├── data/
│   ├── raw/                    # 原始行情（每日更新）
│   ├── raw_cleaned/            # [新增] 清洗后的标准行情（无重复/无异常/无零值）
│   ├── processed/              # 特征矩阵 & 标签
│   ├── price/                  # price_matrix.parquet
│   ├── index/                  # 指数行情
│   └── meta/                   # 股票列表 & 交易日历
│
├── figures/                    # [新增] EDA 生成的可视化图表 (.png)
├── reports/                    # [新增] EDA 生成的数据报告 (.csv)
│
├── scripts/                    # 命令行脚本
│   ├── update_data.py          # 更新 raw 数据
│   ├── build_index.py
│   ├── build_price_matrix.py
│   ├── rebuild_features.py
│   ├── train_model.py
│   ├── run_predict.py
│   ├── run_backtest.py
│   ├── run_build_portfolio.py
│   ├── run_build_signals.py
│   ├── check_dataset_quality.py
│   └── test_datahub.py
│
├── src/
│   ├── data_source/            # 数据源（akshare / baostock / datahub）
│   ├── preprocessing/          # 特征、标签、Pipeline
│   ├── model/                  # XGBoost 模型、训练器、预测器
│   ├── strategy/               # 信号、组合与风险过滤
│   ├── backtest/               # 回测框架
│   └── utils/                  # 通用工具（日志、IO、质量评估）
│
├── logs/                       # 日志输出（自动生成）
├── README.md
└── requirements.txt
```

## 🚀 Quick Start / 使用说明
#  1. 安装依赖

项目基于 Python 3.10+（建议使用 Conda 虚拟环境）：

```bash
pip install -r requirements.txt
```


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


# 3. 构建特征（features）与标签（labels）

基于 data/raw/ 中的行情数据，计算技术指标、因子，并生成训练标签，输出到 data/processed/：

```bash
python scripts/rebuild_features.py
```


# 4. 训练模型（支持多版本）

使用 XGBoost 对多只股票合并样本进行训练，自动完成训练集 / 验证集划分，并保存模型与评估指标：

```bash
python scripts/train_model.py
```

训练器会自动：
-读取最新 processed 数据
-进行版本管理（例如 h5_excess_index_regression）
-保存模型、评估指标和基础回测结果


# 5. 生成预测信号

基于最新特征数据和训练好的模型，生成逐日逐股的超额收益打分 / 概率：

```bash
python scripts/run_predict.py
```

输出示例：

```text
data/models/{version}/
  └── predictions.parquet   # 包含 ts_code / trade_date / score 等字段
```


# 6. 回测（单策略 / 多策略）

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



