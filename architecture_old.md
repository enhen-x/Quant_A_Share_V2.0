# Quant_A_Share_V2.0 架构说明（architecture）

> 最后更新: 2025-01-19 | 版本: V2.2

---

## 1. 项目定位与目标

Quant_A_Share_V2.0 的定位是：

面向个人投资者的、基于机器学习的 A 股多标的选股与推荐系统，
同时提供完整的历史回测、Walk-forward 验证与（未来）实盘对接能力。

围绕这个定位，本项目希望支持：

每日/定期股票推荐：基于最新行情和模型预测，为个人投资者生成一份可执行的推荐列表；

可回测的策略逻辑：任何“推荐规则”都可以在历史数据上跑回测，给出年化收益、回撤、夏普等指标；

可扩展的研究平台：方便接入新的数据源、特征、模型和策略，而不需要大范围改动已有代码。

2. 总体架构概览

项目主要目录结构如下（略）：

config/
data/
src/
  data_source/
  preprocessing/
  model/
  strategy/
  backtest/
  live/
  utils/
scripts/
notebooks/
docs/


抽象成三层：

数据 & 机器学习层（Data & ML Engine）

负责数据：下载、清洗、特征工程、标签生成、模型训练与预测。
数据下载：

使用baostock、akshare进行下载，并进行






策略 & 回测层（Strategy & Backtest Engine）

负责将模型预测转换成可执行的交易信号与组合，并在历史数据上进行回测。

推荐 & 使用层（Recommendation & Interface）

负责面向“个人用户”的推荐输出与脚本入口（如每日推荐、一次性回测、walk-forward 等）。

3. 模块说明
### 3.1 config/ — 配置中心

项目使用单一配置文件 `config/main.yaml`，包含以下主要配置模块：

- **paths**：数据、模型、日志等路径配置
- **data_source**：数据源选择（baostock/akshare）
- **stock_pool**：股票池过滤规则（排除ST、科创板等）
- **preprocessing**：特征工程、标签生成、数据质检参数
- **model**：XGBoost 模型参数、滚动训练配置、TensorBoard 监控、特征筛选开关
- **strategy**：选股逻辑、Top-K、风控过滤规则
- **backtest**：回测窗口、交易成本、轮动模式

原则：所有可调参数集中在 `main.yaml` 中，脚本只负责"读配置 + 调用模块"。

3.2 data/ — 数据存储

data/raw/：原始行情与基础数据（按数据源或股票划分）。

data/processed/：特征与标签处理后的数据。

典型：data/processed/{version}/all_stocks.parquet

data/meta/：元信息（股票名称、行业分类、交易日历等）。

data/models/：模型文件与预测结果。

data/models/{version}/model.pkl

data/models/{version}/predictions.parquet

3.3 src/data_source/ — 数据源统一接口

base.py：DataSource 抽象基类；定义统一的接口，如 get_daily(ts_code, start, end)。

akshare_source.py / baostock_source.py：具体数据源实现。

datahub.py：对外统一入口。
其它模块只调用 datahub，不直接依赖 Akshare/Baostock 等第三方库。

目标： 替换数据源只需改 data_source/，其余模块无感知。

3.4 src/preprocessing/ — 特征 & 标签管道

features.py：技术指标、因子等的计算逻辑。

label.py：标签生成（如 label_ret_5d、label_excess_5d）。

pipeline.py：从 raw 到 all_stocks.parquet 的完整流水线。

输出统一格式： data/processed/{version}/all_stocks.parquet

约定（建议）：

索引：date, ts_code

基础列：open, high, low, close, volume, amount, ...

特征列：统一以 feat_ 前缀命名，如 feat_ema_12, feat_rsi_14。

标签列：统一以 label_ 前缀命名，如 label_ret_5d, label_excess_5d。

### 3.5 src/model/ — 模型训练与预测

**核心模块：**

- `trainer.py`：训练器，支持特征筛选，读入数据并训练模型

- `xgb_model.py`：XGBoost 模型封装（GPU/Hist 加速、TensorBoard 回调、早停机制）

- `training_monitor.py` **[NEW]**：TensorBoard 训练监控
  - 实时记录训练/验证损失曲线
  - 记录特征重要性和超参数配置
  - 支持多次实验对比

- `predictor.py`：推断接口

- `calibrator.py`：概率/分位数校准模块

模型输出契约：

data/models/{version}/predictions.parquet，至少包含：

date：交易日

ts_code：股票代码

pred：模型预测得分（越大通常表示预期收益越高）

所有后续策略 & 推荐层都以此为统一输入。

3.6 src/strategy/ — 策略 & 推荐逻辑

当前包含：

signal.py：将模型预测转换为交易信号（如 Top-K 排名、打分）。

portfolio.py：根据信号构建组合（持仓权重）。

risk_filter.py：风险过滤（ST、新股、涨跌停、流动性等）。

标准数据契约：

预测输入： pred_df，至少含 date, ts_code, pred。

信号输出（signal.py）： signal_df，建议包含：

date, ts_code, score, signal
其中 score 一般来自 pred，signal 为入选标记或方向。

风险过滤（risk_filter.py）：
输入 signal_df + 必要的行情/标记，输出过滤后的信号。

组合构建（portfolio.py）：
输入过滤后的信号 + 价格数据，输出：

weights: DataFrame，index = date, columns = ts_code；

daily_return: Series，组合日收益；

equity_curve: Series，组合资金曲线。

未来可以增加：

recommend.py：面向个人用户的推荐接口，例如：

根据 “个人资金/风险偏好/黑名单” 对组合进一步裁剪；

输出单日/当前时点的推荐列表。

3.7 src/backtest/ — 回测框架

backtester.py：回测主逻辑，封装一次完整回测流程：

预测 → 信号 → 风控 → 组合 → 资金曲线 → 指标。

metrics.py：回测指标计算（累计收益、年化收益、波动率、夏普、最大回撤等）。

monte_carlo.py：Monte Carlo 多样化回测（未来计划）。

walk_forward.py：Walk-forward 滚动训练 **[已完成]** - 支持扩张窗口/滚动窗口。

event_engine.py：事件驱动回测（细粒度撮合）。

核心接口：

Backtester.run_with_data(pred_df) -> result_dict
其中 pred_df 为 predictions.parquet 格式。

result_dict 至少包含：

{
    "equity_curve": pd.Series,
    "daily_return": pd.Series,
    "weights": pd.DataFrame,
    "metrics": dict,  # 来自 metrics.compute_metrics(...)
}

3.8 src/live/ — 实盘/准实盘模块（预留）

live_engine.py：实盘运行入口，可与券商/模拟交易 API 对接。

order_simulator.py：模拟撮合与成交。

monitor.py：实时监控预测与策略输出。

设计目标：

尽量复用 strategy 与 backtest 中已有逻辑；

简单更换“数据输入源”（从历史数据 → 实时行情）就能跑起来。

3.9 src/utils/ — 通用工具

logger.py：统一日志记录。

calendar.py：交易日历、公休日处理。

io.py：通用 IO 工具（读写 parquet/csv/pickle 等）。

common.py：通用小工具函数。

3.10 scripts/ — 命令行脚本入口

面向使用者的“场景入口”，而不是逻辑实现地。

目前主要脚本包括：

update_data.py：更新原始数据到 data/raw/。

rebuild_features.py：执行预处理管道，生成 all_stocks.parquet。

train_model.py：训练指定版本模型并产出 predictions.parquet。

run_backtest.py：运行一次回测（给定模型版本 + 策略配置）。

run_walkforward.py：运行 walk-forward 验证（待完善）。

run_live.py：运行实盘/模拟盘引擎（待完善）。

（未来）run_recommendation.py：面向个人用户的每日推荐。

4. 数据流与关键数据契约
4.1 全流程数据流（高层视角）

数据更新：

scripts/update_data.py
→ 调用 data_source.datahub
→ 写入 data/raw/。

特征 & 标签生成：

scripts/rebuild_features.py
→ 调用 preprocessing.pipeline
→ 读 data/raw/
→ 生成 data/processed/{version}/all_stocks.parquet。

模型训练 & 预测：

scripts/train_model.py
→ 读 config/model_config.yaml
→ 读 all_stocks.parquet
→ model.trainer 训练模型并保存
→ model.predictor 生成 predictions.parquet。

策略 & 回测：

scripts/run_backtest.py
→ 读 backtest_config.yaml + strategy_config.yaml
→ 读 predictions.parquet
→ 调用 Backtester.run_with_data(pred_df)
→ 输出资金曲线 & 回测指标。

推荐输出（未来）：

scripts/run_recommendation.py
→ 读最新 predictions.parquet
→ 读 user_profile.yaml（个人约束）
→ 调用 strategy.recommend.make_recommendation()
→ 输出推荐列表（csv/html/markdown）。

5. 使用场景 & 典型流程
5.1 研究模式（离线）

更新数据（按周/月）：

python scripts/update_data.py
python scripts/rebuild_features.py


训练新版本模型：

python scripts/train_model.py --version h5_excess_index_regression


回测模型 + 策略组合：

python scripts/run_backtest.py --version h5_excess_index_regression


分析结果、调参/调策略。

5.2 日常推荐模式（面向个人）

每日/每周：

使用历史数据 + 最新一部分数据做预测，更新 predictions.parquet；

运行 run_recommendation.py 生成推荐列表；

手动或通过券商 APP 执行调仓。

（run_recommendation.py 作为未来扩展重点。）

6. 可扩展性设计要点

数据源可替换：所有数据获取通过 data_source.datahub，不直接依赖具体供应商。

特征/标签可扩展：features.py 与 label.py 只依赖 all_stocks 的输入格式，可以随时增加/修改因子与标签。

模型可替换：只要新的模型实现能够输出 predictions.parquet，就可以无缝接入策略层与回测层。

策略可组合：signal / risk_filter / portfolio 拆分独立，方便组合不同选股逻辑、风控规则与组合构建方式。

回测模式可扩展：在 backtest 中新增 Monte Carlo、Walk-forward 等，只要底层调用 Backtester 即可。

7. 命名与约定（建议）

特征列：统一使用 feat_ 前缀。

标签列：统一使用 label_ 前缀。

模型预测列：统一使用 pred。

策略信号列：统一使用 score（打分）和 signal（入选/方向）。

文件命名：

处理后的全市场数据：all_stocks.parquet

预测结果：predictions.parquet

配置文件中使用的 version 作为模型/数据版本号。

这份 architecture.md 可以作为你之后所有“改代码之前”的参照物：
新需求出现时，先看它落在这套架构的哪一层、影响哪些契约，再决定是加文件、加配置，还是改现有模块。

---

## 8. 已完成功能 (V2.0 - V2.1)

| 功能 | 状态 | 说明 |
|------|------|------|
| 数据下载与清洗 | Done | 多源支持、断点续传、智能质检 |
| 特征工程 | Done | MA/MACD/RSI/KDJ/BOLL/量价等 |
| 标签生成 | Done | VWAP计价、超额收益、一字板过滤 |
| XGBoost 训练 | Done | GPU加速、早停机制 |
| Walk-Forward 滚动训练 | Done | 按年滚动、无未来函数 |
| TensorBoard 训练监控 | Done | 损失曲线、特征重要性可视化 |
| 特征筛选 | Done | 基于 IC 和相关性自动筛选 |
| 正则化优化 | Done | L1/L2 正则化、参数调优 |
| 策略回测 | Done | 分仓轮动、成本扣除 |
| 压力测试 | Done | 成本敏感性、熊市生存测试 |
| 每日推荐 | Done | 自动选股、涨停过滤 |

---

## 9. 未来计划 (Roadmap)

### P1 (近期优先)
- SHAP 可解释性分析：输出特征贡献度
- 交互式回测报告：生成 HTML 格式报告
- 模型版本对比工具

### P2 (中期计划)
- Monte Carlo 模拟：评估极端风险
- 多模型融合：XGBoost + LightGBM + CatBoost
- 因子挖掘框架

### P3 (远期愿景)
- 深度学习模型：LSTM / Transformer
- Live Engine 实盘：对接券商 API
- Web Dashboard：图形化管理平台

---

## 10. 更新日志

| 日期 | 版本 | 更新内容 |
|------|------|----------|
| 2025-12-15 | V2.1 | 添加 TensorBoard 监控、特征筛选、正则化优化 |
| 2025-12-14 | V2.0 | Walk-Forward 滚动训练、压力测试 |
| 2025-12-07 | V1.5 | 策略回测、每日推荐 |
| 2025-12-01 | V1.0 | 初始版本，数据下载与特征工程 |