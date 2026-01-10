# Copilot 指南（Quant_A_Share_V2.0）
## 总览
- 项目链路为数据/预处理 → 模型 → 策略/回测 → 实盘，详情见 [architecture.md](architecture.md)；所有表默认以 (date, ts_code) 为多级索引，并遵循 feat_/label_ 命名。
- 数据血缘：scripts/date_landing → [data/raw](data/raw) → [data/raw_cleaned](data/raw_cleaned) → preprocessing → [data/processed](data/processed) → [data/models](data/models)；后续模块默认这些目录存在。
- 每个模型版本目录下必须包含 predictions.parquet（列: date, ts_code, pred），策略与回测直接读取该文件，无额外校验。
## 配置与流程
- [config/main.yaml](config/main.yaml) 是唯一可信配置：路径、股票池过滤、特征开关、模型参数、双头权重、策略/回测开关都来自这里，改代码前先同步配置。
- 离线全流程命令：`python scripts/date_landing/init_stock_pool.py` → `python scripts/date_landing/download_data.py`（或 auto_run）→ `python scripts/analisis/clean_and_check.py` → `python scripts/feature_create/rebuild_features.py` → `python scripts/model_train/run_walkforward.py` → `python scripts/back_test/run_backtest.py`。
- 默认要求 GPU 训练（XGBoost tree_method=hist, device=cuda；LightGBM device=gpu）；只有在配置暴露开关时才允许回退 CPU。
## 预处理规范
- [src/preprocessing/pipeline.py](src/preprocessing/pipeline.py) 通过 paths.* 读取/写入，调用方必须传入完整配置 dict，保证 batch.concat_file 指向 all_stocks.parquet。
- [src/preprocessing/features.py](src/preprocessing/features.py) 依据 enable_* 开关输出 feat_* 列；新增因子时务必添加配置开关，并保持 pandas/numpy 向量化实现以免拖慢批处理。
- [src/preprocessing/labels.py](src/preprocessing/labels.py) 固定使用 VWAP 计价、三相屏障、winsorize、涨停过滤等逻辑；保持超额收益语义并复用现有工具函数。
## 建模约束
- [src/model/trainer.py](src/model/trainer.py) 在 use_feature_selection=true 时会从 data/processed/selected_features.txt 读取列名，新增特征需同步更新该文件。
- dual_head 逻辑完全依赖 model.dual_head 配置；回归/分类权重在 trainer/predictor 融合前需相加为 1，并尊重 normalize 标志。
- Walk-forward（model.walk_forward）默认按年扩张窗口切片；新实验输出放在 data/models/WF_*，方便 `scripts/back_test/run_backtest.py` 自动读取最新版本。
## 策略与回测
- [src/strategy/signal.py](src/strategy/signal.py) 默认引用 config.strategy 中的 score_smoothing、min_pred、top_k 等阈值，不要在代码里写死。
- [src/backtest/backtester.py](src/backtest/backtester.py) 需要 predictions.parquet + processed 价格数据，严格执行 T+1、entry_price=vwap、交易成本来自 config.strategy/entry_price 与 backtest.cost。
- 所有报表/图像输出到 [reports](reports) 与 [figures](figures)，文件名需带时间戳 YYYYMMDD_HHMMSS，供 notebook/脚本按最新结果 glob。
## 实盘与自动化
- [scripts/live](scripts/live) 依赖 [data/live_trading](data/live_trading) 下的配置（含雪球 cookies）；不要把凭证写进代码，运行时读取 config.example.txt 的副本。
- [architecture/require.md](architecture/require.md) 规定了每日计划、失败重试次数、告警方式；新增 cron 脚本必须满足：clean_and_check 缺失率>5% 立刻终止并报警。
- [src/live](src/live) 里的 trade recorder / scheduler 依赖“按持有天数平均仓位”的滚动逻辑，并把心跳写入 logs/；接入新监控时输出结构化 JSON。
## 其他约定
- 统一使用 [src/utils/logger.py](src/utils/logger.py) 写日志、[src/utils/config.py](src/utils/config.py) 读配置，保证路径与环境变量一致。
- 命名约定：feat_*（特征）、label_*（标签）、pred（预测得分）、WF_YYYYMMDD_*（walk-forward 版本目录）；脚本通常按这些前缀 glob。
- 合并全市场 parquet 容易爆内存，优先使用 batch.save_each + concat_all 组合；诊断脚本不要一次性读入全部个股。
- 新增流程尽量复用 scripts/ 下的入口（argparse + 调用 src 模块），重逻辑放在 src，便于测试与复用。
