# scripts/analisis/return_attribution.py
# ============================================================================
# 收益归因分析模块 - 区分模型Alpha和市场Beta贡献
# ============================================================================

import os
import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 路径适配
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.config import GLOBAL_CONFIG
from src.utils.logger import get_logger
from src.utils.io import read_parquet

logger = get_logger()


def analyze_return_attribution(
    start_date: str = None,
    end_date: str = None,
    model_version: str = None
):
    """
    收益归因分析：分离策略收益中的Alpha和Beta成分
    
    参数:
    - start_date: 分析起始日期 (格式: "YYYY-MM-DD")
    - end_date: 分析结束日期
    - model_version: 指定模型版本，默认使用最新
    
    归因公式:
    策略收益 = Alpha (超额收益) + Beta × 市场收益
    """
    
    # 1. 确定模型版本
    models_dir = GLOBAL_CONFIG["paths"]["models"]
    if model_version is None:
        subdirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
        subdirs.sort(reverse=True)
        model_version = subdirs[0] if subdirs else None
    
    if model_version is None:
        logger.error("未找到模型版本")
        return
    
    logger.info(f"=== 收益归因分析 (Model: {model_version}) ===")
    
    # 2. 加载策略预测数据
    model_dir = os.path.join(models_dir, model_version)
    pred_path = os.path.join(model_dir, "predictions.parquet")
    
    if not os.path.exists(pred_path):
        logger.error(f"预测文件不存在: {pred_path}")
        return
    
    pred_df = read_parquet(pred_path)
    pred_df["date"] = pd.to_datetime(pred_df["date"])
    
    # 3. 加载基准指数数据
    idx_code = GLOBAL_CONFIG.get("preprocessing", {}).get("labels", {}).get("index_code", "000852.SH")
    idx_file = os.path.join(GLOBAL_CONFIG["paths"]["data_raw"], f"index_{idx_code.replace('.', '')}.parquet")
    
    if not os.path.exists(idx_file):
        logger.error(f"指数文件不存在: {idx_file}")
        return
    
    idx_df = read_parquet(idx_file)
    idx_df["date"] = pd.to_datetime(idx_df["date"])
    idx_df = idx_df.set_index("date").sort_index()
    
    # 4. 确定分析时间范围
    pred_dates = pred_df["date"].unique()
    pred_min, pred_max = pred_dates.min(), pred_dates.max()
    
    if start_date:
        analysis_start = max(pd.to_datetime(start_date), pred_min)
    else:
        analysis_start = pred_min
    
    if end_date:
        analysis_end = min(pd.to_datetime(end_date), pred_max)
    else:
        analysis_end = pred_max
    
    logger.info(f"分析区间: {analysis_start.strftime('%Y-%m-%d')} ~ {analysis_end.strftime('%Y-%m-%d')}")
    
    # 5. 加载策略净值曲线
    # 重新运行回测以获取完整净值数据
    from src.strategy.signal import TopKSignalStrategy
    from src.backtest.backtester import VectorBacktester
    
    # 动态融合预测分数
    dual_head_cfg = GLOBAL_CONFIG.get("model", {}).get("dual_head", {})
    has_reg = "pred_reg" in pred_df.columns
    has_cls = "pred_cls" in pred_df.columns
    
    if has_reg and has_cls:
        reg_weight = dual_head_cfg.get("regression", {}).get("weight", 0.6)
        cls_weight = dual_head_cfg.get("classification", {}).get("weight", 0.4)
        
        def min_max_normalize(arr):
            arr = np.array(arr)
            min_val, max_val = np.nanmin(arr), np.nanmax(arr)
            if max_val - min_val > 1e-8:
                return (arr - min_val) / (max_val - min_val)
            return np.zeros_like(arr)
        
        pred_reg_norm = min_max_normalize(pred_df["pred_reg"].values)
        pred_cls_norm = min_max_normalize(pred_df["pred_cls"].values)
        pred_df["pred_score"] = reg_weight * pred_reg_norm + cls_weight * pred_cls_norm
    
    # 生成信号并回测
    strategy = TopKSignalStrategy()
    signal_df = strategy.generate(pred_df)
    
    backtester = VectorBacktester()
    out_dir = os.path.join(model_dir, "attribution_analysis")
    os.makedirs(out_dir, exist_ok=True)
    
    metrics = backtester.run(signal_df, output_dir=out_dir, start_date=str(analysis_start.date()), end_date=str(analysis_end.date()))
    
    if "equity_curve" not in metrics:
        logger.error("回测失败，无法获取净值曲线")
        return
    
    equity_curve = metrics["equity_curve"]
    
    # 6. 计算基准指数收益
    idx_sub = idx_df.loc[analysis_start:analysis_end, "close"]
    benchmark_curve = idx_sub / idx_sub.iloc[0]
    
    # 对齐日期
    common_dates = equity_curve.index.intersection(benchmark_curve.index)
    strategy_returns = equity_curve.loc[common_dates]
    benchmark_returns = benchmark_curve.loc[common_dates]
    
    # 7. 核心归因计算
    # 总收益
    total_return = strategy_returns.iloc[-1] / strategy_returns.iloc[0] - 1
    benchmark_total = benchmark_returns.iloc[-1] / benchmark_returns.iloc[0] - 1
    
    # Alpha (超额收益)
    alpha = total_return - benchmark_total
    
    # 日收益率
    strategy_daily = strategy_returns.pct_change().dropna()
    benchmark_daily = benchmark_returns.pct_change().dropna()
    
    # Beta 计算 (使用回归)
    common_idx = strategy_daily.index.intersection(benchmark_daily.index)
    strat_ret = strategy_daily.loc[common_idx].values
    bench_ret = benchmark_daily.loc[common_idx].values
    
    if len(common_idx) > 10:
        # 简单线性回归: strategy_ret = alpha + beta * benchmark_ret
        from scipy.stats import linregress
        beta, reg_alpha, r_value, p_value, std_err = linregress(bench_ret, strat_ret)
    else:
        beta = 1.0
        r_value = 0.0
        reg_alpha = 0.0
    
    # 年化处理
    trading_days = len(common_dates)
    years = trading_days / 252
    
    ann_total_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    ann_benchmark = (1 + benchmark_total) ** (1 / years) - 1 if years > 0 else 0
    ann_alpha = ann_total_return - beta * ann_benchmark
    
    # 8. 打印详细归因结果
    print("\n" + "=" * 70)
    print("[收益归因分析报告]")
    print("=" * 70)
    print(f"分析区间: {analysis_start.strftime('%Y-%m-%d')} ~ {analysis_end.strftime('%Y-%m-%d')}")
    print(f"交易天数: {trading_days} 天 ({years:.2f} 年)")
    print("-" * 70)
    
    print("\n[绝对收益分解]")
    print(f"  策略总收益:     {total_return:>10.2%}")
    print(f"  基准总收益:     {benchmark_total:>10.2%}  (中证1000)")
    print(f"  ----------------------")
    print(f"  超额收益 (Alpha):   {alpha:>10.2%}  = 策略 - 基准")
    
    print("\n[风险调整分析]")
    print(f"  策略 Beta:      {beta:>10.2f}  (相对基准的敏感度)")
    print(f"  R-squared:      {r_value**2:>10.2%}  (收益由市场解释的比例)")
    
    print("\n[年化指标]")
    print(f"  策略年化收益:   {ann_total_return:>10.2%}")
    print(f"  基准年化收益:   {ann_benchmark:>10.2%}")
    print(f"  年化 Alpha:     {ann_alpha:>10.2%}")
    
    # 收益来源归因
    beta_contribution = beta * benchmark_total  # 市场敞口贡献
    alpha_contribution = total_return - beta_contribution  # 真正的选股能力
    
    print("\n[收益来源归因]")
    print(f"  市场敞口贡献 (Beta x 基准):  {beta_contribution:>10.2%}")
    print(f"  选股能力贡献 (Alpha):        {alpha_contribution:>10.2%}")
    
    # 判断结论
    print("\n" + "=" * 70)
    print("[结论]")
    if alpha_contribution > 0.01:  # >1%
        print(f"   [OK] 模型确实产生了 {alpha_contribution:.2%} 的超额收益 (Alpha)")
        print(f"   * 扣除大盘涨幅后，模型仍贡献了 {alpha:.2%} 的超额表现")
    elif alpha_contribution > -0.01:  # -1% ~ 1%
        print(f"   [WARN] 模型超额收益接近于零 ({alpha_contribution:.2%})")
        print(f"   * 策略收益主要来自市场整体上涨，而非选股能力")
    else:
        print(f"   [FAIL] 模型产生了负 Alpha ({alpha_contribution:.2%})")
        print(f"   * 策略跑输基准，选股能力有待提升")
    print("=" * 70)
    
    # 9. 生成可视化图表（简化版，2行1列）
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[
            "净值曲线对比 (策略 vs 中证1000)",
            "超额收益走势 (策略相对基准)"
        ],
        row_heights=[0.5, 0.5],
        vertical_spacing=0.12
    )
    
    # 图1: 净值曲线对比
    fig.add_trace(
        go.Scatter(x=strategy_returns.index, y=strategy_returns.values, 
                   name="策略净值", line=dict(color="red", width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=benchmark_returns.index, y=benchmark_returns.values, 
                   name="基准净值 (中证1000)", line=dict(color="gray", width=2, dash="dash")),
        row=1, col=1
    )
    
    # 图2: 超额收益走势
    excess_curve = strategy_returns / benchmark_returns
    excess_values = excess_curve.values - 1
    
    # 使用颜色区分正负超额
    colors = ['green' if v >= 0 else 'red' for v in excess_values]
    fig.add_trace(
        go.Scatter(x=excess_curve.index, y=excess_values, 
                   name="超额收益率", fill='tozeroy', 
                   fillcolor='rgba(0,128,0,0.3)',
                   line=dict(color='green', width=1.5)),
        row=2, col=1
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    
    # 添加收益归因注释
    annotation_text = f"""
    <b>收益归因:</b><br>
    市场敞口贡献 (β×基准): {beta_contribution:.2%}<br>
    选股能力贡献 (α): {alpha_contribution:.2%}
    """
    fig.add_annotation(
        x=0.98, y=0.02, xref="paper", yref="paper",
        text=annotation_text,
        showarrow=False,
        align="right",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="gray",
        borderwidth=1,
        font=dict(size=11)
    )
    
    # 更新布局
    fig.update_layout(
        title=dict(
            text=f"收益归因分析 | 策略收益: {total_return:.2%} | 基准收益: {benchmark_total:.2%} | 超额收益(α): {alpha:.2%}",
            font=dict(size=14)
        ),
        height=600,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.12, x=0.5, xanchor="center")
    )
    
    # 更新坐标轴标题
    fig.update_xaxes(title_text="日期", row=1, col=1)
    fig.update_yaxes(title_text="净值", row=1, col=1)
    fig.update_xaxes(title_text="日期", row=2, col=1)
    fig.update_yaxes(title_text="超额收益率", tickformat=".1%", row=2, col=1)
    
    # 保存图表
    output_path = os.path.join(out_dir, "return_attribution.png")
    fig.write_image(output_path, width=1200, height=600, scale=2)
    fig.write_html(os.path.join(out_dir, "return_attribution.html"))
    
    logger.info(f"归因分析图表已保存至: {output_path}")
    
    # 返回详细结果
    return {
        "total_return": total_return,
        "benchmark_return": benchmark_total,
        "alpha": alpha,
        "beta": beta,
        "r_squared": r_value ** 2,
        "ann_total_return": ann_total_return,
        "ann_benchmark": ann_benchmark,
        "ann_alpha": ann_alpha,
        "beta_contribution": beta_contribution,
        "alpha_contribution": alpha_contribution,
        "trading_days": trading_days
    }



def analyze_random_periods(
    model_version: str = None,
    samples: int = 200,
    duration_days: int = 35
):
    """
    随机周期分析：统计策略在随机抽取的固定时长窗口下的胜率
    
    参数:
    - samples: 抽样次数
    - duration_days: 每个窗口的交易日数量 (35交易日约等于1.5个月)
    """
    logger.info(f"=== 开始随机周期分析 (抽样: {samples}次, 窗口: {duration_days}天) ===")
    
    # 1. 准备数据
    models_dir = GLOBAL_CONFIG["paths"]["models"]
    if model_version is None:
        subdirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
        subdirs.sort(reverse=True)
        model_version = subdirs[0] if subdirs else None
        
    model_dir = os.path.join(models_dir, model_version)
    pred_path = os.path.join(model_dir, "predictions.parquet")
    idx_code = GLOBAL_CONFIG.get("preprocessing", {}).get("labels", {}).get("index_code", "000852.SH")
    idx_file = os.path.join(GLOBAL_CONFIG["paths"]["data_raw"], f"index_{idx_code.replace('.', '')}.parquet")
    
    pred_df = read_parquet(pred_path)
    pred_df["date"] = pd.to_datetime(pred_df["date"])
    
    idx_df = read_parquet(idx_file)
    idx_df["date"] = pd.to_datetime(idx_df["date"])
    idx_df = idx_df.set_index("date").sort_index()
    
    # 2. 运行全量回测获取每日净值
    from src.strategy.signal import TopKSignalStrategy
    from src.backtest.backtester import VectorBacktester
    
    # 简单的融合逻辑 (如果已存在融合列则跳过)
    if "pred_score" not in pred_df.columns:
        pred_df["pred_score"] = pred_df["pred_reg"] # 简化假设
        
    strategy = TopKSignalStrategy()
    signal_df = strategy.generate(pred_df)
    backtester = VectorBacktester()
    
    # 为了速度，不画图，只获取数据
    # 创建临时输出目录
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        metrics = backtester.run(signal_df, output_dir=tmp_dir)
        
    equity_curve = metrics["equity_curve"]
    # 确保和指数日期对齐
    common_dates = equity_curve.index.intersection(idx_df.index)
    strat_nav = equity_curve.loc[common_dates]
    bench_nav = idx_df.loc[common_dates, "close"]
    
    # 3. 随机抽样
    # 有效起始点：0 到 len - duration
    total_days = len(common_dates)
    if total_days < duration_days:
        logger.error("数据长度不足")
        return
        
    valid_starts = np.arange(total_days - duration_days)
    # 随机选择起始点
    chosen_starts = np.random.choice(valid_starts, size=samples, replace=True)
    
    results = []
    
    for start_idx in tqdm(chosen_starts, desc="分析随机窗口"):
        end_idx = start_idx + duration_days
        
        # 窗口期数据
        s_start = strat_nav.iloc[start_idx]
        s_end = strat_nav.iloc[end_idx]
        b_start = bench_nav.iloc[start_idx]
        b_end = bench_nav.iloc[end_idx]
        
        strat_ret = s_end / s_start - 1
        bench_ret = b_end / b_start - 1
        alpha = strat_ret - bench_ret
        
        start_date = common_dates[start_idx]
        
        results.append({
            "start_date": start_date,
            "strat_ret": strat_ret,
            "bench_ret": bench_ret,
            "alpha": alpha,
            "win": alpha > 0
        })
        
    df_res = pd.DataFrame(results)
    
    # 4. 统计结果
    win_rate = df_res["win"].mean()
    avg_alpha = df_res["alpha"].mean()
    median_alpha = df_res["alpha"].median()
    
    print("\n" + "="*60)
    print(f"🎲 随机周期分析结果 (基于过去 {len(common_dates)} 个交易日)")
    print("="*60)
    print(f"抽样参数: {samples} 次测试, 每次持仓 {duration_days} 天 (约1.5个月)")
    print("-" * 60)
    print(f"🏆 胜率 (跑赢基准):     {win_rate:>8.2%}")
    print(f"💰 平均超额收益 (Mean): {avg_alpha:>8.2%}")
    print(f"📊 中位数超额 (Median): {median_alpha:>8.2%}")
    print(f"📈 最好表现:            {df_res['alpha'].max():>8.2%}")
    print(f"📉 最差表现:            {df_res['alpha'].min():>8.2%}")
    print("="*60)
    
    # 保存统计结果到 CSV
    summary_path = os.path.join(model_dir, "attribution_analysis", "random_analysis_summary.csv")
    df_res.to_csv(os.path.join(model_dir, "attribution_analysis", "random_analysis_details.csv"), index=False)
    
    summary_data = {
        "timestamp": [pd.Timestamp.now()],
        "samples": [samples],
        "duration_days": [duration_days],
        "win_rate": [win_rate],
        "avg_alpha": [avg_alpha],
        "median_alpha": [median_alpha],
        "max_alpha": [df_res['alpha'].max()],
        "min_alpha": [df_res['alpha'].min()]
    }
    pd.DataFrame(summary_data).to_csv(summary_path, index=False)
    logger.info(f"详细统计已保存: {summary_path}")

    # 生成综合分析图表
    try:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "超额收益(Alpha)分布直方图", "策略收益 vs 基准收益散点图",
                "Alpha随起始时间变化趋势", "不同年份的平均胜率heatmap(如有)"
            ],
            specs=[[{"type": "histogram"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "table"}]]
        )
        
        # 1. Alpha 分布直方图
        fig.add_trace(
            go.Histogram(x=df_res["alpha"], nbinsx=30, name="Alpha分布", marker_color='blue', opacity=0.7),
            row=1, col=1
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="0%", row=1, col=1)
        # fig.add_vline(x=-0.0607, line_dash="dash", line_color="green", annotation_text="当前实盘", row=1, col=1)

        # 2. 策略 vs 基准 散点图
        fig.add_trace(
            go.Scatter(
                x=df_res["bench_ret"], 
                y=df_res["strat_ret"], 
                mode='markers',
                marker=dict(
                    color=df_res["alpha"], 
                    colorscale='RdBu', 
                    showscale=True,
                    colorbar=dict(title="Alpha", len=0.4, y=0.8)
                ),
                text=[f"时间: {d.date()}<br>Alpha: {a:.2%}" for d, a in zip(df_res["start_date"], df_res["alpha"])],
                name="样本点"
            ),
            row=1, col=2
        )
        # 添加 y=x 参考线
        min_ret = min(df_res["bench_ret"].min(), df_res["strat_ret"].min())
        max_ret = max(df_res["bench_ret"].max(), df_res["strat_ret"].max())
        fig.add_trace(
            go.Scatter(x=[min_ret, max_ret], y=[min_ret, max_ret], mode='lines', line=dict(color='gray', dash='dash'), name="跑平基准"),
            row=1, col=2
        )

        # 3. Alpha 随时间变化
        df_sorted = df_res.sort_values("start_date")
        fig.add_trace(
            go.Scatter(x=df_sorted["start_date"], y=df_sorted["alpha"], mode='lines', name="Alpha趋势", line=dict(width=1)),
            row=2, col=1
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=1)

        # 4. 统计表格
        fig.add_trace(
            go.Table(
                header=dict(values=["指标", "数值"], fill_color='paleturquoise', align='left'),
                cells=dict(values=[
                    ["抽样次数", "胜率 (Win Rate)", "平均 Alpha", "中位数 Alpha", "最大 Alpha", "最小 Alpha"],
                    [
                        f"{samples}",
                        f"{win_rate:.2%}",
                        f"{avg_alpha:.2%}",
                        f"{median_alpha:.2%}",
                        f"{df_res['alpha'].max():.2%}",
                        f"{df_res['alpha'].min():.2%}"
                    ]
                ], fill_color='lavender', align='left')
            ),
            row=2, col=2
        )

        fig.update_layout(
            title_text=f"随机周期分析报告 (窗口={duration_days}交易日, 样本={samples})",
            height=900,
            showlegend=False
        )
        
        # 坐标轴标签
        fig.update_xaxes(title_text="Alpha", tickformat=".1%", row=1, col=1)
        fig.update_yaxes(title_text="频次", row=1, col=1)
        
        fig.update_xaxes(title_text="基准收益", tickformat=".1%", row=1, col=2)
        fig.update_yaxes(title_text="策略收益", tickformat=".1%", row=1, col=2)
        
        fig.update_xaxes(title_text="起始日期", row=2, col=1)
        fig.update_yaxes(title_text="Alpha", tickformat=".1%", row=2, col=1)

        out_dir = os.path.join(model_dir, "attribution_analysis")
        os.makedirs(out_dir, exist_ok=True)
        
        # 保存 HTML (交互式)
        out_path_html = os.path.join(out_dir, "random_analysis.html")
        fig.write_html(out_path_html)
        
        # 保存 PNG (静态)
        out_path_png = os.path.join(out_dir, "random_analysis.png")
        fig.write_image(out_path_png, scale=2)
        
        logger.info(f"随机分析图表已保存:\n  HTML: {out_path_html}\n  PNG:  {out_path_png}")
    except Exception as e:
        logger.error(f"生成随机分析图表失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    from tqdm import tqdm
    
    parser = argparse.ArgumentParser(description="收益归因分析工具")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "real", "random"], help="分析模式: all(全部), real(仅实盘), random(仅随机验证)")
    parser.add_argument("--start_date", type=str, default="2025-11-27", help="实盘分析开始日期 (YYYY-MM-DD)")
    parser.add_argument("--samples", type=int, default=500, help="随机验证抽样次数")
    parser.add_argument("--duration", type=int, default=35, help="随机验证持仓天数")
    
    args = parser.parse_args()
    
    # 1. 分析当前实盘周期
    if args.mode in ["all", "real"]:
        print(f"\n>>> 分析 1: 当前实盘周期 ({args.start_date} ~ 今)")
        analyze_return_attribution(
            start_date=args.start_date,
            end_date=None 
        )
    
    # 2. 随机周期验证
    if args.mode in ["all", "random"]:
        print("\n>>> 分析 2: 历史随机周期验证 (验证策略稳健性)")
        analyze_random_periods(
            samples=args.samples,
            duration_days=args.duration
        )
