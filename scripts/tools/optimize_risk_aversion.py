# scripts/tools/optimize_risk_aversion.py
"""
效用函数风险厌恶系数（risk_aversion）优化工具

使用网格搜索找到最优的 risk_aversion 值，以最大化夏普比率或其他指标。
"""

import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import plotly.graph_objects as go

# 路径适配
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.config import GLOBAL_CONFIG
from src.utils.logger import get_logger
from src.utils.io import read_parquet, ensure_dir

logger = get_logger()


def load_predictions(model_version: str = None):
    """加载预测数据"""
    models_dir = GLOBAL_CONFIG["paths"]["models"]
    
    if model_version is None:
        # 自动选择最新版本
        subdirs = [d for d in os.listdir(models_dir) 
                   if os.path.isdir(os.path.join(models_dir, d)) and d.startswith("WF_")]
        if not subdirs:
            raise FileNotFoundError("未找到任何 WF_ 模型目录")
        subdirs.sort(reverse=True)
        model_version = subdirs[0]
    
    pred_path = os.path.join(models_dir, model_version, "predictions.parquet")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"预测文件不存在: {pred_path}")
    
    logger.info(f"加载预测数据: {pred_path}")
    df = read_parquet(pred_path)
    df["date"] = pd.to_datetime(df["date"])
    
    return df, model_version


def fuse_with_utility(pred_return: np.ndarray, pred_risk: np.ndarray, 
                      risk_aversion: float, normalize: bool = True) -> np.ndarray:
    """
    使用效用函数融合预测
    公式: U = E[R] - λ * σ²
    """
    if normalize:
        # Min-Max 归一化
        r_min, r_max = pred_return.min(), pred_return.max()
        v_min, v_max = pred_risk.min(), pred_risk.max()
        
        if r_max - r_min > 1e-8:
            pred_return = (pred_return - r_min) / (r_max - r_min)
        else:
            pred_return = np.zeros_like(pred_return)
            
        if v_max - v_min > 1e-8:
            pred_risk = (pred_risk - v_min) / (v_max - v_min)
        else:
            pred_risk = np.zeros_like(pred_risk)
    
    return pred_return - risk_aversion * (pred_risk ** 2)


def simple_backtest(df: pd.DataFrame, top_k: int = 5, horizon: int = 4) -> dict:
    """
    使用实际的 VectorBacktester 进行回测
    
    Args:
        df: 包含 pred_score 的数据
        top_k: 每期选股数量
        horizon: 持仓周期（天）
    
    Returns:
        dict: 包含年化收益、夏普比率、最大回撤等
    """
    from src.strategy.signal import TopKSignalStrategy
    from src.backtest.backtester import VectorBacktester
    import tempfile
    import shutil
    
    try:
        # 生成策略信号
        strategy = TopKSignalStrategy(top_k=top_k)
        signal_df = strategy.generate(df)
        
        # 使用临时目录
        temp_dir = tempfile.mkdtemp()
        
        try:
            # 运行回测
            backtester = VectorBacktester()
            metrics = backtester.run(signal_df, output_dir=temp_dir)
            
            return {
                "sharpe": metrics.get("sharpe", 0),
                "annual_return": metrics.get("annual_return", 0),
                "max_dd": metrics.get("max_drawdown", 0),
                "total_return": metrics.get("total_return", 0),
                "win_rate": metrics.get("win_rate", 0)
            }
        finally:
            # 清理临时目录
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    except Exception as e:
        logger.warning(f"回测失败: {e}")
        return {"sharpe": -999, "annual_return": 0, "max_dd": 0, "total_return": 0, "win_rate": 0}


def optimize_risk_aversion(
    ra_range: tuple = (0.1, 3.0),
    n_points: int = 30,
    target_metric: str = "sharpe",
    top_k: int = 5,
    model_version: str = None
):
    """
    网格搜索最优风险厌恶系数
    
    Args:
        ra_range: (min, max) 搜索范围
        n_points: 网格点数
        target_metric: 优化目标 ("sharpe", "annual_return", "max_dd")
        top_k: 每日选股数量
        model_version: 指定模型版本，None 则自动选择最新
    """
    logger.info("=" * 60)
    logger.info("=== 风险厌恶系数 (risk_aversion) 优化 ===")
    logger.info("=" * 60)
    
    # 1. 加载数据
    df, version = load_predictions(model_version)
    
    if "pred_return" not in df.columns or "pred_risk" not in df.columns:
        raise ValueError("预测数据中缺少 pred_return 或 pred_risk 列，请确保使用双头模型")
    
    logger.info(f"模型版本: {version}")
    logger.info(f"数据量: {len(df):,} 行")
    logger.info(f"日期范围: {df['date'].min().date()} ~ {df['date'].max().date()}")
    logger.info(f"搜索范围: {ra_range[0]} ~ {ra_range[1]}")
    logger.info(f"优化目标: {target_metric}")
    
    # 2. 网格搜索
    ra_values = np.linspace(ra_range[0], ra_range[1], n_points)
    results = []
    
    pred_return = df["pred_return"].values
    pred_risk = df["pred_risk"].values
    
    logger.info("\n开始网格搜索...")
    for ra in tqdm(ra_values, desc="Testing risk_aversion"):
        # 融合预测
        df["pred_score"] = fuse_with_utility(pred_return, pred_risk, ra, normalize=True)
        
        # 回测
        metrics = simple_backtest(df, top_k=top_k)
        metrics["risk_aversion"] = ra
        results.append(metrics)
    
    # 3. 分析结果
    results_df = pd.DataFrame(results)
    
    # 找最优
    if target_metric == "max_dd":
        # 最大回撤越接近 0 越好
        best_idx = results_df[target_metric].abs().idxmin()
    else:
        best_idx = results_df[target_metric].idxmax()
    
    best_ra = results_df.loc[best_idx, "risk_aversion"]
    best_metrics = results_df.loc[best_idx]
    
    logger.info("\n" + "=" * 60)
    logger.info("=== 优化结果 ===")
    logger.info("=" * 60)
    logger.info(f"最优 risk_aversion: {best_ra:.4f}")
    logger.info(f"  - 夏普比率: {best_metrics['sharpe']:.2f}")
    logger.info(f"  - 年化收益: {best_metrics['annual_return']:.2f}%")
    logger.info(f"  - 最大回撤: {best_metrics['max_dd']:.2f}%")
    logger.info(f"  - 总收益率: {best_metrics['total_return']:.2f}%")
    logger.info(f"  - 日胜率: {best_metrics['win_rate']:.2f}%")
    
    # 4. 可视化
    output_dir = os.path.join(GLOBAL_CONFIG["paths"]["reports"], "optimization")
    ensure_dir(output_dir)
    
    fig = go.Figure()
    
    # 夏普比率曲线
    fig.add_trace(go.Scatter(
        x=results_df["risk_aversion"],
        y=results_df["sharpe"],
        name="Sharpe Ratio",
        mode="lines+markers",
        line=dict(color="blue", width=2),
        yaxis="y1"
    ))
    
    # 年化收益曲线
    fig.add_trace(go.Scatter(
        x=results_df["risk_aversion"],
        y=results_df["annual_return"],
        name="Annual Return (%)",
        mode="lines+markers",
        line=dict(color="green", width=2),
        yaxis="y2"
    ))
    
    # 最大回撤曲线
    fig.add_trace(go.Scatter(
        x=results_df["risk_aversion"],
        y=results_df["max_dd"],
        name="Max Drawdown (%)",
        mode="lines+markers",
        line=dict(color="red", width=2),
        yaxis="y2"
    ))
    
    # 标记最优点
    fig.add_vline(x=best_ra, line_dash="dash", line_color="purple", 
                  annotation_text=f"Best: {best_ra:.2f}")
    
    fig.update_layout(
        title=f"Risk Aversion 优化曲线 (Top-{top_k})",
        xaxis_title="Risk Aversion (λ)",
        yaxis=dict(title="Sharpe Ratio", side="left", color="blue"),
        yaxis2=dict(title="Return / Drawdown (%)", side="right", overlaying="y", color="green"),
        hovermode="x unified",
        template="plotly_white",
        width=1000,
        height=600
    )
    
    # 保存图表
    fig_path = os.path.join(output_dir, f"risk_aversion_optimization_{version}.png")
    fig.write_image(fig_path)
    logger.info(f"\n优化曲线已保存: {fig_path}")
    
    # 保存结果表
    csv_path = os.path.join(output_dir, f"risk_aversion_results_{version}.csv")
    results_df.to_csv(csv_path, index=False)
    logger.info(f"详细结果已保存: {csv_path}")
    
    # 5. 推荐配置
    logger.info("\n" + "=" * 60)
    logger.info("=== 推荐配置 ===")
    logger.info("=" * 60)
    logger.info("请将以下配置更新到 config/main.yaml:")
    logger.info(f"""
model:
  dual_head:
    fusion:
      method: "utility"
      risk_aversion: {best_ra:.2f}
""")
    
    return best_ra, results_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="优化效用函数的风险厌恶系数")
    parser.add_argument("--min", type=float, default=0.1, help="搜索范围下限")
    parser.add_argument("--max", type=float, default=5.0, help="搜索范围上限")
    parser.add_argument("--n", type=int, default=50, help="网格点数")
    parser.add_argument("--target", type=str, default="sharpe", 
                        choices=["sharpe", "annual_return", "max_dd"],
                        help="优化目标")
    parser.add_argument("--top_k", type=int, default=5, help="每日选股数量")
    parser.add_argument("--version", type=str, default=None, help="指定模型版本")
    
    args = parser.parse_args()
    
    optimize_risk_aversion(
        ra_range=(args.min, args.max),
        n_points=args.n,
        target_metric=args.target,
        top_k=args.top_k,
        model_version=args.version
    )
