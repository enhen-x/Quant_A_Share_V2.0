# scripts/analisis/calc_cost_impact.py
# ============================================================================
# 资金规模对实际收益影响估算 (Cost Drag vs Capital Size)
# ============================================================================
#
# 用法示例:
# python scripts/analisis/calc_cost_impact.py --capitals 20000,100000,200000 \
#   --gross-daily-return 0.0018 --rebalance-days 4 --top-k 5 \
#   --min-commission 5 --commission-rate 0 --stamp-tax-rate 0.001
#
# 说明:
# - 成本分为固定成本(佣金保底)和比例成本(印花税/过户费等)
# - 仅估算“成本摊薄”对净收益率的影响，不包含滑点/冲击成本
# ============================================================================

import os
import sys
import argparse
import pandas as pd

# Matplotlib 字体配置
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

# 路径适配
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.config import GLOBAL_CONFIG
from src.utils.logger import get_logger
from src.utils.io import ensure_dir

logger = get_logger()

# ============================================================================
# 可修改的默认参数（不想每次传命令行就改这里）
# ============================================================================
DEFAULTS = {
    "capitals": "20000,50000,100000,200000",  # 本金列表（逗号分隔）
    "gross_daily_return": 0.0028,               # 预期/毛日均收益率（不扣费），例如 0.0018
    "rebalance_days": 4,                   # 调仓间隔天数（优先级最高）
    "rebalance_per_month": None,              # 每月调仓次数（与 rebalance_days 二选一）
    "trading_days": 252,                      # 年交易日数（用于年化）
    "turnover": 1.0,                          # 单次调仓单边换手比例（1.0 = 全换）
    "top_k": None,                            # 持仓数量（空则读取 config）
    "commission_rate": 0.0005,                   # 佣金比例（单边，十进制）
    "min_commission": 5.0,                    # 单笔佣金保底
    "trades_per_rebalance": None,             # 单次调仓交易笔数（默认=2*top_k）
    "fixed_commission": None,                 # 单次调仓固定佣金总额（若设置则忽略佣金比例/保底）
    "stamp_tax_rate": 0.0005,                    # 印花税比例（仅卖出，十进制）
    "transfer_fee_rate": 0.0,                 # 过户费比例（单边，十进制）
    "other_fee_rate": 0.0,                    # 其他比例费用（单边，十进制）
    "output_dir": "reports/cost_impact",      # 图片输出目录
    "plot_file": "capital_vs_net_daily_return.png",  # 图片文件名
}


def _parse_capitals(text: str):
    return [float(x) for x in text.split(",") if x.strip()]


def _infer_rebalance_days(args, config):
    if args.rebalance_days:
        return float(args.rebalance_days)
    if args.rebalance_per_month:
        return float(args.trading_days) / (float(args.rebalance_per_month) * 12.0)
    strategy_cfg = config.get("strategy", {})
    mode = strategy_cfg.get("rebalance_mode", "rolling")
    if mode == "periodic":
        return float(config.get("preprocessing", {}).get("labels", {}).get("horizon", 4))
    return 1.0


def _calc_commission_total(
    capital,
    turnover,
    trades_per_rebalance,
    commission_rate,
    min_commission,
    fixed_commission,
):
    if fixed_commission is not None:
        return float(fixed_commission)
    trades = max(1, int(trades_per_rebalance))
    total_trade_amount = capital * turnover * 2.0
    per_trade_amount = total_trade_amount / trades
    per_trade_commission = commission_rate * per_trade_amount
    if min_commission is not None:
        per_trade_commission = max(float(min_commission), per_trade_commission)
    return per_trade_commission * trades


def _estimate_annual_returns(
    gross_daily_return,
    daily_cost_rate,
    trading_days,
):
    gross_annual = (1.0 + gross_daily_return) ** trading_days - 1.0
    net_daily = gross_daily_return - daily_cost_rate
    net_annual = (1.0 + net_daily) ** trading_days - 1.0
    return gross_annual, net_annual


def _plot_net_daily_return(df: pd.DataFrame, output_dir: str, plot_file: str):
    if "net_daily_return" not in df.columns:
        logger.warning("net_daily_return not found; skip plot.")
        return

    if df.empty:
        logger.warning("result is empty; skip plot.")
        return

    plot_df = df.sort_values("capital")
    x = plot_df["capital"].values
    y = plot_df["net_daily_return"].values * 100.0

    ensure_dir(output_dir)
    out_path = os.path.join(output_dir, plot_file)

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o", linewidth=2, color="#2c7fb8")
    plt.xlabel("本金")
    plt.ylabel("净日均收益率 (%)")
    plt.title("本金 vs 净日均收益率")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"图片已保存: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Estimate cost drag vs capital size.")
    parser.add_argument("--capitals", type=str, default=None)
    parser.add_argument("--gross-daily-return", type=float, default=None)
    parser.add_argument("--rebalance-days", type=float, default=None)
    parser.add_argument("--rebalance-per-month", type=float, default=None)
    parser.add_argument("--trading-days", type=int, default=None)
    parser.add_argument("--turnover", type=float, default=None, help="one-way turnover per rebalance")
    parser.add_argument("--top-k", type=int, default=None)

    # Commission/fees
    parser.add_argument("--commission-rate", type=float, default=None, help="per-side rate, decimal")
    parser.add_argument("--min-commission", type=float, default=None, help="per-trade minimum")
    parser.add_argument("--trades-per-rebalance", type=int, default=None)
    parser.add_argument("--fixed-commission", type=float, default=None, help="override total commission per rebalance")
    parser.add_argument("--stamp-tax-rate", type=float, default=None, help="sell-only rate, decimal")
    parser.add_argument("--transfer-fee-rate", type=float, default=None, help="per-side rate, decimal")
    parser.add_argument("--other-fee-rate", type=float, default=None, help="per-side rate, decimal")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--plot-file", type=str, default=None)

    args = parser.parse_args()

    capitals_text = args.capitals if args.capitals is not None else DEFAULTS["capitals"]
    capitals = _parse_capitals(capitals_text)
    if not capitals:
        logger.error("capitals is empty.")
        return

    top_k = args.top_k if args.top_k is not None else DEFAULTS["top_k"]
    if top_k is None:
        top_k = int(GLOBAL_CONFIG.get("strategy", {}).get("top_k", 5))

    args.trading_days = args.trading_days if args.trading_days is not None else DEFAULTS["trading_days"]
    args.turnover = args.turnover if args.turnover is not None else DEFAULTS["turnover"]
    args.gross_daily_return = (
        args.gross_daily_return
        if args.gross_daily_return is not None
        else DEFAULTS["gross_daily_return"]
    )
    args.rebalance_days = (
        args.rebalance_days
        if args.rebalance_days is not None
        else DEFAULTS["rebalance_days"]
    )
    args.rebalance_per_month = (
        args.rebalance_per_month
        if args.rebalance_per_month is not None
        else DEFAULTS["rebalance_per_month"]
    )
    args.commission_rate = (
        args.commission_rate
        if args.commission_rate is not None
        else DEFAULTS["commission_rate"]
    )
    args.min_commission = (
        args.min_commission
        if args.min_commission is not None
        else DEFAULTS["min_commission"]
    )
    args.trades_per_rebalance = (
        args.trades_per_rebalance
        if args.trades_per_rebalance is not None
        else DEFAULTS["trades_per_rebalance"]
    )
    args.fixed_commission = (
        args.fixed_commission
        if args.fixed_commission is not None
        else DEFAULTS["fixed_commission"]
    )
    args.stamp_tax_rate = (
        args.stamp_tax_rate
        if args.stamp_tax_rate is not None
        else DEFAULTS["stamp_tax_rate"]
    )
    args.transfer_fee_rate = (
        args.transfer_fee_rate
        if args.transfer_fee_rate is not None
        else DEFAULTS["transfer_fee_rate"]
    )
    args.other_fee_rate = (
        args.other_fee_rate
        if args.other_fee_rate is not None
        else DEFAULTS["other_fee_rate"]
    )
    args.output_dir = (
        args.output_dir
        if args.output_dir is not None
        else DEFAULTS["output_dir"]
    )
    args.plot_file = (
        args.plot_file
        if args.plot_file is not None
        else DEFAULTS["plot_file"]
    )

    rebalance_days = _infer_rebalance_days(args, GLOBAL_CONFIG)
    rebalances_per_year = float(args.trading_days) / rebalance_days

    trades_per_rebalance = args.trades_per_rebalance
    if trades_per_rebalance is None:
        trades_per_rebalance = 2 * top_k

    rows = []
    for cap in capitals:
        commission_total = _calc_commission_total(
            capital=cap,
            turnover=args.turnover,
            trades_per_rebalance=trades_per_rebalance,
            commission_rate=args.commission_rate,
            min_commission=args.min_commission,
            fixed_commission=args.fixed_commission,
        )

        stamp_tax = cap * args.turnover * args.stamp_tax_rate
        transfer_fee = cap * args.turnover * 2.0 * args.transfer_fee_rate
        other_fee = cap * args.turnover * 2.0 * args.other_fee_rate

        cost_per_rebalance = commission_total + stamp_tax + transfer_fee + other_fee
        cost_rate_per_rebalance = cost_per_rebalance / cap
        daily_cost_rate = cost_rate_per_rebalance / rebalance_days

        row = {
            "capital": cap,
            "rebalance_days": rebalance_days,
            "rebalance_per_year": rebalances_per_year,
            "cost_per_rebalance": cost_per_rebalance,
            "cost_rate_per_rebalance": cost_rate_per_rebalance,
            "daily_cost_rate": daily_cost_rate,
        }

        if args.gross_daily_return is not None:
            gross_annual, net_annual = _estimate_annual_returns(
                args.gross_daily_return,
                daily_cost_rate,
                args.trading_days,
            )
            row.update(
                {
                    "gross_daily_return": args.gross_daily_return,
                    "net_daily_return": args.gross_daily_return - daily_cost_rate,
                    "gross_annual_return": gross_annual,
                    "net_annual_return": net_annual,
                    "annual_cost_drag": gross_annual - net_annual,
                }
            )

        rows.append(row)

    df = pd.DataFrame(rows)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    print("\n=== Cost Impact Summary ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    if args.gross_daily_return is None:
        print("\nNote: pass --gross-daily-return to estimate net return impact.")
    else:
        _plot_net_daily_return(df, args.output_dir, args.plot_file)


if __name__ == "__main__":
    main()
