# scripts/live/run_auto_trading.py
"""
自动交易主执行脚本（简洁版输出）
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger
from src.live.config import get_config
from src.live.trading_scheduler import TradingScheduler

logger = get_logger()


def load_and_validate_config():
    """加载并验证配置文件"""
    try:
        config = get_config()
        config.validate()
        return config
    except FileNotFoundError as e:
        print(f"\n❌ 配置文件不存在: {e}")
        print("   请创建配置文件: data/live_trading/config.txt")
        return None
    except ValueError as e:
        print(f"\n❌ 配置验证失败: {e}")
        return None
    except Exception as e:
        print(f"\n❌ 配置加载异常: {e}")
        return None


def main(dry_run=True):
    """
    主执行函数
    
    Args:
        dry_run: 是否为模拟模式（True=模拟，False=真实下单）
    """
    print("\n🚀 自动交易系统启动")
    
    # 加载配置
    config = load_and_validate_config()
    if config is None:
        return
    
    mode_str = "模拟" if dry_run else "真实"
    print(f"   模式: {mode_str} | 组合: {config.portfolio_code} | 资金: {config.initial_capital:,}元")
    
    if not dry_run:
        confirm = input("\n⚠️  确认真实下单？(输入 yes): ")
        if confirm.lower() != 'yes':
            print("已取消")
            return
    
    # 初始化调度器
    mode = 'sim' if dry_run else 'real'
    scheduler = TradingScheduler(mode=mode)
    
    # 1. 卖出检查
    sold = scheduler.check_and_sell(dry_run=dry_run)
    if sold:
        print(f"\n📤 卖出: {len(sold)} 只")
    
    # 2. 读取推荐
    picks = scheduler.get_today_picks()
    if picks is None:
        print("\n❌ 未找到推荐数据，请先运行 run_recommendation.py")
        return
    
    # 3. 买入流程
    buy_plan = scheduler.create_buy_plan(picks)
    if not buy_plan:
        print("\n❌ 无法生成买入计划")
        return
    
    filtered_plan = scheduler.filter_existing_holdings(buy_plan)
    
    if filtered_plan:
        success = scheduler.execute_buy(filtered_plan, dry_run=dry_run)
        if success:
            print(f"\n📥 买入: {len(success)}/{len(filtered_plan)} 只成功")
    else:
        print("\n⏭️  无新股票需买入")
    
    # 统计摘要
    summary = scheduler.recorder.get_summary()
    print(f"\n📊 持仓: {summary['current_holdings']} 只 | 累计盈亏: {summary['total_profit']:.2f}元")
    
    if summary['holding_symbols']:
        symbols = ', '.join([str(s) for s in summary['holding_symbols']])
        print(f"   股票: {symbols}")
    
    print("\n✅ 完成\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='自动交易执行脚本')
    parser.add_argument('--sim', action='store_true', help='模拟模式')
    parser.add_argument('--config-only', action='store_true', help='仅验证配置')
    
    args = parser.parse_args()
    
    if args.config_only:
        config = load_and_validate_config()
        if config:
            config.show()
            print("\n✅ 配置有效")
        sys.exit(0 if config else 1)
    
    main(dry_run=args.sim)
