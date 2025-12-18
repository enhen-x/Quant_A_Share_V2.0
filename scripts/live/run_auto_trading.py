# scripts/live/run_auto_trading.py
"""
自动交易主执行脚本

核心改进：
1. 在执行任何操作前验证配置文件
2. 显示配置信息供用户确认
3. 配置验证失败时拒绝执行
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
    """
    加载并验证配置文件
    
    Returns:
        LiveTradingConfig: 配置对象，如果验证失败则返回 None
    """
    print("=" * 70)
    print("步骤 0：加载配置文件")
    print("=" * 70)
    
    try:
        config = get_config()
        config.validate()
        config.show()
        print("\n✅ 配置验证通过\n")
        return config
    except FileNotFoundError as e:
        print(f"\n❌ 配置文件不存在:")
        print(f"   {e}")
        print("\n💡 请创建配置文件: data/live_trading/config.txt")
        print("   参考格式:")
        print("   cookies=your_xueqiu_cookies")
        print("   portfolio_code=ZH1234567")
        print("   initial_capital=100000")
        print("   hold_days=5")
        print("   max_stocks_per_day=10")
        return None
    except ValueError as e:
        print(f"\n❌ 配置验证失败:")
        print(f"   {e}")
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
    print("=" * 70)
    print("自动交易系统")
    print("=" * 70)
    
    # 步骤 0：加载并验证配置
    config = load_and_validate_config()
    if config is None:
        print("\n❌ 配置验证失败，程序终止")
        return
    
    if dry_run:
        print("\n🔸 模拟模式：不会实际下单到雪球")
    else:
        print("\n⚠️  真实模式：将实际下单到雪球模拟盘！")
        print(f"   组合代码: {config.portfolio_code}")
        print(f"   初始资金: {config.initial_capital:,} 元")
        print(f"   持有天数: {config.hold_days} 天")
        print(f"   每日买入: {config.max_stocks_per_day} 只")
        confirm = input("\n确认继续？(输入 yes 继续): ")
        if confirm.lower() != 'yes':
            print("已取消")
            return
    
    # 初始化调度器
    mode = 'sim' if dry_run else 'real'
    scheduler = TradingScheduler(mode=mode)
    
    # 步骤1：检查并执行卖出（优先）
    print("\n" + "=" * 70)
    print("步骤1：检查是否有需要卖出的股票")
    print("=" * 70)
    
    sold = scheduler.check_and_sell(dry_run=dry_run)
    
    if sold:
        print(f"\n✅ 成功卖出 {len(sold)} 只股票")
    else:
        print("\n✅ 当前无需卖出")
    
    # 步骤2：读取今日推荐
    print("\n" + "=" * 70)
    print("步骤2：读取今日推荐")
    print("=" * 70)
    
    picks = scheduler.get_today_picks()
    
    if picks is None:
        print("\n❌ 未找到推荐数据")
        print("请运行: python scripts/back_test/run_recommendation.py")
        return
    
    print(f"\n✅ 成功读取推荐，共 {len(picks)} 只股票")
    
    # 步骤3：生成买入计划（等权分配）
    print("\n" + "=" * 70)
    print("步骤3：生成买入计划（等权分配）")
    print("=" * 70)
    
    buy_plan = scheduler.create_buy_plan(picks)
    
    if not buy_plan:
        print("\n❌ 无法生成买入计划")
        return
    
    # 步骤4：去重检查
    print("\n" + "=" * 70)
    print("步骤4：去重检查")
    print("=" * 70)
    
    filtered_plan = scheduler.filter_existing_holdings(buy_plan)
    
    if not filtered_plan:
        print("\n⚠️  所有股票已持有，无需买入")
    else:
        # 步骤5：执行买入
        print("\n" + "=" * 70)
        print("步骤5：执行买入")
        print("=" * 70)
        
        success = scheduler.execute_buy(filtered_plan, dry_run=dry_run)
        
        if success:
            print(f"\n✅ 买入完成: {len(success)}/{len(filtered_plan)} 只成功")
        else:
            print("\n❌ 买入失败")
    
    # 显示统计
    print("\n" + "=" * 70)
    print("交易统计")
    print("=" * 70)
    
    summary = scheduler.recorder.get_summary()
    print(f"\n总交易次数: {summary['total_trades']}")
    print(f"总盈亏: {summary['total_profit']:.2f} 元")
    print(f"当前持仓数量: {summary['current_holdings']}")
    if summary['holding_symbols']:
        holding_symbols_str = [str(s) for s in summary['holding_symbols']]
        print(f"持仓股票: {', '.join(holding_symbols_str)}")
    
    print("\n" + "=" * 70)
    print("执行完成！")
    print("=" * 70)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='自动交易执行脚本')
    parser.add_argument(
        '--sim', 
        action='store_true', 
        help='模拟模式（不实际下单）'
    )
    parser.add_argument(
        '--config-only',
        action='store_true',
        help='仅验证配置文件，不执行交易'
    )
    
    args = parser.parse_args()
    
    # 仅验证配置模式
    if args.config_only:
        config = load_and_validate_config()
        if config:
            print("配置文件有效，可以进行交易。")
        sys.exit(0 if config else 1)
    
    # 默认真实模式，--sim 参数切换到模拟模式
    dry_run = args.sim
    
    main(dry_run=dry_run)

