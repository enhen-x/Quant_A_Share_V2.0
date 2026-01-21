# scripts/live/run_weekly_rebalance.py
"""
周期性全仓换股脚本

核心功能：
1. 读取 data/live_trading/config_week_change.txt 配置
2. 卖出所有现有持仓
3. 买入当日推荐的全部股票

适用场景：
- 每周定期换仓
- 每 N 天全仓换股的策略
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger
from src.live.xueqiu_broker import XueqiuBroker
from src.live.trade_recorder import TradeRecorder

logger = get_logger()


class WeeklyRebalanceConfig:
    """周期性换仓配置"""
    
    def __init__(self):
        self.config_file = project_root / 'data' / 'live_trading' / 'config_week_change.txt'
        self._config = {}
        self.load()
    
    def load(self):
        """加载配置"""
        if not self.config_file.exists():
            raise FileNotFoundError(
                f"配置文件不存在: {self.config_file}\n"
                f"请创建配置文件并填写雪球账号信息。"
            )
        
        with open(self.config_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    self._config[key.strip()] = value.strip()
    
    def get(self, key, default=None):
        return self._config.get(key, default)
    
    def get_int(self, key, default=0):
        value = self.get(key, default)
        return int(value) if value else default
    
    @property
    def cookies(self):
        return self.get('cookies')
    
    @property
    def portfolio_code(self):
        return self.get('portfolio_code')
    
    @property
    def portfolio_market(self):
        return self.get('portfolio_market', 'cn')
    
    @property
    def initial_capital(self):
        return self.get_int('initial_capital', 100000)
    
    @property
    def max_stocks(self):
        return self.get_int('max_stocks', 10)
    
    def validate(self):
        """验证必需配置"""
        required = {'cookies': '雪球 Cookies', 'portfolio_code': '组合代码'}
        missing = [name for key, name in required.items() if not self.get(key)]
        if missing:
            raise ValueError(f"配置文件中缺少必需项: {', '.join(missing)}")
    
    def show(self):
        """显示配置"""
        print("=" * 60)
        print("周期性换仓配置")
        print("=" * 60)
        print(f"配置文件: {self.config_file}")
        print()
        print("雪球配置:")
        print(f"  Cookies: {'已设置 [OK]' if self.cookies else '未设置 [MISSING]'}")
        print(f"  组合代码: {self.portfolio_code}")
        print(f"  交易市场: {self.portfolio_market}")
        print()
        print("交易配置:")
        print(f"  初始资金: {self.initial_capital:,} 元")
        print(f"  最大持股: {self.max_stocks} 只")
        print("=" * 60)


def load_today_picks(max_stocks=10):
    """加载今日推荐"""
    import pandas as pd
    
    daily_picks_dir = project_root / 'reports' / 'daily_picks'
    today = datetime.now().strftime('%Y%m%d')
    
    pattern = f"*{today}*.csv"
    matching_files = list(daily_picks_dir.glob(pattern))
    
    if not matching_files:
        logger.warning(f"未找到 {today} 的推荐文件")
        return None
    
    latest_file = max(matching_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"[OK] 找到推荐文件: {latest_file.name}")
    
    df = pd.read_csv(latest_file, dtype={'symbol': str})
    if 'symbol' in df.columns:
        df['symbol'] = df['symbol'].apply(lambda x: str(x).zfill(6))
    
    # 按 pred_score 取 top N
    if 'pred_score' in df.columns and len(df) > max_stocks:
        df = df.nlargest(max_stocks, 'pred_score')
    elif len(df) > max_stocks:
        df = df.head(max_stocks)
    
    logger.info(f"加载推荐: {len(df)} 只股票")
    return df


class CookieInvalidError(Exception):
    """Cookie 失效异常"""
    pass


def safe_adjust_weight(broker, symbol, weight):
    """安全调仓，捕获 Cookie 失效错误"""
    try:
        broker.user.adjust_weight(symbol, weight)
    except Exception as e:
        if "stocks" in str(e) and isinstance(e, KeyError):
            print(f"\n[ERROR] ❌ 调仓失败 ({symbol}): Cookie 已失效！")
            print("👉 请运行 python scripts/live/check_xq_cookie.py 检查并更新 Cookie")
            raise CookieInvalidError("CookieInvalid")
        else:
            raise e


def main(dry_run=False):
    """主函数"""
    print("=" * 70)
    print("周期性全仓换股系统")
    print("=" * 70)
    
    # 步骤 0：加载配置
    print("\n" + "=" * 70)
    print("步骤 0：加载配置文件")
    print("=" * 70)
    
    try:
        config = WeeklyRebalanceConfig()
        config.validate()
        config.show()
        print("\n[OK] 配置验证通过\n")
    except Exception as e:
        print(f"\n[ERROR] 配置加载失败: {e}")
        return
    
    # 确认操作
    if not dry_run:
        print("\n[WARNING]  真实模式：将执行以下操作：")
        print("   1. 卖出所有现有持仓")
        print("   2. 买入今日推荐股票")
        confirm = input("\n确认继续？(输入 yes 继续): ")
        if confirm.lower() != 'yes':
            print("已取消")
            return
    else:
        print("\n[INFO] 模拟模式：不会实际下单")
    
    # 步骤 1：连接雪球
    print("\n" + "=" * 70)
    print("步骤 1：连接雪球")
    print("=" * 70)
    
    broker = XueqiuBroker(
        cookies=config.cookies,
        portfolio_code=config.portfolio_code,
        portfolio_market=config.portfolio_market
    )
    
    # 获取当前持仓
    current_positions = broker.get_positions()
    logger.info(f"当前持仓: {len(current_positions)} 只股票")
    for pos in current_positions:
        logger.info(f"  {pos['symbol']} 权重: {pos.get('weight', 0):.2f}%")
    
    # 步骤 2：加载今日推荐
    print("\n" + "=" * 70)
    print("步骤 2：加载今日推荐")
    print("=" * 70)
    
    picks = load_today_picks(config.max_stocks)
    if picks is None or picks.empty:
        print("\n[ERROR] 未找到今日推荐，请先运行推荐脚本")
        return
    
    # 计算等权权重
    weight_per_stock = 100.0 / len(picks)
    logger.info(f"每只股票权重: {weight_per_stock:.2f}%")
    
    # 构建新的持仓列表
    new_holdings = []
    for _, row in picks.iterrows():
        symbol = row['symbol']
        new_holdings.append({
            'symbol': symbol,
            'weight': weight_per_stock
        })
        logger.info(f"  {symbol} -> {weight_per_stock:.2f}%")
    
    # 步骤 3：执行换仓（先买后卖，确保不空仓）
    print("\n" + "=" * 70)
    print("步骤 3：执行全仓换股（先买后卖）")
    print("=" * 70)
    
    # 计算需要卖出的股票（在旧持仓但不在新持仓中）
    current_symbols = {p['symbol'] for p in current_positions}
    new_symbols = {h['symbol'] for h in new_holdings}
    
    to_sell = current_symbols - new_symbols  # 需要卖出的
    to_buy = new_symbols - current_symbols   # 需要买入的
    to_keep = current_symbols & new_symbols  # 保持的（可能需要调整权重）
    
    logger.info(f"保持持仓: {len(to_keep)} 只")
    logger.info(f"新买入: {len(to_buy)} 只")
    logger.info(f"需卖出: {len(to_sell)} 只")
    
    if dry_run:
        print("\n[INFO] [模拟] 换仓计划:")
        if to_keep:
            print(f"   保持: {list(to_keep)}")
        if to_buy:
            print(f"   买入: {list(to_buy)}")
        if to_sell:
            print(f"   卖出: {list(to_sell)}")
        print("\n[INFO] [模拟] 模拟模式完成，未实际下单")
    else:
        try:
            # 先买后卖策略：
            # 1. 先买入新股票（给予较小权重，避免超过100%）
            # 2. 再卖出旧股票（设置权重为0）
            # 3. 最后调整所有新持仓到目标权重
            
            print("\n[BUY] 执行买入...")
            for symbol in to_buy:
                weight = weight_per_stock
                logger.info(f"  买入 {symbol} 权重: {weight:.2f}%")
                safe_adjust_weight(broker, symbol, weight)
            
            print("\n[SELL] 执行卖出...")
            for symbol in to_sell:
                logger.info(f"  卖出 {symbol} (权重 -> 0)")
                safe_adjust_weight(broker, symbol, 0)
            
            print("\n[ADJUST] 调整权重...")
            for holding in new_holdings:
                symbol = holding['symbol']
                weight = holding['weight']
                logger.info(f"  调整 {symbol} -> {weight:.2f}%")
                safe_adjust_weight(broker, symbol, weight)
            
            print("\n[OK] 全仓换股成功!")
            print(f"   新持仓: {len(new_holdings)} 只股票")

        except CookieInvalidError:
            print("\n[INFO] 程序因 Cookie 失效终止，请更新 Cookie 后重试。")
            
        except Exception as e:
            print(f"\n[ERROR] 换仓异常: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("执行完成！")
    print("=" * 70)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='周期性全仓换股脚本')
    parser.add_argument(
        '--sim',
        action='store_true',
        help='模拟模式（不实际下单）'
    )
    parser.add_argument(
        '--config-only',
        action='store_true',
        help='仅验证配置文件'
    )
    
    args = parser.parse_args()
    
    if args.config_only:
        try:
            config = WeeklyRebalanceConfig()
            config.validate()
            config.show()
            print("\n[OK] 配置有效")
        except Exception as e:
            print(f"\n[ERROR] 配置无效: {e}")
            sys.exit(1)
        sys.exit(0)
    
    main(dry_run=args.sim)
