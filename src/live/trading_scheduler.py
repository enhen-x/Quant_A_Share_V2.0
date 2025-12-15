# src/live/trading_scheduler.py
"""
交易调度器 - 自动交易的核心控制模块
"""

import os
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.logger import get_logger
from src.live.config import get_config
from src.live.xueqiu_broker import XueqiuBroker
from src.live.trade_recorder import TradeRecorder

logger = get_logger()

class TradingScheduler:
    """交易调度器"""
    
    def __init__(self, mode='real'):
        """
        初始化
        
        Args:
            mode: 运行模式 'real' (实盘) 或 'sim' (模拟)
        """
        # 加载配置
        self.config = get_config()
        self.config.validate()
        
        self.mode = mode
        
        # 根据模式选择交易记录文件
        if self.mode == 'sim':
            record_file = 'data/live_trading/trade_records_sim.csv'
            logger.info("🔸 运行在模拟模式 (使用 trade_records_sim.csv)")
        else:
            record_file = 'data/live_trading/trade_records.csv'
            logger.info("🚀 运行在实盘模式 (使用 trade_records.csv)")
            
        # 初始化组件
        self.broker = None
        self.recorder = TradeRecorder(records_file=record_file)
        
        # 项目路径
        self.project_root = Path(__file__).parent.parent.parent
        self.daily_picks_dir = self.project_root / 'reports' / 'daily_picks'
    
    def connect_broker(self):
        """连接券商（雪球）"""
        if self.broker is None:
            self.broker = XueqiuBroker(
                cookies=self.config.cookies,
                portfolio_code=self.config.portfolio_code,
                portfolio_market=self.config.portfolio_market
            )
        return self.broker
    
    def check_daily_picks(self, date=None):
        """
        检查指定日期是否有推荐数据
        
        Args:
            date: 日期字符串 (YYYY-MM-DD) 或 datetime，None表示今天
        
        Returns:
            str: 推荐文件路径，如果没有则返回 None
        """
        if date is None:
            date = datetime.now().strftime('%Y%m%d')
        elif isinstance(date, datetime):
            date = date.strftime('%Y%m%d')
        else:
            # 移除日期中的分隔符
            date = date.replace('-', '').replace('/', '')
        
        logger.info(f"检查 {date} 的推荐数据...")
        
        # 检查目录是否存在
        if not self.daily_picks_dir.exists():
            logger.warning(f"推荐目录不存在: {self.daily_picks_dir}")
            return None
        
        # 查找包含日期的文件
        # 文件名格式: picks_WF_20251215_104447_20251215.csv 或 picks_20251215_*.csv
        pattern = f"*{date}*.csv"
        matching_files = list(self.daily_picks_dir.glob(pattern))
        
        if not matching_files:
            logger.warning(f"未找到 {date} 的推荐文件")
            logger.info(f"查找模式: {pattern}")
            logger.info(f"查找目录: {self.daily_picks_dir}")
            return None
        
        # 如果有多个文件，取最新的
        latest_file = max(matching_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"✅ 找到推荐文件: {latest_file.name}")
        
        return str(latest_file)
    
    def load_daily_picks(self, file_path):
        """
        加载推荐数据
        
        Args:
            file_path: 推荐文件路径
        
        Returns:
            DataFrame: 推荐列表 (columns: symbol, pred_score, ...)
        """
        try:
            # 强制指定 symbol列 为字符串，防止前导0丢失
            df = pd.read_csv(file_path, dtype={'symbol': str})
            
            # 确保 symbol 为6位
            if 'symbol' in df.columns:
                df['symbol'] = df['symbol'].apply(lambda x: str(x).zfill(6))
                
            logger.info(f"加载推荐数据: {len(df)} 只股票")
            
            # 显示推荐列表
            if 'symbol' in df.columns and 'pred_score' in df.columns:
                logger.info("推荐股票列表:")
                for idx, row in df.iterrows():
                    symbol = row['symbol']
                    score = row.get('pred_score', 0)
                    name = row.get('name', '')
                    logger.info(f"  {idx+1}. {symbol} {name} (得分: {score:.4f})")
            
            return df
            
        except Exception as e:
            logger.error(f"加载推荐文件失败: {e}")
            return None
    
    def get_today_picks(self, max_stocks=None):
        """
        获取今日推荐
        
        Args:
            max_stocks: 最多返回多少只股票，None则使用配置中的值
        
        Returns:
            DataFrame: 推荐列表，如果没有则返回 None
        """
        # 检查是否有推荐文件
        file_path = self.check_daily_picks()
        
        if file_path is None:
            logger.warning("⚠️  今日无推荐数据")
            logger.info("💡 请运行以下命令生成推荐:")
            logger.info("   python scripts/back_test/run_recommendation.py")
            return None
        
        # 加载推荐
        picks = self.load_daily_picks(file_path)
        
        if picks is None or picks.empty:
            return None
        
        # 限制股票数量（取预测分数最高的前N只）
        if max_stocks is None:
            max_stocks = self.config.max_stocks_per_day
        
        if len(picks) > max_stocks:
            # 按pred_score降序排序，取前N只
            if 'pred_score' in picks.columns:
                picks = picks.nlargest(max_stocks, 'pred_score')
                logger.info(f"✂️ 限制买入数量: 从 {len(self.load_daily_picks(file_path))} 只中选择前 {max_stocks} 只")
            else:
                picks = picks.head(max_stocks)
                logger.info(f"✂️ 限制买入数量: 取前 {max_stocks} 只")
        
        return picks
    
    def calculate_daily_budget(self):
        """
        计算每日可用预算（滚动周期策略）
        
        滚动周期说明：
        - 总资金被分成 hold_days 份
        - 每天使用 total_capital / hold_days
        - 例如：20万总资金，持有5天，每天用4万
        
        Returns:
            float: 每日预算
        """
        total_capital = self.config.initial_capital
        hold_days = self.config.hold_days
        
        daily_budget = total_capital / hold_days
        
        logger.info(f"资金配置 (滚动周期策略):")
        logger.info(f"  总资金: {total_capital:,.0f} 元")
        logger.info(f"  持有天数: {hold_days} 天")
        logger.info(f"  每日预算: {daily_budget:,.0f} 元")
        
        return daily_budget
    
    def create_buy_plan(self, picks):
        """
        创建买入计划（精确计算金额、手数和权重）
        
        Args:
            picks: DataFrame, 推荐列表
        
        Returns:
            list: 买入计划列表，每个元素为 {
                'symbol': 股票代码,
                'name': 股票名称,
                'price': 股票价格,
                'quantity': 买入手数,
                'amount': 买入金额,
                'weight': 权重（百分比）
            }
        """
        if picks is None or picks.empty:
            logger.warning("推荐列表为空，无法生成买入计划")
            return []
        
        # 连接券商获取净值
        self.connect_broker()
        
        # 1. 获取当前账户总资产
        balance = self.broker.get_balance()
        if balance and 'total_assets' in balance:
            total_assets = balance['total_assets']
            logger.info(f"当前总资产: {total_assets:,.0f}元")
        else:
            # 获取失败的回退逻辑
            logger.warning("无法获取账户余额，使用配置的初始资金")
            total_assets = self.config.initial_capital
            
        # 2. 滚动周期：每天用1/hold_days 资金
        daily_capital = total_assets / self.config.hold_days
        logger.info(f"持有周期: {self.config.hold_days}天")
        logger.info(f"每日预算: {daily_capital:,.0f}元")
        
        # 4. 等权分配
        n_stocks = len(picks)
        capital_per_stock = daily_capital / n_stocks
        
        logger.info(f"\n等权分配计划（精确计算）:")
        logger.info(f"  推荐数量: {n_stocks} 只")
        logger.info(f"  每只分配: {capital_per_stock:,.0f} 元")
        
        buy_plan = []
        total_weight = 0
        
        for idx, row in picks.iterrows():
            symbol = row['symbol']
            name = row.get('name', '')
            
            # 获取实时股价
            price = self.broker.get_stock_price(symbol)
            
            if price is None or price <= 0:
                logger.warning(f"  {symbol} {name}: 无法获取价格，跳过")
                continue
            
            # 计算手数（100股为1手）
            quantity = int(capital_per_stock / price / 100) * 100
            
            if quantity < 100:
                logger.warning(f"  {symbol} {name}: 预算不足1手（需{price*100:.0f}元），跳过")
                continue
            
            # 实际金额
            amount = quantity * price
            
            # 反推权重 = 实际金额 / 实际资产 * 100
            weight = (amount / total_assets) * 100
            total_weight += weight
            
            buy_plan.append({
                'symbol': symbol,
                'name': name,
                'price': price,
                'quantity': quantity,
                'amount': amount,
                'weight': weight
            })
            
            logger.info(f"  {symbol} {name}:")
            logger.info(f"    价格: {price:.2f}元")
            logger.info(f"    手数: {int(quantity/100)}手 = {quantity}股")
            logger.info(f"    金额: {amount:,.0f}元")
            logger.info(f"    权重: {weight:.2f}%")
        
        logger.info(f"\n生成买入计划: {len(buy_plan)} 只股票")
        logger.info(f"总权重: {total_weight:.2f}%")
        logger.info(f"总金额: {sum(p['amount'] for p in buy_plan):,.0f}元")
        
        return buy_plan
    
    def filter_existing_holdings(self, buy_plan):
        """
        过滤已持有的股票（去重检查）
        
        Args:
            buy_plan: list, 买入计划
        
        Returns:
            list: 过滤后的买入计划
        """
        if not buy_plan:
            return []
        
        # 1. 获取本地持仓
        local_holdings_df = self.recorder.get_holdings()
        local_symbols = set(local_holdings_df['symbol'].tolist()) if not local_holdings_df.empty else set()
        
        # 2. 获取雪球真实持仓
        broker_symbols = set()
        try:
            # 确保已连接
            if self.broker is None:
                self.connect_broker()
            
            positions = self.broker.get_positions()
            broker_symbols = {p['symbol'] for p in positions}
            if broker_symbols:
                logger.info(f"雪球真实持仓: {', '.join(broker_symbols)}")
        except Exception as e:
            logger.warning(f"获取雪球持仓失败，仅使用本地记录: {e}")
            
        # 合并持仓 (确保都是字符串且6位)
        all_holdings = {str(s).zfill(6) for s in local_symbols.union(broker_symbols)}
        
        if not all_holdings:
            logger.info("当前无持仓，无需去重")
            return buy_plan
            
        logger.info(f"当前持仓(合并): {', '.join(all_holdings)}")
        
        # 过滤
        filtered = []
        skipped = []
        
        for plan in buy_plan:
            symbol = str(plan['symbol']).zfill(6)
            if symbol in all_holdings:
                skipped.append(symbol)
                logger.warning(f"  跳过 {symbol} (已持有)")
            else:
                filtered.append(plan)
        
        if skipped:
            logger.info(f"去重检查: 跳过 {len(skipped)} 只已持有股票")
        
        return filtered
    
    def execute_buy(self, buy_plan, dry_run=True):
        """
        执行买入操作（使用权重调仓）
        
        Args:
            buy_plan: list, 买入计划（包含权重）
            dry_run: bool, 是否为模拟模式
        
        Returns:
            list: 成功买入的股票列表
        """
        if not buy_plan:
            logger.warning("买入计划为空，无需执行")
            return []
        
        if dry_run:
            logger.warning("🔸 模拟模式：不会实际下单")
            # 模拟模式：记录到本地
            buy_date = datetime.now()
            for plan in buy_plan:
                self.recorder.record_buy(
                    symbol=plan['symbol'],
                    quantity=0,  # 模拟模式不记录数量
                    price=0,     # 模拟模式不记录价格  
                    buy_date=buy_date,
                    hold_days=self.config.hold_days
                )
            logger.info(f"🔸 [模拟] 已记录 {len(buy_plan)} 只股票")
            return buy_plan
        
        # 真实模式：连接券商
        self.connect_broker()
        
        logger.info(f"\n开始执行买入 ({len(buy_plan)} 只股票):") 
        logger.info("=" * 60)
        
        # 准备调仓列表
        rebalance_list = []
        for plan in buy_plan:
            rebalance_list.append({
                'symbol': plan['symbol'],
                'weight': plan['weight']
            })
        
        try:
            # 一次性调仓（雪球推荐方式）
            success = self.broker.adjust_weight(rebalance_list)
            
            if success:
                logger.info(f"\n✅ 调仓成功!")
                
                # 记录到本地（权重模式下不记录具体数量和价格）
                buy_date = datetime.now()
                for plan in buy_plan:
                    self.recorder.record_buy(
                        symbol=plan['symbol'],
                        quantity=plan['quantity'],  # 记录计划买入数量
                        price=plan['price'],        # 记录参考价格
                        buy_date=buy_date,
                        hold_days=self.config.hold_days
                    )
                
                return buy_plan
            else:
                logger.error(f"\n❌ 调仓失败")
                return []
                
        except Exception as e:
            logger.error(f"\n❌ 调仓异常: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def check_and_sell(self, dry_run=True):
        """
        检查并执行卖出操作（权重调仓方式）
        
        Args:
            dry_run: bool, 是否为模拟模式
        
        Returns:
            list: 成功卖出的股票列表
        """
        current_date = datetime.now()
        
        # 获取需要卖出的股票
        to_sell = self.recorder.get_to_sell(current_date)
        
        if to_sell.empty:
            logger.info("✅ 当前无需卖出的股票")
            return []
        
        logger.info(f"\n检测到 {len(to_sell)} 只股票需要卖出:")
        for idx, row in to_sell.iterrows():
            logger.info(f"  {row['symbol']} (买入日期: {row['date']}, 计划卖出: {row['plan_sell_date']})")
        
        if dry_run:
            logger.warning("🔸 模拟模式：不会实际下单")
            # 模拟模式
            sell_date = current_date
            for idx, row in to_sell.iterrows():
                self.recorder.record_sell(
                    symbol=row['symbol'],
                    quantity=0,
                    price=0,
                    sell_date=sell_date
                )
            logger.info(f"🔸 [模拟] 已记录卖出 {len(to_sell)} 只股票")
            return to_sell['symbol'].tolist()
        
        # 真实模式：连接券商
        self.connect_broker()
        
        logger.info(f"\n开始执行卖出:")
        logger.info("=" * 60)
        
        success_list = []
        sell_date = current_date
        
        # 获取当前持仓
        current_positions = self.broker.get_positions()
        
        # 构建新的权重列表（移除要卖出的股票）
        new_holdings = []
        for pos in current_positions:
            if pos['symbol'] not in to_sell['symbol'].values:
                new_holdings.append({
                    'symbol': pos['symbol'],
                    'weight': pos.get('weight', 0)
                })
        
        try:
            # 调仓（移除卖出的股票）
            if new_holdings or len(current_positions) > len(to_sell):
                success = self.broker.adjust_weight(new_holdings)
            else:
                # 如果要全部卖出，调仓为空列表
                success = self.broker.adjust_weight([])
            
            if success:
                logger.info(f"\n✅ 卖出调仓成功!")
                
                # 记录卖出
                for idx, row in to_sell.iterrows():
                    self.recorder.record_sell(
                        symbol=row['symbol'],
                        quantity=0,
                        price=0,
                        sell_date=sell_date
                    )
                    success_list.append(row['symbol'])
                
                return success_list
            else:
                logger.error(f"\n❌ 卖出调仓失败")
                return []
                
        except Exception as e:
            logger.error(f"\n❌ 卖出异常: {e}")
            import traceback
            traceback.print_exc()
            return []


if __name__ == '__main__':
    # 测试完整交易流程
    print("=" * 70)
    print("测试：完整自动交易流程（模拟模式）")
    print("=" * 70)
    
    scheduler = TradingScheduler()
    
    # 步骤1：检查并执行卖出（优先）
    print("\n" + "=" * 70)
    print("步骤1：检查是否有需要卖出的股票")
    print("=" * 70)
    
    sold = scheduler.check_and_sell(dry_run=True)
    
    # 步骤2：读取推荐
    print("\n" + "=" * 70)
    print("步骤2：读取今日推荐")
    print("=" * 70)
    
    picks = scheduler.get_today_picks()
    
    if picks is None:
        print("\n❌ 未找到推荐数据，测试终止")
        exit(1)
    
    print(f"\n✅ 成功读取推荐，共 {len(picks)} 只股票")
    
    # 步骤3：计算预算并生成买入计划
    print("\n" + "=" * 70)
    print("步骤3：生成买入计划")
    print("=" * 70)
    
    daily_budget = scheduler.calculate_daily_budget()
    buy_plan = scheduler.create_buy_plan(picks, daily_budget)
    
    if not buy_plan:
        print("\n❌ 无法生成买入计划")
        exit(1)
    
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
        print("步骤5：执行买入（模拟模式）")
        print("=" * 70)
        
        success = scheduler.execute_buy(filtered_plan, dry_run=True)
        
        print(f"\n✅ 买入操作完成")
    
    # 显示统计
    print("\n" + "=" * 70)
    print("交易统计")
    print("=" * 70)
    
    summary = scheduler.recorder.get_summary()
    print(f"\n总交易次数: {summary['total_trades']}")
    print(f"总盈亏: {summary['total_profit']:.2f} 元")
    print(f"当前持仓数量: {summary['current_holdings']}")
    if summary['holding_symbols']:
        # 转换为字符串
        holding_symbols_str = [str(s) for s in summary['holding_symbols']]
        print(f"持仓股票: {', '.join(holding_symbols_str)}")
    
    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)
