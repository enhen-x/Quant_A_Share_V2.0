# src/live/trade_recorder.py
"""
本地交易记录管理器
记录买入/卖出信息，计算持有天数
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from src.utils.logger import get_logger
from src.utils.io import read_parquet

logger = get_logger()

class TradeRecorder:
    """交易记录管理器"""
    
    def __init__(self, records_file='data/live_trading/trade_records.csv'):
        """
        初始化
        
        Args:
            records_file: 交易记录文件路径
        """
        project_root = Path(__file__).parent.parent.parent
        self.records_file = project_root / records_file
        
        # 确保目录存在
        self.records_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 加载或创建记录
        self.records = self._load_records()
    
    def _load_records(self):
        """加载交易记录"""
        if self.records_file.exists():
            try:
                df = pd.read_csv(self.records_file, parse_dates=['date', 'plan_sell_date', 'actual_sell_date'])
                logger.info(f"加载交易记录: {len(df)} 条")
                return df
            except Exception as e:
                logger.warning(f"加载交易记录失败: {e}，创建新记录")
        
        # 创建空DataFrame
        return pd.DataFrame(columns=[
            'symbol', 'action', 'date', 'quantity', 'price', 'amount',
            'plan_sell_date', 'actual_sell_date', 'status', 'profit'
        ])
    
    def _save_records(self):
        """保存交易记录"""
        try:
            self.records.to_csv(self.records_file, index=False, encoding='utf-8-sig')
            logger.info(f"保存交易记录: {len(self.records)} 条")
        except Exception as e:
            logger.error(f"保存交易记录失败: {e}")
    
    def record_buy(self, symbol, quantity, price, buy_date, hold_days=5):
        """
        记录买入
        
        Args:
            symbol: 股票代码
            quantity: 数量
            price: 价格
            buy_date: 买入日期
            hold_days: 持有天数
        """
        # 计算计划卖出日期（考虑交易日历）
        plan_sell_date = self._calculate_sell_date(buy_date, hold_days)
        
        # 添加记录
        new_record = {
            'symbol': symbol,
            'action': 'buy',
            'date': buy_date,
            'quantity': quantity,
            'price': price,
            'amount': quantity * price,
            'plan_sell_date': plan_sell_date,
            'actual_sell_date': None,
            'status': 'holding',
            'profit': 0
        }
        
        new_record_df = pd.DataFrame([new_record])
        if not new_record_df.empty:
             # 确保类型一致以避免 FutureWarning
             new_record_df = new_record_df.astype(self.records.dtypes)
             self.records = pd.concat([self.records, new_record_df], ignore_index=True)
        self._save_records()
        
        logger.info(f"记录买入: {symbol} x {quantity}股 @ {price}元, 计划卖出日期: {plan_sell_date}")
    
    def record_sell(self, symbol, quantity, price, sell_date):
        """
        记录卖出
        
        Args:
            symbol: 股票代码
            quantity: 数量
            price: 价格
            sell_date: 卖出日期
        """
        # 查找对应的买入记录
        buy_records = self.records[
            (self.records['symbol'] == symbol) &
            (self.records['action'] == 'buy') &
            (self.records['status'] == 'holding')
        ]
        
        if buy_records.empty:
            logger.warning(f"未找到 {symbol} 的买入记录")
            return
        
        # 取最早的买入记录
        buy_record = buy_records.iloc[0]
        buy_price = buy_record['price']
        profit = (price - buy_price) * quantity
        
        # 更新买入记录状态
        idx = buy_record.name
        self.records.at[idx, 'actual_sell_date'] = sell_date
        self.records.at[idx, 'status'] = 'completed'
        self.records.at[idx, 'profit'] = profit
        
        # 添加卖出记录
        new_record = {
            'symbol': symbol,
            'action': 'sell',
            'date': sell_date,
            'quantity': quantity,
            'price': price,
            'amount': quantity * price,
            'plan_sell_date': None,
            'actual_sell_date': sell_date,
            'status': 'completed',
            'profit': profit
        }
        
        self.records = pd.concat([self.records, pd.DataFrame([new_record])], ignore_index=True)
        self._save_records()
        
        profit_pct = (profit / (buy_price * quantity)) * 100
        logger.info(f"记录卖出: {symbol} x {quantity}股 @ {price}元, 盈亏: {profit:.2f}元 ({profit_pct:.2f}%)")
    
    def get_holdings(self):
        """
        获取当前持仓（status=holding）
        
        Returns:
            DataFrame: 持仓记录
        """
        holdings = self.records[
            (self.records['action'] == 'buy') &
            (self.records['status'] == 'holding')
        ]
        return holdings.copy()
    
    def get_to_sell(self, current_date):
        """
        获取今日需要卖出的股票
        
        Args:
            current_date: 当前日期
        
        Returns:
            DataFrame: 待卖出记录
        """
        current_date = pd.to_datetime(current_date)
        
        to_sell = self.records[
            (self.records['action'] == 'buy') &
            (self.records['status'] == 'holding') &
            (self.records['plan_sell_date'] <= current_date)
        ]
        
        return to_sell.copy()
    
    def is_holding(self, symbol):
        """
        检查是否已持有某股票
        
        Args:
            symbol: 股票代码
        
        Returns:
            bool: 是否持有
        """
        holdings = self.get_holdings()
        return symbol in holdings['symbol'].values
    
    def _calculate_sell_date(self, buy_date, hold_days):
        """
        计算卖出日期（考虑交易日历）
        
        Args:
            buy_date: 买入日期
            hold_days: 持有天数
        
        Returns:
            datetime: 卖出日期
        """
        try:
            # 加载交易日历
            project_root = Path(__file__).parent.parent.parent
            calendar_file = project_root / 'data' / 'meta' / 'trade_calendar.parquet'
            
            if not calendar_file.exists():
                # 如果没有交易日历，简单加天数
                logger.warning("未找到交易日历，使用简单日期计算")
                return pd.to_datetime(buy_date) + timedelta(days=hold_days)
            
            # 读取交易日历
            calendar = read_parquet(calendar_file)
            calendar['date'] = pd.to_datetime(calendar['date'])
            
            # 筛选交易日 (该文件本身只包含交易日，或通过date列)
            if 'is_trading_day' in calendar.columns:
                trade_days = calendar[calendar['is_trading_day'] == 1]['date'].sort_values()
            else:
                # 如果只有date列，假设所有行都是交易日
                trade_days = calendar['date'].sort_values()
            
            # 找到买入日期后的第 hold_days 个交易日
            buy_date = pd.to_datetime(buy_date)
            future_days = trade_days[trade_days > buy_date]
            
            if len(future_days) >= hold_days:
                sell_date = future_days.iloc[hold_days - 1]
            else:
                # 如果交易日不够，使用简单加天数
                logger.warning(f"交易日历数据不足，使用简单日期计算")
                sell_date = buy_date + timedelta(days=hold_days)
            
            return sell_date
            
        except Exception as e:
            logger.error(f"计算卖出日期失败: {e}，使用简单日期计算")
            return pd.to_datetime(buy_date) + timedelta(days=hold_days)
    
    def get_summary(self):
        """
        获取交易统计摘要
        
        Returns:
            dict: 统计信息
        """
        completed = self.records[self.records['status'] == 'completed']
        holdings = self.get_holdings()
        
        total_trades = len(completed[completed['action'] == 'buy'])
        total_profit = completed[completed['action'] == 'sell']['profit'].sum()
        
        return {
            'total_trades': total_trades,
            'total_profit': total_profit,
            'current_holdings': len(holdings),
            'holding_symbols': holdings['symbol'].tolist() if not holdings.empty else []
        }
