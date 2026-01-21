# src/live/xueqiu_broker.py
"""
雪球券商接口封装
"""

import easytrader
from src.utils.logger import get_logger

logger = get_logger()

class XueqiuBroker:
    """雪球券商接口"""
    
    def __init__(self, cookies, portfolio_code, portfolio_market='cn'):
        """
        初始化雪球连接
        
        Args:
            cookies: 雪球 cookies
            portfolio_code: 组合代码
            portfolio_market: 交易市场 ('cn', 'us', 'hk')
        """
        self.cookies = cookies
        self.portfolio_code = portfolio_code
        self.portfolio_market = portfolio_market
        self.user = None
        self._connect()
    
    def _connect(self):
        """连接到雪球"""
        try:
            logger.info(f"正在连接雪球组合 {self.portfolio_code}...")
            self.user = easytrader.use('xq')
            self.user.prepare(
                cookies=self.cookies,
                portfolio_code=self.portfolio_code,
                portfolio_market=self.portfolio_market
            )
            logger.info("✅ 成功连接到雪球!")
        except Exception as e:
            logger.error(f"❌ 连接雪球失败: {e}")
            raise
    
    def get_balance(self):
        """
        获取账户余额信息
        
        Returns:
            dict: {
                'total_assets': 总资产,
                'available_cash': 可用资金,
                'market_value': 持仓市值
            }
        """
        try:
            balance_list = self.user.balance
            if not balance_list:
                return {'total_assets': 0, 'available_cash': 0, 'market_value': 0}
            
            balance = balance_list[0]  # 取第一个账户
            return {
                'total_assets': balance.get('asset_balance', 0),
                'available_cash': balance.get('enable_balance', 0),
                'market_value': balance.get('market_value', 0)
            }
        except Exception as e:
            logger.error(f"获取余额失败: {e}")
            return None
    
    def get_portfolio_net_value(self):
        """
        获取组合净值
        
        Returns:
            float: 净值（如1.05表示增长5%）
        """
        try:
            # 通过balance获取净值
            balance = self.get_balance()
            if balance:
                # 假设初始资金在配置中
                # 净值 = 当前总资产 / 初始资金
                # 这里先返回1.0，实际需要从雪球API获取
                return 1.0  # TODO: 从雪球API获取真实净值
            return 1.0
        except Exception as e:
            logger.error(f"获取净值失败: {e}")
            return 1.0
    
    def get_stock_price(self, symbol):
        """
        获取股票当前价格（从本地数据）
        
        Args:
            symbol: 股票代码（str或int）
        
        Returns:
            float: 收盘价格
        """
        try:
            import pandas as pd
            from pathlib import Path
            
            # 确保symbol为字符串并补齐6位
            symbol = str(symbol).zfill(6)  # 补齐前导0，如2949 -> 002949
            
            # 本地数据路径: data/raw/{symbol}.parquet
            project_root = Path(__file__).parent.parent.parent
            data_file = project_root / "data" / "raw" / f"{symbol}.parquet"
            
            if not data_file.exists():
                logger.warning(f"未找到本地股票数据文件: {data_file}")
                # 尝试查找其他可能的路径或文件名格式? 
                # 目前假设只有这一种格式
                return None
            
            # 读取Parquet数据
            df = pd.read_parquet(data_file)
            
            if not df.empty:
                # 假设数据按时间排序，取最后一行
                # 如果没有按时间排序，可能需要 sort_values('date')
                # 通常parquet数据是按时间追加的
                latest_data = df.iloc[-1]
                price = float(latest_data['close'])
                date = latest_data.get('date', 'unknown')
                logger.info(f"获取 {symbol} 收盘价: {price:.2f}元 (日期: {date}, 本地数据)")
                return price
            else:
                logger.warning(f"本地数据为空: {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"获取股票价格失败 {symbol}: {e}")
            # import traceback
            # traceback.print_exc()
            return None
    
    def get_positions(self):
        """
        获取当前持仓
        
        Returns:
            list: 持仓列表，每个元素为 {
                'symbol': 股票代码,
                'quantity': 持有数量,
                'cost_price': 成本价,
                'current_price': 当前价,
                'profit_loss': 盈亏
            }
        """
        try:
            positions = self.user.position
            if not positions:
                return []
            
            result = []
            for pos in positions:
                # 确保股票代码为6位数字字符串
                raw_code = str(pos.get('stock_code', '')).upper()
                # 移除 SZ/SH 前缀
                symbol = raw_code.replace('SZ', '').replace('SH', '')
                # 补齐前导0
                symbol = symbol.zfill(6)
                
                result.append({
                    'symbol': symbol,
                    'name': pos.get('stock_name', ''),
                    'quantity': pos.get('current_amount', 0),
                    'available_qty': pos.get('enable_amount', 0),
                    'cost_price': pos.get('avg_cost', 0),
                    'current_price': pos.get('current', 0),
                    'market_value': pos.get('market_value', 0),
                    'profit_loss': pos.get('income', 0),
                    'profit_loss_pct': pos.get('income_rate', 0)
                })
            return result
        except Exception as e:
            if "get portfolio info error" in str(e):
                logger.error(f"❌ 获取持仓失败: 无法获取组合信息。可能 Cookies 失效或组合代码错误。")
            else:
                logger.error(f"获取持仓失败: {e}")
            return []
    
    
    def adjust_weight(self, rebalance_stock_list):
        """
        调整组合权重（雪球推荐方式）
        
        Args:
            rebalance_stock_list: 调仓列表，格式:
                [
                    {'symbol': '603879', 'weight': 20.0},
                    {'symbol': '600493', 'weight': 20.0},
                    ...
                ]
        
        Returns:
            bool: 是否成功
        """
        try:
            logger.info(f"调整组合权重: {len(rebalance_stock_list)} 只股票")
            
            for stock in rebalance_stock_list:
                logger.info(f"  {stock['symbol']}: {stock['weight']:.2f}%")
            
            # easytrader adjust_weight 需要一次调整一个股票
            # 或者使用 rebalance 方法
            for stock in rebalance_stock_list:
                result = self.user.adjust_weight(stock['symbol'], stock['weight'])
                logger.info(f"  调仓 {stock['symbol']}: {result}")
            
            return True
            
        except Exception as e:
            if "stocks" in str(e) and isinstance(e, KeyError):
                logger.error("❌ 调仓失败: 无法获取股票信息。通常是因为雪球 Cookies 失效。")
                logger.error("👉 请更新配置文件中的 Config Cookies。")
            else:
                logger.error(f"调仓失败: {e}")
                import traceback
                traceback.print_exc()
            return False
    
    def buy(self, symbol, weight):
        """
        买入股票（通过调整权重）
        
        Args:
            symbol: 股票代码
            weight: 权重（百分比，如20.0表示20%）
        
        Returns:
            bool: 是否成功
        """
        # 获取当前权重
        current_holdings = self.get_positions()
        
        # 构建新的权重列表（保留现有+添加新的）
        new_holdings = []
        
        # 保留现有持仓（权重保持）
        for pos in current_holdings:
            new_holdings.append({
                'symbol': pos['symbol'],
                'weight': pos.get('weight', 0)  
            })
        
        # 添加新买入
        new_holdings.append({
            'symbol': symbol,
            'weight': weight
        })
        
        # 调用调仓
        return self.adjust_weight(new_holdings)
    
    def sell(self, symbol):
        """
        卖出股票（通过调整权重为0）
        
        Args:
            symbol: 股票代码
        
        Returns:
            bool: 是否成功
        """
        # 获取当前权重
        current_holdings = self.get_positions()
        
        # 构建新的权重列表（移除要卖出的股票）
        new_holdings = []
        
        for pos in current_holdings:
            if pos['symbol'] != symbol:
                # 保留其他股票
                new_holdings.append({
                    'symbol': pos['symbol'],
                    'weight': pos.get('weight', 0)
                })
        
        # 调用调仓（不包含要卖出的股票，等于权重为0）
        return self.adjust_weight(new_holdings)
    
    def _get_limit_up_price(self, symbol):
        """
        获取涨停价
        TODO: 实现涨停价计算逻辑
        """
        # 暂时返回一个较高的价格
        # 实际应该根据昨收价计算: 昨收价 * 1.10 (四舍五入到分)
        return 999.99
    
    def _get_limit_down_price(self, symbol):
        """
        获取跌停价
        TODO: 实现跌停价计算逻辑
        """
        # 暂时返回一个较低的价格
        # 实际应该根据昨收价计算: 昨收价 * 0.90 (四舍五入到分)
        return 0.01
    
    def get_current_price(self, symbol):
        """
        获取股票当前价格
        TODO: 实现价格查询
        """
        # 可以使用 akshare 或其他数据源获取实时价格
        return None
