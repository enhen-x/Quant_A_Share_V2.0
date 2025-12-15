# src/live/config.py
"""
实盘交易配置加载模块
从 data/live_trading/config.txt 加载配置
"""

import os
from pathlib import Path

class LiveTradingConfig:
    """实盘交易配置管理"""
    
    def __init__(self):
        # 配置文件路径
        project_root = Path(__file__).parent.parent.parent
        self.config_file = project_root / 'data' / 'live_trading' / 'config.txt'
        
        # 加载配置
        self._config = {}
        self.load()
    
    def load(self):
        """从文件加载配置"""
        if not self.config_file.exists():
            raise FileNotFoundError(
                f"配置文件不存在: {self.config_file}\n"
                f"请创建配置文件并填写雪球账号信息。"
            )
        
        with open(self.config_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                # 跳过注释和空行
                if not line or line.startswith('#'):
                    continue
                
                # 解析 key=value
                if '=' in line:
                    key, value = line.split('=', 1)
                    self._config[key.strip()] = value.strip()
    
    def get(self, key, default=None):
        """获取配置值"""
        return self._config.get(key, default)
    
    def get_int(self, key, default=0):
        """获取整数配置"""
        value = self.get(key, default)
        return int(value) if value else default
    
    # 雪球配置
    @property
    def cookies(self):
        return self.get('cookies')
    
    @property
    def portfolio_code(self):
        return self.get('portfolio_code')
    
    @property
    def portfolio_market(self):
        return self.get('portfolio_market', 'cn')
    
    # 交易配置
    @property
    def initial_capital(self):
        return self.get_int('initial_capital', 200000)
    
    @property
    def hold_days(self):
        return self.get_int('hold_days', 5)
    
    @property
    def min_order_amount(self):
        return self.get_int('min_order_amount', 1000)
    
    @property
    def max_stocks_per_day(self):
        """每天最多买入股票数量"""
        return self.get_int('max_stocks_per_day', 5)
    
    def validate(self):
        """验证必需配置是否存在"""
        required = {
            'cookies': '雪球 Cookies',
            'portfolio_code': '组合代码'
        }
        
        missing = []
        for key, name in required.items():
            if not self.get(key):
                missing.append(name)
        
        if missing:
            raise ValueError(f"配置文件中缺少必需项: {', '.join(missing)}")
    
    def show(self, hide_sensitive=True):
        """显示配置（隐藏敏感信息）"""
        print("=" * 60)
        print("实盘交易配置")
        print("=" * 60)
        print(f"配置文件: {self.config_file}")
        print()
        print("雪球配置:")
        if hide_sensitive:
            print(f"  Cookies: {'已设置 ✓' if self.cookies else '未设置 ✗'}")
        else:
            print(f"  Cookies: {self.cookies[:50]}...")
        print(f"  组合代码: {self.portfolio_code}")
        print(f"  交易市场: {self.portfolio_market}")
        print()
        print("交易配置:")
        print(f"  初始资金: {self.initial_capital:,} 元")
        print(f"  持有天数: {self.hold_days} 天")
        print(f"  最小下单金额: {self.min_order_amount} 元")
        print("=" * 60)


# 全局配置实例
_config_instance = None

def get_config():
    """获取配置实例（单例模式）"""
    global _config_instance
    if _config_instance is None:
        _config_instance = LiveTradingConfig()
    return _config_instance


if __name__ == '__main__':
    # 测试配置加载
    config = get_config()
    config.validate()
    config.show()
