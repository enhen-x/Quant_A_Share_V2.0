

import os
import sys
# 获取当前脚本所在绝对路径 (G:\理财\Quant_A_Share_V2.0\scripts)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (G:\理财\Quant_A_Share_V2.0)
project_root = os.path.dirname(current_dir)
# 将根目录加入 Python 搜索路径
if project_root not in sys.path:
    sys.path.append(project_root)




# scripts/debug_source.py
from src.data_source.akshare_source import AkShareSource
from src.utils.config import GLOBAL_CONFIG

def main():
    print("=== 配置测试 ===")
    print("数据源配置:", GLOBAL_CONFIG.get("data"))
    
    source = AkShareSource()
    
    print("\n=== 1. 测试股票列表获取 ===")
    df_list = source.get_stock_list()
    print(df_list.head())
    
    if not df_list.empty:
        symbol = df_list.iloc[0]["symbol"]
        print(f"\n=== 2. 测试获取 {symbol} 行情 ===")
        df_price = source.get_price(symbol, "2023-01-01", "2023-02-01")
        print(df_price.head())

if __name__ == "__main__":
    main()