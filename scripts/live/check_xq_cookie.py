import sys
from pathlib import Path
import os

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.live.xueqiu_broker import XueqiuBroker
from src.utils.logger import get_logger

logger = get_logger()

def check_cookies():
    print("="*60)
    print("雪球 Cookie 有效性检查")
    print("="*60)

    # Load config
    config_file = project_root / 'data' / 'live_trading' / 'config_week_change.txt'
    if not config_file.exists():
        logger.error(f"配置文件不存在: {config_file}")
        return

    config = {}
    with open(config_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                k, v = line.split('=', 1)
                config[k.strip()] = v.strip()

    cookies = config.get('cookies', '')
    if not cookies:
        logger.error("配置文件中未找到 cookies")
        return
    
    print(f"当前配置文件: {config_file}")
    print(f"Cookie 长度: {len(cookies)}")
    
    try:
        broker = XueqiuBroker(
            cookies=cookies,
            portfolio_code=config.get('portfolio_code'),
            portfolio_market=config.get('portfolio_market', 'cn')
        )
        
        print("\n正在尝试获取持仓...")
        positions = broker.get_positions()
        
        if not positions and broker.user:
            # 再次确认是否真的是空持仓，还是获取失败
            # 如果 get_positions 返回空，且没有报错日志，说明可能是真的是空仓
            # 但如果日志里有错误，说明失败。
            # 由于 get_positions 捕获了异常，我们检查日志输出（但在脚本里很难）
            # 所以我们手动调用一下更底层的
            pass

        # 尝试一个搜索请求来验证
        print("正在尝试搜索股票 (验证 API 权限)...")
        try:
             # Search for Moutai
            res = broker.user._search_stock_info('600519')
            if res:
                print("\n✅ Cookie 有效！")
                print(f"测试搜索结果: {res.get('name')} ({res.get('code')})")
                
                if positions:
                    print(f"当前持仓: {len(positions)} 只股票")
                else:
                    print("当前持仓为空 (或获取失败，请查看日志)")
            else:
                print("\n❌ Cookie 可能失效 (搜索返回空)")
        except Exception as e:
            if "stocks" in str(e) and isinstance(e, KeyError):
                print("\n❌ Cookie 已失效！")
                print("原因: API 返回数据缺失，通常表示未登录。")
            else:
                print(f"\n❌ 验证失败: {e}")

    except Exception as e:
        print(f"\n❌ 初始化失败: {e}")

    print("\n如何更新 Cookie:")
    print("1. 在浏览器打开 xueqiu.com 并登录")
    print("2. 按 F12 打开开发者工具 -> Network")
    print("3. 刷新页面，点击任意请求 (如 xueqiu.com)")
    print("4. 复制 Request Headers 中的 Cookie 值")
    print("5. 更新 config_week_change.txt")

if __name__ == "__main__":
    check_cookies()
