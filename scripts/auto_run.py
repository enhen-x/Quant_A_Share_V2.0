# scripts/auto_run.py

import subprocess
import time
import os
import sys

# 解释器路径 (根据你的环境自动获取)
PYTHON_EXE = sys.executable
# 下载脚本路径
SCRIPT_PATH = os.path.join("scripts", "download_data.py")

def main():
    print(f"=== 开始全自动挂机下载 ===")
    print(f"策略：报错 -> 等待 5 分钟 -> 自动重启 (断点续传)")
    
    retry_count = 0
    
    while True:
        try:
            # 启动子进程运行 download_data.py，不加 --force 默认就是跳过已存在
            print(f"\n[第 {retry_count + 1} 次启动] 正在执行下载任务...")
            
            # 使用 subprocess.call 会等待脚本运行结束
            # 如果脚本因为报错退出了，这里会拿到返回码
            exit_code = subprocess.call([PYTHON_EXE, SCRIPT_PATH])
            
            if exit_code == 0:
                print(">>> 恭喜！所有数据下载完成！")
                break
            else:
                print(f">>> 脚本异常退出 (代码 {exit_code})")
                
        except KeyboardInterrupt:
            print("\n用户手动停止挂机。")
            break
        except Exception as e:
            print(f"发生系统错误: {e}")

        # 如果没运行完就退出了，说明被封了或者出错了
        # 冷却时间：建议设为 300秒 (5分钟) 或 600秒 (10分钟)
        sleep_seconds = 300 
        print(f"检测到异常退出，可能触发了反爬风控。")
        print(f"正在冷却 IP... 挂机等待 {sleep_seconds} 秒后自动重试...")
        
        for i in range(sleep_seconds, 0, -1):
            # 倒计时显示
            print(f"倒计时: {i} 秒...", end="\r")
            time.sleep(1)
        
        retry_count += 1
        print("\n" + "="*40)

if __name__ == "__main__":
    main()