# scripts/tools/start_tensorboard.py
"""
启动 TensorBoard 可视化服务

使用方法:
    python scripts/tools/start_tensorboard.py

启动后访问: http://localhost:6006
"""

import os
import sys
import subprocess
import webbrowser
import time

# 路径适配
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.config import GLOBAL_CONFIG


def main():
    # 获取日志目录
    log_dir = os.path.join(GLOBAL_CONFIG["paths"]["logs"], "tensorboard")
    
    # 确保目录存在
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"[INFO] 创建日志目录: {log_dir}")
    
    print("=" * 60)
    print("  TensorBoard 训练监控可视化")
    print("=" * 60)
    print(f"\n📁 日志目录: {log_dir}")
    print(f"🌐 访问地址: http://localhost:6006")
    print(f"\n按 Ctrl+C 停止服务\n")
    print("=" * 60)
    
    # 启动 TensorBoard
    try:
        # 2 秒后自动打开浏览器
        def open_browser():
            time.sleep(2)
            webbrowser.open("http://localhost:6006")
        
        import threading
        browser_thread = threading.Thread(target=open_browser, daemon=True)
        browser_thread.start()
        
        # 运行 TensorBoard
        subprocess.run([
            sys.executable, "-m", "tensorboard.main",
            "--logdir", log_dir,
            "--host", "localhost",
            "--port", "6006"
        ])
    except KeyboardInterrupt:
        print("\n\n[INFO] TensorBoard 服务已停止")
    except Exception as e:
        print(f"\n[ERROR] 启动失败: {e}")
        print("\n尝试使用命令行启动:")
        print(f"  tensorboard --logdir={log_dir}")


if __name__ == "__main__":
    main()
