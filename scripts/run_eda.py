# scripts/run_eda.py

import os
import sys
import argparse

# 路径适配
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.analysis.eda_engine import EDAEngine

def main():
    parser = argparse.ArgumentParser(description="量化数据探索性分析 (EDA)")
    parser.add_argument("--sample", "-s", type=int, default=None, help="采样股票数量")
    args = parser.parse_args()

    print(">>> 开始运行全维度数据分析...")
    
    try:
        engine = EDAEngine()
        # 这里调用 run_full_scan 会依次触发 5 项分析
        engine.run_full_scan(sample_size=args.sample)
        
    except Exception as e:
        print(f"运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()