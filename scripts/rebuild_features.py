# scripts/rebuild_features.py

import os
import sys

# 路径适配
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.preprocessing.pipeline import PreprocessPipeline

def main():
    try:
        pipeline = PreprocessPipeline()
        pipeline.run()
    except Exception as e:
        print(f"执行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()