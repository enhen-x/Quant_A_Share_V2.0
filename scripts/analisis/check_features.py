# scripts/check_features.py

import os
import sys

# 路径适配
current_dir = os.path.dirname(os.path.abspath(__file__))
# 从当前文件位置 (scripts/analisis) 返回两级到项目根目录
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.analysis.factor_checker import FactorChecker

def main():
    print(">>> 开始对 Processed 数据进行特征/标签有效性验证...")
    try:
        checker = FactorChecker()
        checker.run()
    except Exception as e:
        print(f"运行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()