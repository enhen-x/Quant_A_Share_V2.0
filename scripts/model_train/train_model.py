# scripts/train_model.py
import sys
import os
# 路径适配... (同其他脚本)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 从当前文件位置 (scripts/analisis) 返回两级到项目根目录
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.model.trainer import ModelTrainer

def main():
    try:
        trainer = ModelTrainer()
        trainer.run()
    except Exception as e:
        print(f"训练失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()