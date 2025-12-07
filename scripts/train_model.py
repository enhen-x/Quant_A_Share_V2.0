# scripts/train_model.py
import sys
import os
# 路径适配... (同其他脚本)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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