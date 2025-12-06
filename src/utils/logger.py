# src/utils/logger.py
import logging
import os
import sys

def get_logger(name="quant", level=logging.INFO):
    logger = logging.getLogger(name)
    
    # 防止重复添加 Handler
    if logger.handlers:
        return logger
        
    logger.setLevel(level)
    
    # 格式
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 控制台输出
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # 文件输出 (可选)
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(log_dir, "quant.log"), encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger