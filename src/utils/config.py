# src/utils/config.py
import os
from src.utils.io import load_yaml

class ConfigLoader:
    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
            cls._instance._load()
        return cls._instance

    def _load(self):
        # 默认路径
        config_path = os.path.join("config", "main.yaml")
        
        # 加载配置
        try:
            self._config = load_yaml(config_path)
            print(f"成功加载配置文件: {config_path}")
        except Exception as e:
            # 如果加载失败，给一个空的默认结构防止报错，或者直接抛出
            print(f"警告: 无法加载配置文件 {config_path}: {e}")
            self._config = {}

    @property
    def data(self):
        return self._config

# 全局单例，其他文件直接引用这个变量
GLOBAL_CONFIG = ConfigLoader().data