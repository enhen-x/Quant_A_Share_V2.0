# src/utils/io.py
import yaml
import os
import pandas as pd

def load_yaml(path: str):
    """加载 YAML 配置文件"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_yaml(data: dict, path: str):
    """保存 YAML"""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True)

def ensure_dir(path: str):
    """确保目录存在"""
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def read_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)

def save_parquet(df: pd.DataFrame, path: str):
    ensure_dir(os.path.dirname(path))
    df.to_parquet(path, index=False)

def save_csv(df: pd.DataFrame, path: str):
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False)