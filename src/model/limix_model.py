import os
import sys
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from src.utils.config import GLOBAL_CONFIG
from src.utils.logger import get_logger

logger = get_logger()


class LimiXModelWrapper:
    """
    Lightweight wrapper around LimiX inference to match the project's model API.
    Note: LimiX is inference-only and requires train data as context at predict time.
    """

    _ALLOWED_CFG_KEYS = {
        "limix_repo_path",
        "model_path",
        "inference_config",
        "device",
        "mix_precision",
        "outlier_remove_std",
        "softmax_temperature",
        "seed",
        "inference_with_ddp",
        "categorical_features_indices",
        "persist_training_data",
        "max_train_samples",
        "sample_seed",
        "predict_batch_size",
    }

    def __init__(self, task_type: str = "regression", custom_params: Optional[dict] = None):
        self.task_type = task_type
        self.conf = GLOBAL_CONFIG.get("model", {})
        base_cfg = dict(self.conf.get("limix", {}))
        if custom_params:
            base_cfg.update(custom_params)
        self.limix_cfg = self._filter_cfg(base_cfg)

        self.project_root = GLOBAL_CONFIG.get("paths", {}).get("project_root", ".")
        self._apply_cfg()

        self.predictor = None
        self.X_train = None
        self.y_train = None
        self.feature_names = None

    def _filter_cfg(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        filtered = {k: v for k, v in cfg.items() if k in self._ALLOWED_CFG_KEYS}
        unknown = set(cfg.keys()) - set(filtered.keys())
        if unknown:
            logger.debug(f"LimiX config ignored keys: {sorted(unknown)}")
        return filtered

    def _resolve_path(self, path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        if os.path.isabs(path):
            return path
        return os.path.normpath(os.path.join(self.project_root, path))

    def _apply_cfg(self):
        self.limix_repo_path = self._resolve_path(self.limix_cfg.get("limix_repo_path", "external/LimiX"))
        self.model_path = self._resolve_path(self.limix_cfg.get("model_path"))
        self.inference_config = self._resolve_path(self.limix_cfg.get("inference_config"))
        self.device = self.limix_cfg.get("device", "cuda")
        self.mix_precision = self.limix_cfg.get("mix_precision", True)
        self.outlier_remove_std = self.limix_cfg.get("outlier_remove_std", 12)
        self.softmax_temperature = self.limix_cfg.get("softmax_temperature", 0.9)
        self.seed = self.limix_cfg.get("seed", 0)
        self.inference_with_ddp = self.limix_cfg.get("inference_with_ddp", False)
        self.categorical_features_indices = self.limix_cfg.get("categorical_features_indices", None)
        self.persist_training_data = self.limix_cfg.get("persist_training_data", False)
        self.max_train_samples = self.limix_cfg.get("max_train_samples", None)
        self.sample_seed = self.limix_cfg.get("sample_seed", 0)
        self.predict_batch_size = self.limix_cfg.get("predict_batch_size", None)

    def _ensure_predictor(self):
        if self.predictor is not None:
            return
        if not self.limix_repo_path or not os.path.isdir(self.limix_repo_path):
            raise FileNotFoundError(f"LimiX repo not found at: {self.limix_repo_path}")
        if not self.model_path or not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                f"LimiX model checkpoint not found at: {self.model_path}. "
                f"Please download a ckpt file and update model.limix.model_path."
            )
        if not self.inference_config:
            raise ValueError("LimiX inference_config is required. Set model.limix.inference_config.")

        if self.limix_repo_path not in sys.path:
            sys.path.insert(0, self.limix_repo_path)

        try:
            import torch
            from inference.predictor import LimiXPredictor
        except Exception as exc:
            raise ImportError(
                "Failed to import LimiX dependencies. Install required packages and retry."
            ) from exc

        self.predictor = LimiXPredictor(
            device=torch.device(self.device),
            model_path=self.model_path,
            inference_config=self.inference_config,
            mix_precision=self.mix_precision,
            outlier_remove_std=self.outlier_remove_std,
            softmax_temperature=self.softmax_temperature,
            categorical_features_indices=self.categorical_features_indices,
            inference_with_DDP=self.inference_with_ddp,
            seed=self.seed,
        )

    def _to_numpy(self, data, dtype=np.float32):
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            return data.to_numpy(dtype=dtype)
        return np.asarray(data, dtype=dtype)

    def _maybe_subsample(self, X_train: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.max_train_samples:
            return X_train, y_train
        if X_train.shape[0] <= self.max_train_samples:
            return X_train, y_train
        rng = np.random.default_rng(self.sample_seed)
        idx = rng.choice(X_train.shape[0], size=self.max_train_samples, replace=False)
        logger.warning(
            f"LimiX training data subsampled from {X_train.shape[0]} to {len(idx)} rows."
        )
        return X_train[idx], y_train[idx]

    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        self._ensure_predictor()

        if hasattr(X_train, "columns"):
            self.feature_names = list(X_train.columns)

        X_np = self._to_numpy(X_train)
        y_np = self._to_numpy(y_train).reshape(-1)

        X_np, y_np = self._maybe_subsample(X_np, y_np)

        self.X_train = X_np
        self.y_train = y_np

        logger.info("LimiX is inference-only; training data cached for prediction.")

    def predict(self, X):
        if self.predictor is None or self.X_train is None or self.y_train is None:
            raise ValueError("LimiX model is not initialized. Call train() before predict().")

        if self.feature_names and hasattr(X, "columns"):
            missing = [f for f in self.feature_names if f not in X.columns]
            if missing:
                raise ValueError(f"Input data missing required features: {missing}")
            X = X[self.feature_names]

        X_np = self._to_numpy(X)

        task_type = "Regression" if self.task_type == "regression" else "Classification"
        batch_size = self.predict_batch_size
        if batch_size and X_np.shape[0] > batch_size:
            total = X_np.shape[0]
            num_batches = (total + batch_size - 1) // batch_size
            logger.info(f"LimiX batch prediction: total={total}, batch_size={batch_size}, batches={num_batches}")
            preds_list = []
            for i in range(num_batches):
                start = i * batch_size
                end = min(start + batch_size, total)
                batch = X_np[start:end]
                batch_preds = self.predictor.predict(
                    self.X_train, self.y_train, batch, task_type=task_type
                )
                if hasattr(batch_preds, "detach"):
                    batch_preds = batch_preds.detach().cpu().numpy()
                preds_list.append(np.asarray(batch_preds))
                if (i + 1) % 10 == 0 or i == num_batches - 1:
                    logger.info(f"LimiX batch prediction progress: {i + 1}/{num_batches}")
            return np.concatenate(preds_list, axis=0)

        preds = self.predictor.predict(self.X_train, self.y_train, X_np, task_type=task_type)
        if hasattr(preds, "detach"):
            preds = preds.detach().cpu().numpy()
        return np.asarray(preds)

    def save(self, path: str):
        data = {
            "task_type": self.task_type,
            "limix_cfg": self.limix_cfg,
            "feature_names": self.feature_names,
        }
        if self.persist_training_data:
            data["X_train"] = self.X_train
            data["y_train"] = self.y_train
        import joblib

        joblib.dump(data, path)
        logger.info(f"LimiX metadata saved to {path}")

    def load(self, path: str):
        import joblib

        data = joblib.load(path)
        self.task_type = data.get("task_type", self.task_type)
        self.limix_cfg = data.get("limix_cfg", self.limix_cfg)
        self._apply_cfg()
        self.feature_names = data.get("feature_names")
        self.X_train = data.get("X_train")
        self.y_train = data.get("y_train")
        self.predictor = None
        logger.info(f"LimiX metadata loaded from {path}")
