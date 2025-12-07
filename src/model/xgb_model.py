import xgboost as xgb
from src.utils.config import GLOBAL_CONFIG
from src.utils.logger import get_logger

logger = get_logger()

class XGBModelWrapper:
    def __init__(self):
        self.conf = GLOBAL_CONFIG["model"]
        self.params = self.conf["params"]
        self.model = None

    def train(self, X_train, y_train, X_val=None, y_val=None):
        logger.info(f"初始化 XGBoost (Params: {self.params})")
        
        # 转换为 DMatrix (XGBoost 专有格式，速度更快)
        dtrain = xgb.DMatrix(X_train, label=y_train)
        evals = []
        if X_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals = [(dtrain, 'train'), (dval, 'eval')]
        train_params = self.params.copy()
        train_params.pop('n_estimators', None)  # 移除 sklearn 风格参数
        # 开始训练
        self.model = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=self.params.get("n_estimators", 1000),
            evals=evals,
            early_stopping_rounds=50,
            verbose_eval=100
        )
        logger.info("模型训练完成。")

    def predict(self, X):
        if self.model is None:
            raise ValueError("模型尚未训练")
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
    
    def save(self, path):
        self.model.save_model(path)

    def load(self, path):
        self.model = xgb.Booster()
        self.model.load_model(path)