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
        
        # 转换为 DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        evals = []
        if X_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals = [(dtrain, 'train'), (dval, 'eval')]
            
        # 1. 拷贝一份参数，并移除 'n_estimators'
        train_params = self.params.copy()
        num_rounds = train_params.pop('n_estimators', 1000) # 取出并移除
        
        # 开始训练
        self.model = xgb.train(
            # ================== 修改点 ==================
            params=train_params,  # 这里要传处理过(移除n_estimators)的 train_params，而不是 self.params
            # ===========================================
            dtrain=dtrain,
            num_boost_round=num_rounds, # 使用刚才 pop 出来的值
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