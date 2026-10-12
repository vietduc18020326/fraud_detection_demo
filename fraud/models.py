"""Định nghĩa các mô hình: baseline và cải tiến."""
from __future__ import annotations

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Nhóm mô hình: dùng để tô màu/giải thích trong báo cáo và demo
MODEL_GROUPS = {
    "LogReg": "Baseline",
    "RandomForest": "Baseline",
    "LogReg+class_weight": "Cải tiến A (mất cân bằng)",
    "RF+SMOTE": "Cải tiến A (mất cân bằng)",
    "XGBoost": "Cải tiến B (boosting)",
    "IsolationForest": "Cải tiến C (phát hiện bất thường)",
}


class IsoScorer(BaseEstimator, ClassifierMixin):
    """Isolation Forest huấn luyện CHỈ trên giao dịch bình thường (bán giám sát).

    Điểm bất thường được chuẩn hoá về [0, 1] bằng phân vị của điểm trên tập huấn luyện,
    để có cùng giao diện predict_proba với các mô hình khác. Điểm này KHÔNG phải xác suất
    hiệu chỉnh nên ngưỡng 0,5 không có ý nghĩa; chỉ nên dùng ngưỡng chọn theo chi phí.
    """

    def __init__(self, n_estimators=200, random_state=42, n_jobs=-1):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.model_ = IsolationForest(
            n_estimators=self.n_estimators, random_state=self.random_state, n_jobs=self.n_jobs
        ).fit(X[y == 0])
        raw = -self.model_.score_samples(X[y == 0])
        self.ref_ = np.sort(raw)
        self.classes_ = np.array([0, 1])
        return self

    def _score(self, X):
        raw = -self.model_.score_samples(np.asarray(X))
        return np.searchsorted(self.ref_, raw, side="right") / len(self.ref_)

    def predict_proba(self, X):
        s = self._score(X)
        return np.column_stack([1 - s, s])

    def predict(self, X):
        return (self._score(X) >= 0.99).astype(int)


def build_models(y_train, seed: int = 42, fast: bool = False) -> dict:
    n_trees = 60 if fast else 150
    pos = int(np.sum(y_train))
    neg = len(y_train) - pos
    spw = neg / max(pos, 1)

    models = {
        "LogReg": Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))]),
        "RandomForest": RandomForestClassifier(
            n_estimators=n_trees, n_jobs=-1, random_state=seed, min_samples_leaf=1
        ),
        "LogReg+class_weight": Pipeline(
            [("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))]
        ),
        # SMOTE chỉ nằm trong pipeline huấn luyện => chỉ tác động lên tập train, không rò rỉ sang val/test
        "RF+SMOTE": ImbPipeline(
            [
                ("smote", SMOTE(sampling_strategy=0.1, random_state=seed)),
                ("clf", RandomForestClassifier(n_estimators=n_trees, n_jobs=-1, random_state=seed)),
            ]
        ),
        "IsolationForest": IsoScorer(n_estimators=100 if fast else 200, random_state=seed),
    }
    try:
        from xgboost import XGBClassifier

        models["XGBoost"] = XGBClassifier(
            n_estimators=150 if fast else 400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=spw,
            eval_metric="aucpr",
            tree_method="hist",
            n_jobs=-1,
            random_state=seed,
        )
    except ImportError:  # xgboost không cài => dùng boosting của sklearn
        from sklearn.ensemble import HistGradientBoostingClassifier

        models["XGBoost"] = HistGradientBoostingClassifier(
            class_weight="balanced", max_iter=200, random_state=seed
        )
    order = ["LogReg", "RandomForest", "LogReg+class_weight", "RF+SMOTE", "XGBoost", "IsolationForest"]
    return {k: models[k] for k in order}


def predict_score(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, 1]
