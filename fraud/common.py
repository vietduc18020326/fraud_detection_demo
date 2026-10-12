"""Hàm dùng chung: nạp dữ liệu, tạo đặc trưng, chia dữ liệu, metric, chi phí."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

V_COLS = [f"V{i}" for i in range(1, 29)]
REQUIRED_COLS = ["Time", *V_COLS, "Amount", "Class"]
FEATURE_COLS = [*V_COLS, "log_amount", "hour_sin", "hour_cos"]

OPENML_ID = 1597  # bản "creditcard" trên OpenML (chỉ là phương án dự phòng, cần kiểm tra cột)


# --------------------------------------------------------------------------- dữ liệu
def load_data(path: str | None = "data/creditcard.csv", allow_openml: bool = False) -> pd.DataFrame:
    """Đọc creditcard.csv (Kaggle, ULB/Worldline). Nếu không có file và allow_openml=True thì thử OpenML."""
    if path and Path(path).exists():
        df = pd.read_csv(path)
    elif allow_openml:
        from sklearn.datasets import fetch_openml

        raw = fetch_openml(data_id=OPENML_ID, as_frame=True, parser="auto").frame
        df = raw.rename(columns={c: c.strip("'\"") for c in raw.columns})
    else:
        raise FileNotFoundError(
            f"Không thấy '{path}'. Tải creditcard.csv từ Kaggle (Credit Card Fraud Detection, ULB) "
            "và đặt vào thư mục data/, hoặc chạy với --openml để thử tải từ OpenML."
        )
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Dữ liệu thiếu cột: {missing}. Cần các cột {REQUIRED_COLS}.")
    df = df[REQUIRED_COLS].copy()
    df["Class"] = df["Class"].astype(str).str.strip("'\"").astype(float).astype(int)
    return df.sort_values("Time", kind="stable").reset_index(drop=True)


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo đặc trưng không cần fit (nên dùng được cho dữ liệu người dùng tải lên).

    - log_amount: log(1 + Amount) để giảm lệch phân bố.
    - hour_sin/hour_cos: giờ trong ngày (từ Time) dạng tuần hoàn.
    Không đưa Time thô vào mô hình vì khi chia theo thời gian, giá trị Time của tập test nằm ngoài miền huấn luyện.
    """
    out = df[V_COLS].copy()
    out["log_amount"] = np.log1p(df["Amount"].clip(lower=0))
    hour = (df["Time"] // 3600) % 24
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return out[FEATURE_COLS]


# --------------------------------------------------------------------------- chia dữ liệu
def split_data(df: pd.DataFrame, mode: str = "time", test_frac: float = 0.2, val_frac: float = 0.15, seed: int = 42):
    """Trả về (train, val, test).

    mode="time"  : sắp xếp theo Time; phần đầu = train, kế tiếp = val, cuối cùng = test.
    mode="random": chia ngẫu nhiên phân tầng theo Class.
    """
    if mode == "time":
        n = len(df)
        n_test = int(n * test_frac)
        n_val = int(n * val_frac)
        n_train = n - n_test - n_val
        return df.iloc[:n_train], df.iloc[n_train : n_train + n_val], df.iloc[n_train + n_val :]
    if mode == "random":
        from sklearn.model_selection import train_test_split

        train_val, test = train_test_split(df, test_size=test_frac, stratify=df["Class"], random_state=seed)
        rel_val = val_frac / (1 - test_frac)
        train, val = train_test_split(train_val, test_size=rel_val, stratify=train_val["Class"], random_state=seed)
        return train, val, test
    raise ValueError("mode phải là 'time' hoặc 'random'")


# --------------------------------------------------------------------------- metric
def confusion(y, score, thr):
    pred = score >= thr
    y = np.asarray(y).astype(bool)
    tp = int(np.sum(pred & y))
    fp = int(np.sum(pred & ~y))
    fn = int(np.sum(~pred & y))
    tn = int(np.sum(~pred & ~y))
    return tp, fp, fn, tn


def prf(y, score, thr):
    tp, fp, fn, tn = confusion(y, score, thr)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def recall_at_precision(y, score, target=0.8):
    """Recall lớn nhất đạt được khi Precision >= target (0 nếu không đạt)."""
    p, r, _ = precision_recall_curve(y, score)
    ok = p >= target
    return float(r[ok].max()) if ok.any() else 0.0


def total_cost(y, score, amount, thr, c_fp):
    """Chi phí giả định: bỏ sót gian lận mất đúng số tiền giao dịch; mỗi cảnh báo sai tốn c_fp."""
    y = np.asarray(y).astype(bool)
    amount = np.asarray(amount, dtype=float)
    pred = np.asarray(score) >= thr
    missed = float(amount[(~pred) & y].sum())
    false_alarms = int(((pred) & (~y)).sum())
    return missed + c_fp * false_alarms


def cost_curve(y, score, amount, c_fp, n_points=400):
    """Chi phí theo ngưỡng. Ngưỡng lấy theo phân vị của điểm số để phủ đều vùng quan trọng."""
    score = np.asarray(score)
    qs = np.unique(np.quantile(score, np.linspace(0.0, 1.0, n_points)))
    thr = np.unique(np.concatenate([[0.0], qs, [score.max() + 1e-9]]))
    costs = np.array([total_cost(y, score, amount, t, c_fp) for t in thr])
    return thr, costs


def best_threshold(y, score, amount, c_fp):
    thr, costs = cost_curve(y, score, amount, c_fp)
    i = int(np.argmin(costs))
    return float(thr[i]), float(costs[i])


def summarize(y, score, amount, thr_cost, c_fp):
    """Bảng số liệu của một mô hình trên một tập dữ liệu."""
    p_c, r_c, f_c = prf(y, score, thr_cost)
    return {
        "PR-AUC": average_precision_score(y, score),
        "ROC-AUC": roc_auc_score(y, score),
        "Recall@P80": recall_at_precision(y, score, 0.8),
        "Precision (ngưỡng chi phí)": p_c,
        "Recall (ngưỡng chi phí)": r_c,
        "F1 (ngưỡng chi phí)": f_c,
        "Ngưỡng chi phí": thr_cost,
        "Chi phí (ngưỡng chi phí)": total_cost(y, score, amount, thr_cost, c_fp),
        "Chi phí (không cảnh báo gì)": float(np.asarray(amount, dtype=float)[np.asarray(y).astype(bool)].sum()),
    }
