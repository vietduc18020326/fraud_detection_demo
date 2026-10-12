"""Huấn luyện và đánh giá các mô hình phát hiện gian lận thẻ tín dụng.

Ví dụ:
    python train.py --data data/creditcard.csv                # cả 2 cách chia (time, random)
    python train.py --data data/creditcard.csv --split time   # chỉ chia theo thời gian
    python train.py --data data/creditcard.csv --fast         # chạy nhanh để thử
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from fraud.common import (
    FEATURE_COLS,
    best_threshold,
    load_data,
    make_features,
    split_data,
    summarize,
    total_cost,
)
from fraud.models import MODEL_GROUPS, build_models, predict_score


def run(df: pd.DataFrame, mode: str, args) -> None:
    out = Path(args.out)
    (out / "models" / mode).mkdir(parents=True, exist_ok=True)

    train, val, test = split_data(df, mode=mode, seed=args.seed)
    Xtr, ytr = make_features(train), train["Class"].to_numpy()
    Xva, yva = make_features(val), val["Class"].to_numpy()
    Xte, yte = make_features(test), test["Class"].to_numpy()
    amt_va, amt_te = val["Amount"].to_numpy(), test["Amount"].to_numpy()

    print(f"\n=== Chia '{mode}' ===")
    for name, d in [("train", train), ("val", val), ("test", test)]:
        print(f"  {name:5s}: {len(d):7d} giao dịch, {int(d['Class'].sum()):4d} gian lận ({d['Class'].mean():.4%})")
    if yte.sum() == 0 or yva.sum() == 0:
        raise SystemExit("Tập val/test không có giao dịch gian lận nào; không thể đánh giá. Kiểm tra dữ liệu.")

    models = build_models(ytr, seed=args.seed, fast=args.fast)
    rows, curves = {}, {}
    test_out = test[["Time", "Amount", "Class", *[c for c in test.columns if c.startswith("V")]]].copy()

    for name, model in models.items():
        t0 = time.time()
        model.fit(Xtr, ytr)
        s_va, s_te = predict_score(model, Xva), predict_score(model, Xte)
        # Ngưỡng chọn trên tập VAL (không nhìn tập test)
        thr, _ = best_threshold(yva, s_va, amt_va, args.c_fp)
        row = summarize(yte, s_te, amt_te, thr, args.c_fp)
        row["Nhóm"] = MODEL_GROUPS[name]
        # Đối chiếu với ngưỡng mặc định 0,5 (không có nghĩa với IsolationForest)
        if name == "IsolationForest":
            row["Chi phí (ngưỡng 0.5)"] = np.nan
        else:
            row["Chi phí (ngưỡng 0.5)"] = total_cost(yte, s_te, amt_te, 0.5, args.c_fp)
        row["Thời gian huấn luyện (s)"] = time.time() - t0
        rows[name] = row
        test_out[f"score_{name}"] = s_te
        p, r, _ = precision_recall_curve(yte, s_te)
        curves[name] = (r, p, average_precision_score(yte, s_te))
        joblib.dump(model, out / "models" / mode / f"{name}.joblib")
        print(f"  {name:22s} PR-AUC={row['PR-AUC']:.4f}  Recall@P80={row['Recall@P80']:.3f}  ({row['Thời gian huấn luyện (s)']:.1f}s)")

    res = pd.DataFrame(rows).T
    cols = ["Nhóm"] + [c for c in res.columns if c != "Nhóm"]
    res = res[cols]
    res.index.name = "Mô hình"
    res.to_csv(out / f"results_{mode}.csv", float_format="%.6g")
    test_out.to_csv(out / f"test_scores_{mode}.csv.gz", index=False, float_format="%.6g")
    meta = {
        "split": mode,
        "c_fp": args.c_fp,
        "seed": args.seed,
        "features": FEATURE_COLS,
        "n_train": len(train), "n_val": len(val), "n_test": len(test),
        "fraud_train": int(ytr.sum()), "fraud_val": int(yva.sum()), "fraud_test": int(yte.sum()),
    }
    (out / f"meta_{mode}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (r, p, ap) in curves.items():
        ax.plot(r, p, label=f"{name} (AP={ap:.3f})")
    ax.axhline(yte.mean(), ls="--", c="gray", lw=0.8, label=f"Ngẫu nhiên ({yte.mean():.4f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title(f"Đường cong Precision-Recall (test, chia {mode})")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out / f"pr_curves_{mode}.png", dpi=150); plt.close(fig)

    show = res[["PR-AUC", "ROC-AUC", "Recall@P80", "Precision (ngưỡng chi phí)", "Recall (ngưỡng chi phí)", "Chi phí (ngưỡng chi phí)", "Chi phí (ngưỡng 0.5)"]]
    print("\n" + show.astype(float).round(4).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/creditcard.csv", help="đường dẫn creditcard.csv")
    ap.add_argument("--openml", action="store_true", help="nếu không có file, thử tải từ OpenML")
    ap.add_argument("--split", choices=["time", "random", "both"], default="both")
    ap.add_argument("--c-fp", type=float, default=10.0, help="chi phí giả định cho mỗi cảnh báo sai")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--fast", action="store_true", help="mô hình nhỏ hơn để chạy thử nhanh")
    args = ap.parse_args()

    df = load_data(args.data, allow_openml=args.openml)
    print(f"Đã nạp {len(df)} giao dịch, {int(df['Class'].sum())} gian lận ({df['Class'].mean():.4%}).")
    for mode in (["time", "random"] if args.split == "both" else [args.split]):
        run(df, mode, args)
    print(f"\nXong. Kết quả trong thư mục '{args.out}/'. Chạy demo: streamlit run app.py")


if __name__ == "__main__":
    main()
