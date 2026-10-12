"""Sinh dữ liệu GIẢ LẬP cùng định dạng với creditcard.csv, CHỈ để thử code chạy được.

Không dùng dữ liệu này cho báo cáo: kết quả trên đó không có ý nghĩa khoa học.
    python make_synthetic.py --out data/synthetic.csv --n 40000
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic.csv")
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--fraud-rate", type=float, default=0.004)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    X, y = make_classification(
        n_samples=a.n, n_features=28, n_informative=10, n_redundant=4, n_clusters_per_class=3,
        weights=[1 - a.fraud_rate, a.fraud_rate], class_sep=1.2, flip_y=0.001, random_state=a.seed,
    )
    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", np.sort(rng.uniform(0, 172800, a.n)).round(0))
    df["Amount"] = np.where(y == 1, rng.lognormal(4.5, 1.4, a.n), rng.lognormal(3.4, 1.3, a.n)).round(2)
    df["Class"] = y
    df.to_csv(a.out, index=False)
    print(f"Đã ghi {a.out}: {len(df)} dòng, {int(y.sum())} gian lận.")


if __name__ == "__main__":
    main()
