"""Demo tương tác: phát hiện gian lận thẻ tín dụng.

Chạy:  streamlit run app.py
Cần chạy `python train.py ...` trước để có thư mục outputs/.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import average_precision_score, precision_recall_curve

from fraud.common import (
    REQUIRED_COLS,
    best_threshold,
    confusion,
    cost_curve,
    make_features,
    prf,
    total_cost,
)
from fraud.models import MODEL_GROUPS, predict_score

OUT = Path(os.environ.get("FRAUD_OUT", "outputs"))

st.set_page_config(page_title="Phát hiện gian lận thẻ tín dụng", layout="wide")
st.title("Phát hiện gian lận thẻ tín dụng")
st.caption("Demo cho bài tập nhóm Nhận dạng mẫu. Kéo thanh trượt ngưỡng để xem đánh đổi giữa bắt thêm gian lận và gây thêm cảnh báo sai.")

splits = [m for m in ("time", "random") if (OUT / f"test_scores_{m}.csv.gz").exists()]
if not splits:
    st.error(f"Chưa có kết quả trong '{OUT}/'. Hãy chạy `python train.py --data data/creditcard.csv` trước.")
    st.stop()


@st.cache_data(show_spinner=False)
def load_split(mode: str):
    df = pd.read_csv(OUT / f"test_scores_{mode}.csv.gz")
    res = pd.read_csv(OUT / f"results_{mode}.csv", index_col=0)
    meta = json.loads((OUT / f"meta_{mode}.json").read_text())
    return df, res, meta


@st.cache_resource(show_spinner=False)
def load_model(mode: str, name: str):
    return joblib.load(OUT / "models" / mode / f"{name}.joblib")


# ------------------------------------------------------------------ thanh bên
with st.sidebar:
    st.header("Thiết lập")
    mode = st.radio(
        "Cách chia dữ liệu",
        splits,
        format_func=lambda m: {"time": "Theo thời gian (thực tế hơn)", "random": "Ngẫu nhiên phân tầng"}[m],
    )
    df, res, meta = load_split(mode)
    names = [c.replace("score_", "") for c in df.columns if c.startswith("score_")]
    model_name = st.selectbox("Mô hình", names, index=names.index("XGBoost") if "XGBoost" in names else 0)
    st.caption(f"Nhóm: {MODEL_GROUPS.get(model_name, '')}")
    c_fp = st.slider("Chi phí mỗi cảnh báo sai (đơn vị tiền)", 1, 200, int(meta.get("c_fp", 10)),
                     help="Giả định của nhóm: xác minh một giao dịch bị nghi ngờ tốn chi phí này. Bỏ sót gian lận mất đúng số tiền giao dịch.")
    st.divider()
    st.markdown(
        f"**Tập test ({mode})**: {meta['n_test']:,} giao dịch, **{meta['fraud_test']}** gian lận "
        f"({meta['fraud_test'] / meta['n_test']:.3%})."
    )
    if meta["fraud_test"] < 100:
        st.warning("Tập test có rất ít giao dịch gian lận, số liệu dao động mạnh giữa các lần chia.")

y = df["Class"].to_numpy()
amount = df["Amount"].to_numpy()
score = df[f"score_{model_name}"].to_numpy()

opt_thr, opt_cost = best_threshold(y, score, amount, c_fp)
is_iso = model_name == "IsolationForest"

# ------------------------------------------------------------------ tabs
tab1, tab2, tab3, tab4 = st.tabs(["Ngưỡng và chi phí", "So sánh mô hình", "Giao dịch nghi ngờ", "Tải CSV để chấm điểm"])

# ---------------- Tab 1
with tab1:
    lo, hi = float(score.min()), float(score.max())
    default_thr = float(np.clip(opt_thr, lo, hi))
    if st.button("Đặt về ngưỡng tối ưu chi phí"):
        st.session_state["thr"] = default_thr
    thr = st.slider("Ngưỡng quyết định (cảnh báo khi điểm ≥ ngưỡng)", lo, hi,
                    st.session_state.get("thr", default_thr), step=(hi - lo) / 1000 or 0.001, format="%.4f", key="thr")
    if is_iso:
        st.info("Điểm của Isolation Forest là điểm bất thường chuẩn hoá, không phải xác suất; ngưỡng 0,5 không có ý nghĩa.")

    tp, fp, fn, tn = confusion(y, score, thr)
    p, r, f1 = prf(y, score, thr)
    cost = total_cost(y, score, amount, thr, c_fp)
    no_alarm = float(amount[y == 1].sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Precision", f"{p:.3f}")
    c2.metric("Recall", f"{r:.3f}")
    c3.metric("F1", f"{f1:.3f}")
    c4.metric("Số cảnh báo", f"{tp + fp:,}", help="Số giao dịch bị đánh dấu nghi ngờ")
    c5.metric("Tổng chi phí", f"{cost:,.0f}", delta=f"{cost - no_alarm:,.0f} so với không cảnh báo gì", delta_color="inverse")

    left, right = st.columns(2)
    with left:
        st.markdown("**Ma trận nhầm lẫn**")
        cm = pd.DataFrame(
            [[tn, fp], [fn, tp]],
            index=["Thực tế: bình thường", "Thực tế: gian lận"],
            columns=["Dự đoán: bình thường", "Dự đoán: gian lận"],
        )
        st.dataframe(cm)
        st.caption(f"Bắt được {tp}/{tp + fn} giao dịch gian lận; {fp:,} cảnh báo sai trên {tn + fp:,} giao dịch bình thường.")

        pr_p, pr_r, _ = precision_recall_curve(y, score)
        fig, ax = plt.subplots(figsize=(5, 3.6))
        ax.plot(pr_r, pr_p, label=f"AP = {average_precision_score(y, score):.3f}")
        ax.scatter([r], [p], c="red", zorder=5, label="Ngưỡng hiện tại")
        ax.axhline(y.mean(), ls="--", c="gray", lw=0.8)
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Đường cong Precision-Recall"); ax.legend(); ax.grid(alpha=0.3)
        st.pyplot(fig); plt.close(fig)
    with right:
        thr_grid, costs = cost_curve(y, score, amount, c_fp)
        fig, ax = plt.subplots(figsize=(5, 3.6))
        ax.plot(thr_grid, costs)
        ax.axhline(no_alarm, ls="--", c="gray", lw=0.8, label="Không cảnh báo gì")
        ax.scatter([opt_thr], [opt_cost], c="green", zorder=5, label=f"Tối ưu (ngưỡng {opt_thr:.3f})")
        ax.scatter([thr], [cost], c="red", zorder=5, label="Ngưỡng hiện tại")
        ax.set_xlabel("Ngưỡng"); ax.set_ylabel("Tổng chi phí"); ax.set_title("Chi phí theo ngưỡng"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        st.pyplot(fig); plt.close(fig)
        st.caption("Chi phí = tổng số tiền các giao dịch gian lận bị bỏ sót + chi phí mỗi cảnh báo sai × số cảnh báo sai. Đây là giả định của nhóm, không phải số liệu thực của ngân hàng.")
        st.caption("Ngưỡng tối ưu ở đây tính trực tiếp trên tập test để minh hoạ. Trong thí nghiệm của báo cáo, ngưỡng được chọn trên tập validation (xem results_*.csv).")

# ---------------- Tab 2
with tab2:
    st.markdown(f"**Kết quả trên tập test, chia '{mode}'** (ngưỡng chọn trên tập validation, chi phí với C_fp = {meta['c_fp']:g})")
    show = res.copy()
    num = show.select_dtypes("number").columns
    st.dataframe(show.style.format({c: "{:.4f}" for c in num if "Chi phí" not in c and "Thời gian" not in c}
                                   | {c: "{:,.0f}" for c in num if "Chi phí" in c}
                                   | {"Thời gian huấn luyện (s)": "{:.1f}"}, na_rep="—"))
    img = OUT / f"pr_curves_{mode}.png"
    if img.exists():
        st.image(str(img), caption="Đường cong Precision-Recall của tất cả mô hình trên tập test")
    st.caption("PR-AUC là thước đo chính vì lớp gian lận rất hiếm; ROC-AUC thường cao và dễ gây hiểu nhầm.")

# ---------------- Tab 3
with tab3:
    n_top = st.slider("Số giao dịch hiển thị (điểm cao nhất)", 10, 200, 30)
    top = df.assign(score=score).sort_values("score", ascending=False).head(n_top).copy()
    top["Cảnh báo"] = top["score"] >= thr
    top["Nhãn thật"] = top["Class"].map({0: "bình thường", 1: "GIAN LẬN"})
    st.dataframe(top[["score", "Cảnh báo", "Nhãn thật", "Amount", "Time"]].rename(columns={"score": "Điểm"}))
    st.caption("Các giao dịch gian lận nằm ngoài danh sách này (điểm thấp) chính là các ca bị bỏ sót, nên xem kỹ khi phân tích lỗi.")

    fn_idx = df.index[(y == 1) & (score < thr)]
    st.markdown(f"**Giao dịch gian lận bị bỏ sót ở ngưỡng hiện tại: {len(fn_idx)}**")
    if len(fn_idx):
        st.dataframe(df.loc[fn_idx, ["Time", "Amount"]].assign(Điểm=score[fn_idx]).sort_values("Điểm"))

    st.markdown("**Giải thích một giao dịch (SHAP, chỉ cho mô hình dạng cây)**")
    pick = st.number_input("Chỉ số dòng trong tập test", 0, len(df) - 1, int(top.index[0]))
    try:
        import shap

        mdl = load_model(mode, model_name)
        if model_name in ("XGBoost", "RandomForest"):
            X = make_features(df.loc[[pick]])
            expl = shap.TreeExplainer(mdl)
            sv = expl.shap_values(X)
            sv = sv[1] if isinstance(sv, list) else sv
            sv = np.asarray(sv).reshape(-1)
            if sv.size != X.shape[1]:  # dạng (1, n_feat, 2)
                sv = np.asarray(expl.shap_values(X))[0, :, 1]
            contrib = pd.Series(sv, index=X.columns).sort_values(key=np.abs, ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(6, 3.5))
            contrib[::-1].plot.barh(ax=ax, color=["#c0392b" if v > 0 else "#2471a3" for v in contrib[::-1]])
            ax.set_title(f"Đặc trưng ảnh hưởng nhất (đỏ: tăng nguy cơ), điểm = {score[pick]:.4f}")
            st.pyplot(fig); plt.close(fig)
            st.caption("V1–V28 là thành phần PCA đã ẩn danh, nên chỉ nhận xét ở mức thống kê, không suy diễn ý nghĩa nghiệp vụ.")
        else:
            st.info("Chọn mô hình XGBoost hoặc RandomForest ở thanh bên để xem giải thích SHAP.")
    except ImportError:
        st.info("Chưa cài thư viện shap (pip install shap) nên không hiển thị giải thích.")
    except Exception as e:  # SHAP có thể lỗi theo phiên bản; không để làm hỏng demo
        st.warning(f"Không tính được SHAP: {e}")

# ---------------- Tab 4
with tab4:
    st.markdown("Tải lên tệp CSV có các cột: " + ", ".join(f"`{c}`" for c in REQUIRED_COLS if c != "Class") +
                " (cột `Class` nếu có sẽ dùng để tính chỉ số).")
    up = st.file_uploader("Tệp CSV", type="csv")
    if up is not None:
        data = pd.read_csv(up)
        miss = [c for c in REQUIRED_COLS if c not in data.columns and c != "Class"]
        if miss:
            st.error(f"Thiếu cột: {miss}")
        else:
            s = predict_score(load_model(mode, model_name), make_features(data))
            out_df = data.assign(diem_gian_lan=s, canh_bao=s >= thr).sort_values("diem_gian_lan", ascending=False)
            st.write(f"{int((s >= thr).sum()):,}/{len(data):,} giao dịch bị cảnh báo ở ngưỡng {thr:.4f}.")
            if "Class" in data.columns and data["Class"].nunique() > 1:
                st.write(f"PR-AUC = {average_precision_score(data['Class'], s):.4f}")
            st.dataframe(out_df.head(200))
            st.download_button("Tải kết quả (CSV)", out_df.to_csv(index=False).encode("utf-8"), "ket_qua_cham_diem.csv", "text/csv")
