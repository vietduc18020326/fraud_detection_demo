# Phát hiện gian lận thẻ tín dụng – mã nguồn và demo

Mã nguồn cho bài tập nhóm Nhận dạng mẫu: so sánh baseline và các phương án cải tiến trên dữ liệu mất cân bằng cực độ, kèm demo tương tác (Streamlit).

## 1. Cấu trúc

```
fraud_detection_demo/
├── data/                  # đặt creditcard.csv vào đây (không đưa lên git)
├── fraud/
│   ├── common.py          # nạp dữ liệu, đặc trưng, chia dữ liệu, metric, chi phí
│   └── models.py          # các mô hình (baseline + cải tiến)
├── train.py               # huấn luyện + đánh giá, ghi kết quả vào outputs/
├── app.py                 # demo Streamlit
├── make_synthetic.py      # sinh dữ liệu GIẢ LẬP để thử code (không dùng cho báo cáo)
└── requirements.txt
```

## 2. Cài đặt

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`shap` là tuỳ chọn (chỉ để giải thích từng giao dịch trong demo). Nếu cài lỗi, bỏ dòng đó; demo vẫn chạy.

## 3. Dữ liệu

Bộ **Credit Card Fraud Detection** (ULB và Worldline) trên Kaggle: 284.807 giao dịch, 492 gian lận.

1. Tải bộ dữ liệu từ Kaggle: [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (cần tài khoản Kaggle), giải nén và đặt tệp `creditcard.csv` vào thư mục `data/` (`data/creditcard.csv`).
2. Ghi trong báo cáo: đường dẫn (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), ngày truy cập, điều khoản sử dụng, và trích dẫn Dal Pozzolo et al. (2015).

Phương án dự phòng: `python train.py --openml` thử tải bản "creditcard" từ OpenML (id 1597). Code có kiểm tra cột và báo lỗi nếu bản đó khác định dạng; hãy kiểm tra nguồn và giấy phép trước khi dùng cho báo cáo.

Muốn thử code khi chưa có dữ liệu thật:

```bash
python make_synthetic.py --out data/synthetic.csv
python train.py --data data/synthetic.csv --fast --out outputs_synthetic
FRAUD_OUT=outputs_synthetic streamlit run app.py
```

Kết quả trên dữ liệu giả lập không có ý nghĩa khoa học, tuyệt đối không đưa vào báo cáo.

## 4. Chạy

```bash
python train.py --data data/creditcard.csv        # cả 2 cách chia: time và random
streamlit run app.py
```

Tuỳ chọn của `train.py`: `--split time|random|both`, `--c-fp 10` (chi phí giả định mỗi cảnh báo sai), `--seed 42`, `--fast` (mô hình nhỏ, chạy thử), `--out outputs`.

Thời gian chạy trên dữ liệu thật phụ thuộc máy; `RF+SMOTE` và `RandomForest` là hai mô hình chậm nhất. Nếu máy yếu, dùng `--fast` hoặc `--split time`.

Đầu ra trong `outputs/`:

| Tệp | Nội dung |
|---|---|
| `results_{time,random}.csv` | Bảng số liệu các mô hình trên tập test (dùng để điền Bảng 5, 6, 7 của báo cáo) |
| `pr_curves_*.png` | Đường cong Precision–Recall của tất cả mô hình (Hình 4 của báo cáo) |
| `test_scores_*.csv.gz` | Điểm dự đoán trên tập test, demo dùng để vẽ |
| `models/*/*.joblib` | Mô hình đã huấn luyện |
| `meta_*.json` | Số mẫu, số gian lận mỗi tập, C_fp |

## 5. Các mô hình

| Tên | Nhóm | Ghi chú |
|---|---|---|
| LogReg | Baseline | Có chuẩn hoá đặc trưng |
| RandomForest | Baseline | Tham số mặc định |
| LogReg+class_weight | Cải tiến A | Trọng số lớp `balanced` |
| RF+SMOTE | Cải tiến A | SMOTE (tỷ lệ 0,1) nằm trong pipeline nên chỉ tác động lên tập train |
| XGBoost | Cải tiến B | `scale_pos_weight` theo tỷ lệ lớp; nếu không có xgboost thì dùng HistGradientBoosting |
| IsolationForest | Cải tiến C | Huấn luyện chỉ trên giao dịch bình thường; điểm bất thường chuẩn hoá về [0, 1] |

Cải tiến D (ngưỡng theo chi phí): ngưỡng được chọn trên tập validation để tối thiểu hoá
`tổng số tiền gian lận bị bỏ sót + C_fp × số cảnh báo sai`, rồi áp dụng lên tập test. C_fp là giả định của nhóm.

## 6. Quy ước đánh giá (để tránh lỗi hay gặp)

- Chia `time`: phần đầu để train, kế tiếp validation, cuối cùng test. Chia `random`: ngẫu nhiên phân tầng.
- Chuẩn hoá chỉ fit trên train; SMOTE chỉ áp dụng trên train; ngưỡng chọn trên validation; test chỉ dùng để báo cáo.
- Cột `Time` thô không đưa vào mô hình (tập test chia theo thời gian nằm ngoài miền huấn luyện); thay bằng giờ trong ngày dạng sin/cos.
- Chỉ số chính: PR-AUC. Ngoài ra Recall@Precision80, Precision/Recall/F1 và chi phí. Không dùng Accuracy.
- Tập test chỉ có khoảng 100 giao dịch gian lận khi chia theo thời gian, nên kết quả dao động mạnh: nên chạy nhiều `--seed` (với chia `random`) và nêu độ dao động trong báo cáo.

## 7. Demo (app.py)

- **Ngưỡng và chi phí**: thanh trượt ngưỡng, cập nhật tức thời Precision, Recall, số cảnh báo, ma trận nhầm lẫn, tổng chi phí; nút đặt về ngưỡng tối ưu.
- **So sánh mô hình**: bảng kết quả và đường cong PR.
- **Giao dịch nghi ngờ**: danh sách điểm cao nhất, các ca gian lận bị bỏ sót, giải thích SHAP cho XGBoost/RandomForest.
- **Tải CSV**: chấm điểm tệp giao dịch của người dùng (cùng các cột `Time`, `V1`–`V28`, `Amount`).
