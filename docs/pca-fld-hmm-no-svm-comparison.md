# Đối sánh riêng PCA + FLD + HMM không dùng SVM trong OCR

## 1. Mô hình và phạm vi

Nhánh mới thay bộ phân loại SVM bằng phát xạ Gaussian của HMM trong không gian Fisher: **HOG 540 → StandardScaler → PCA 48 → FLD 34 → Gaussian emissions → HMM/Viterbi**. PCA và FLD là phép chiếu; cần mô hình phát xạ để tính độ phù hợp của quan sát với từng ký tự. Ở đây mỗi ký tự có một vector trung bình riêng, dùng chung ma trận hiệp phương sai nội lớp đầy đủ. Likelihood Gaussian được chuẩn hóa theo hàng với prior ký tự đồng đều trước khi đưa vào decoder.

Nhánh OCR mới không tạo, huấn luyện hoặc gọi SVC. Kiểm thử chặn SVC.fit vẫn huấn luyện được nhánh Gaussian. **Detector của toàn hệ thống vẫn giữ SVM kiểm chứng vùng hiện tại** để phép đối sánh chỉ thay khâu OCR. Vì vậy tên “không SVM” trong báo cáo này chỉ áp dụng cho OCR, không áp dụng cho detector.

Phát xạ Gaussian được fit trên đúng 4812 glyph train của bản phát hành, 35 lớp: 2100 legacy + 2712 VNLP. Scaler, PCA, FLD, trung bình và covariance chỉ học train. Covariance = tổng scatter nội lớp/(N−C), cộng 1e−6 × trace(covariance)/d vào đường chéo. Tham số này được khóa trước test, không chọn theo kết quả test.

Phép chiếu PCA/FLD được kiểm tra bằng số với nhánh có SVM của [thí nghiệm trước](pca-fld-hmm-comparison.md). HMM dùng cùng bigram từ 930 chuỗi train, smoothing=1, template và ngưỡng mean log penalty=0.35. Nhánh Gaussian dùng likelihood thật của phân bố quan sát; nhánh SVM đối chiếu dùng posterior SVC như pseudo-emission. Giữ HOG, detector, glyph filter, giả thuyết phân đoạn, top-k và trọng số rerank. Chọn crop và giả thuyết vẫn dựa trên chất lượng OCR của từng nhánh, không dùng nhãn chuẩn.

Chạy mới cả ba nhánh trên 180 ảnh test C lịch sử (60 ảnh/bố cục), 3 lượt, CPU một luồng, warm-up ba ảnh validation. Không dùng lại thời gian của lần chạy trước. Độ trễ là trung bình median theo ảnh, bao gồm toàn bộ inference từ detector đến xếp hạng; không gồm nạp model, đọc ảnh và mạng. Không có giao danh tính/pixel scene giữa train, validation và test; provenance legacy chưa đầy đủ. Test C đã từng được quan sát, chưa phải holdout độc lập mới.

## 2. Kết quả toàn pipeline

| Nhánh | Đúng toàn biển | Exact match | CER ↓ | Mean ms/ảnh ↓ | P95 ms ↓ |
|---|---:|---:|---:|---:|---:|
| SVM hiện tại | 74/180 | 41.11% | 28.76% | 175.92 | 274.55 |
| PCA + FLD + SVM + HMM | 70/180 | 38.89% | 30.33% | 146.09 | 220.36 |
| PCA + FLD + HMM (Gaussian, không SVM OCR) | 70/180 | 38.89% | 34.90% | 132.81 | 204.82 |

### Chênh lệch ghép cặp của nhánh không SVM OCR

| So với | Chênh exact (điểm %) | CI95% exact | Chênh CER (điểm %) | CI95% CER | Tốc độ × | CI95% tốc độ |
|---|---:|---:|---:|---:|---:|---:|
| SVM hiện tại | -2.22 | [-5.56; +0.56] | +6.15 | [+3.82; +8.69] | 1.325 | [1.307; 1.343] |
| PCA + FLD + SVM + HMM | +0.00 | [-3.33; +3.33] | +4.58 | [+2.15; +7.18] | 1.100 | [1.094; 1.106] |

CI95% dùng bootstrap ghép cặp 10.000 lần theo ảnh; các lần đo được xoay thứ tự nhánh. Không có kiểm định không thua kém với biên sai số định trước. Ảnh không phát hiện được biển, tách sai hoặc xuất chuỗi rỗng đều được giữ khi chấm điểm.

### Thời gian từng công đoạn (ms/ảnh)

| Nhánh | Detector | Phân đoạn | HOG | Phát xạ/bộ phân loại, gồm PCA/FLD | Decoder |
|---|---:|---:|---:|---:|---:|
| SVM hiện tại | 50.95 | 63.39 | 11.39 | 47.38 | 1.10 |
| PCA + FLD + SVM + HMM | 51.34 | 62.83 | 11.12 | 17.39 | 1.87 |
| PCA + FLD + HMM (Gaussian, không SVM OCR) | 51.08 | 62.73 | 10.94 | 4.76 | 1.68 |

Cộng công đoạn trên mọi ứng viên và giả thuyết. Median từng công đoạn được lấy riêng; tổng các cột có thể khác độ trễ toàn pipeline, vốn gồm cả overhead và rerank.

## 3. Chẩn đoán crop GT và glyph

| Nhánh | Crop GT exact | Crop GT CER | Mean ms/crop |
|---|---:|---:|---:|
| SVM hiện tại | 37.22% | 20.56% | 35.68 |
| PCA + FLD + SVM + HMM | 33.33% | 22.68% | 25.40 |
| PCA + FLD + HMM (Gaussian, không SVM OCR) | 34.44% | 25.89% | 20.68 |

Chẩn đoán ký tự dùng 434 glyph có nhãn yếu từ 53/180 biển được segmentation gốc nhận. Đây là tập đã lọc theo khớp count/rows, không thay thế chấm chuỗi trên mọi ảnh. Có 0 glyph trùng hash chuẩn hóa với train.

| Mô hình quan sát (chưa HMM) | Accuracy | Macro F1 | Median ms/batch từ HOG |
|---|---:|---:|---:|
| SVM hiện tại | 91.24% | 85.80% | 181.39 |
| PCA + FLD + SVM + HMM | 90.32% | 81.25% | 52.82 |
| PCA + FLD + HMM (Gaussian, không SVM OCR) | 89.40% | 77.99% | 0.94 |

## 4. Diễn giải

Nhánh không SVM OCR đúng 70/180 biển, so với 74 của SVM hiện tại; tốc độ toàn pipeline đạt 1.325×. Việc bỏ SVM giảm chi phí tính điểm quan sát nhưng cũng thay đổi phân bố xác suất, vì vậy có thể ảnh hưởng lựa chọn phân đoạn và ứng viên ngoài thay đổi ký tự.

So với PCA + FLD + SVM + HMM, nhánh không SVM giảm độ trễ từ 146.09 xuống 132.81 ms/ảnh, nhưng CER tăng từ 30.33% lên 34.90%. Số biển đúng bằng nhau không có nghĩa sai số chuỗi tương đương. Accuracy glyph riêng giảm từ 90.32% xuống 89.40%; ảnh hưởng đến toàn chuỗi còn phụ thuộc phân đoạn và xếp hạng ứng viên.

Kết quả chỉ mô tả Gaussian một thành phần/lớp với covariance dùng chung trong cấu hình này; không đại diện mọi mô hình phát xạ HMM. Không tự động thay bản phát hành đang dùng. Mọi tối ưu tiếp theo cần chọn trên validation và kiểm tra trên holdout mới.

Dự đoán raw, decoded và bbox của cả hai nhánh đối chiếu trùng hoàn toàn với thí nghiệm trước; bbox/score detector trùng trên mọi ảnh giữa ba nhánh. Nhãn glyph còn là nhãn yếu và chưa có holdout theo camera.

## 5. Tái lập

```powershell
.\.venv\Scripts\python.exe scripts/run_no_svm_comparison.py --run reports/my-no-svm-run --reference-run reports/pca-fld-hmm-180-20261026-v2 --repeats 3
.\.venv\Scripts\python.exe scripts/build_no_svm_report.py --run reports/my-no-svm-run
.\.venv\Scripts\python.exe scripts/run_notebook.py 06_pca_fld_hmm_no_svm_comparison
```

Cần Registry/dữ liệu lịch sử và candidate.joblib của lần đối sánh trước. Dùng `--vnlp-root` và `--ocr-root` nếu di chuyển dữ liệu. Tài nguyên cục bộ nằm tại `reports/pca-fld-hmm-no-svm-180-20261026`: protocol khóa trước test, audit, manifest có hash, source snapshot, no_svm_candidate.joblib, dự đoán và thời gian từng lượt, summary.json. Notebook phân tích: `notebooks/06_pca_fld_hmm_no_svm_comparison.executed.ipynb`.

Protocol SHA-256: `ee679cb8aaf3c9eb695285501f8a30085c1777f48755b30520d459c7a06c2c6c`.
