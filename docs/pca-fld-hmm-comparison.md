# Đối sánh SVM và PCA + FLD + SVM + HMM trên toàn pipeline

## 1. Phạm vi và thiết kế

Thí nghiệm ngày 26/10/2026 đánh giá đủ 180 ảnh test C lịch sử, 60 ảnh cho mỗi bố cục: một hàng, hai hàng ô tô và hai hàng xe máy. Đây là đối sánh ghép cặp trên tập đã từng được đánh giá, không phải tập giữ lại độc lập mới.

Notebook tham khảo `04_pca_fld_hmm_recognition.ipynb` huấn luyện PCA + FLD + **SVM** và minh họa HMM bằng xác suất mô phỏng. Bảng được gọi là end-to-end trong notebook chỉ tổng hợp phân loại ký tự; chưa chạy detector, phân đoạn và OCR toàn ảnh xe. Chia ngẫu nhiên từng ký tự có thể đưa ký tự cùng biển vào cả train và test. Thí nghiệm này giữ nguyên chia tập theo danh tính biển số và tái dựng dữ liệu train của bản phát hành hiện tại.

Bốn nhánh dùng cùng 4,812 ký tự train (2,100 legacy + 2,712 VNLP), 35 lớp. Kiểm tra lại hash pixel legacy, hash từng glyph VNLP, thống kê scaler và toàn bộ support vectors của SVM hiện tại đều đạt. Train có 951 scene, validation có 90 scene; không có giao danh tính hay pixel scene giữa ba tập.

Giảm chiều: StandardScaler → PCA 48 → FLD 34 → SVC RBF C=3, class_weight=balanced, probability=True. Scaler, PCA, FLD và SVC chỉ fit train; không tìm siêu tham số trên test. PCA giữ 80.44% phương sai train. FLD dùng solver SVD; PCA không tự bảo đảm ma trận scatter nội lớp khả nghịch.

HMM học bigram ký tự trên 930 chuỗi biển train duy nhất, Laplace smoothing=1, ràng buộc theo các template nghiên cứu sẵn có, giải mã Viterbi. Đầu vào là posterior SVC dùng như pseudo-emission, không phải likelihood ảnh của một HMM sinh. Khi không khớp bố cục hoặc mức giảm log xác suất thị giác trung bình vượt 0.35, giữ chuỗi raw. Các template không đại diện đầy đủ quy định đăng ký biển số.

Detector, bộ lọc glyph, các giả thuyết phân đoạn, HOG 540 chiều, top-k và trọng số xếp hạng giữ nguyên. OCR và decoder vẫn có thể thay đổi thứ hạng giả thuyết và ứng viên theo cơ chế hiện tại. Nhãn chuẩn không được đưa vào inference hoặc dùng để chọn ứng viên.

Mỗi ảnh chạy 3 lần cho mỗi nhánh trên CPU, giới hạn một luồng, warm-up ba ảnh validation. Thứ tự nhánh xoay theo ảnh và lượt. Độ trễ chính là trung bình median mỗi ảnh; tính toàn bộ detector → phân đoạn → HOG → PCA/FLD nếu có → predict_proba → decoder → xếp hạng; không gồm nạp model, đọc/giải mã ảnh và mạng. Thời gian từng công đoạn đo với cùng instrumentation cho cả bốn nhánh.

## 2. Kết quả toàn pipeline

| Nhánh | Đúng toàn biển | Exact match | CER ↓ | Chính xác ký tự theo alignment | Localized exact IoU≥0.5 | Mean ms ↓ | P95 ms ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| SVM hiện tại + decoder hiện tại | 74/180 | 41.11% | 28.76% | 73.36% | 38.33% | 168.45 | 263.07 |
| SVM hiện tại + HMM | 73/180 | 40.56% | 29.23% | 72.88% | 37.78% | 169.31 | 261.47 |
| PCA + FLD + SVM + decoder hiện tại | 73/180 | 40.56% | 29.78% | 72.34% | 37.78% | 139.71 | 214.26 |
| PCA + FLD + SVM + HMM | 70/180 | 38.89% | 30.33% | 71.79% | 36.11% | 140.32 | 215.53 |

PCA + FLD + SVM + HMM so với hệ hiện tại: exact-match chênh -2.22 điểm phần trăm (CI95% [-5.56; +0.56]); CER chênh +1.57 điểm phần trăm (CI95% [+0.34; +2.81]). Có 2 ảnh chỉ nhánh mới đúng, 6 ảnh chỉ hệ hiện tại đúng; McNemar exact p=0.2891.

Tốc độ toàn pipeline đạt 1.200× (CI95% [1.190; 1.211]); chênh độ trễ -28.13 ms/ảnh (CI95% [-29.88; -26.42]). Khoảng tin cậy dùng bootstrap ghép cặp 10.000 lần theo ảnh, không biểu diễn biến thiên giữa nhiều máy hoặc nhiều lần khởi động hệ thống.

### Tách tác động giảm chiều và HMM

| Đối sánh (nhánh sau trừ nhánh trước) | Chênh exact (điểm %) | CI95% chênh exact | Chênh CER (điểm %) | Tốc độ × |
|---|---:|---:|---:|---:|
| SVM hiện tại + HMM − SVM hiện tại + decoder hiện tại | -0.56 | [-2.78; +1.11] | +0.48 | 0.995 |
| PCA + FLD + SVM + decoder hiện tại − SVM hiện tại + decoder hiện tại | -0.56 | [-3.33; +2.22] | +1.02 | 1.206 |
| PCA + FLD + SVM + HMM − SVM hiện tại + decoder hiện tại | -2.22 | [-5.56; +0.56] | +1.57 | 1.200 |
| PCA + FLD + SVM + HMM − PCA + FLD + SVM + decoder hiện tại | -1.67 | [-4.44; +1.11] | +0.55 | 0.996 |

### Thời gian từng công đoạn

| Nhánh | Detector | Phân đoạn | HOG | Bộ phân loại gồm PCA/FLD | Decoder |
|---|---:|---:|---:|---:|---:|
| SVM hiện tại + decoder hiện tại | 49.11 | 61.16 | 10.94 | 44.69 | 0.98 |
| SVM hiện tại + HMM | 49.36 | 61.15 | 10.92 | 44.65 | 1.87 |
| PCA + FLD + SVM + decoder hiện tại | 49.46 | 60.78 | 10.68 | 16.53 | 0.90 |
| PCA + FLD + SVM + HMM | 49.20 | 60.73 | 10.70 | 16.55 | 1.77 |

Đơn vị ms/ảnh; cộng thời gian trên mọi ứng viên và giả thuyết được xử lý. Mỗi công đoạn lấy median riêng nên tổng các cột không nhất thiết bằng mean độ trễ toàn pipeline; phần xếp hạng và overhead nằm ngoài các cột công đoạn.

### Theo bố cục

| Bố cục | SVM exact | PCA/FLD/HMM exact | SVM CER | PCA/FLD/HMM CER |
|---|---:|---:|---:|---:|
| one_row | 60.00% | 58.33% | 17.83% | 18.05% |
| two_rows | 25.00% | 25.00% | 44.16% | 46.07% |
| two_rows_label_xe_may | 38.33% | 33.33% | 24.71% | 27.20% |

## 3. Chẩn đoán crop GT và ký tự

Crop GT bỏ detector nhưng vẫn chạy phân đoạn, bộ lọc, HOG và chọn giả thuyết OCR. Đây là chẩn đoán, không phải kết quả triển khai toàn ảnh.

| Nhánh | Crop GT exact | Crop GT CER | Mean ms/crop |
|---|---:|---:|---:|
| SVM hiện tại + decoder hiện tại | 37.22% | 20.56% | 34.19 |
| SVM hiện tại + HMM | 35.56% | 20.83% | 34.55 |
| PCA + FLD + SVM + decoder hiện tại | 35.00% | 22.47% | 24.09 |
| PCA + FLD + SVM + HMM | 33.33% | 22.68% | 24.24 |

Phân loại ký tự riêng dùng 434 glyph có nhãn yếu, được nhận từ 53/180 crop GT bởi segmentation gốc để tương thích tập train. Chỉ nhận biển khớp count và rows; đây là tập chọn lọc dễ hơn, không thay thế đánh giá chuỗi trên mọi ảnh. Có 0 glyph trùng hash chuẩn hóa với train; báo thêm accuracy sau khi loại các glyph này.

| Bộ phân loại | Accuracy | Macro F1 | Accuracy không trùng train | Median ms/batch predict_proba |
|---|---:|---:|---:|---:|
| SVM hiện tại + decoder hiện tại | 91.24% | 85.80% | 91.24% | 176.09 |
| PCA + FLD + SVM + decoder hiện tại | 90.32% | 81.25% | 90.32% | 51.60 |

Timing batch dùng HOG đã tính trước, gồm phép chiếu và predict_proba; không gồm HOG và HMM. Kết quả không tương đương timing predict() một lần trong notebook tham khảo.

## 4. Diễn giải và giới hạn

Ở lần chạy này, hệ hiện tại đúng 74 biển; nhánh đầy đủ đúng 70 biển. Tốc độ toàn pipeline thay đổi 1.200×. Đánh giá việc thay thế cần căn cứ cả sai số chuỗi lẫn độ trễ, không suy từ số chiều đặc trưng. Chưa thực hiện kiểm định không thua kém vì chưa định trước biên chấp nhận sai số.

Bộ phân loại trong pipeline giảm từ 44.69 xuống 16.55 ms/ảnh, trong khi detector và phân đoạn đã chiếm khoảng 110.26 ms/ảnh. Đây là nguyên nhân mức tăng tốc toàn hệ thống nhỏ hơn nhiều so với đo phân loại ký tự riêng. Trên batch 434 glyph, tốc độ predict_proba tăng 3.41×; accuracy thay đổi từ 91.24% xuống 90.32%.

Thêm HMM vào nhánh giảm chiều làm số biển đúng thay đổi từ 73 xuống 70. Cấu hình HMM này chưa cho thấy lợi ích trên test C. CER của nhánh đầy đủ tăng với CI95% chênh lệch nằm trên 0; mức giảm accuracy ký tự nhỏ không bảo đảm giữ chất lượng chuỗi. Giữ bản phát hành SVM hiện tại; chỉ xem nhánh giảm chiều là ứng viên nghiên cứu. Mọi tối ưu tiếp theo nên chọn trên validation và xác minh bằng holdout mới, không chọn lại theo test C.

Đối chiếu hệ SVM chạy lại với dự đoán bản phát hành lịch sử: 180/180 chuỗi raw, decoded và bbox trùng khớp. Danh sách bbox/score detector cũng trùng nhau trên mọi ảnh giữa bốn nhánh. Mọi ảnh không phát hiện được biển hoặc tách sai vẫn nằm trong mẫu số toàn pipeline. Không tự động thay thế bản phát hành đang dùng.

Các giới hạn: test C đã từng được quan sát; nhãn glyph là nhãn yếu; nguồn gốc augmentation legacy và quan hệ nguồn ảnh legacy–VNLP chưa xác minh; chưa có holdout theo camera. Thời gian phản ánh CPU/môi trường cục bộ và inference model đã nạp; không đánh giá độ trễ URL hoặc thời gian khởi động.

### Ví dụ thay đổi kết quả

| Trường hợp | Nhãn chuẩn | SVM hiện tại | PCA/FLD/SVM/HMM |
|---|---|---|---|
| Chỉ hệ hiện tại đúng | 36A40043 | 36A40043 | 36A400432 |
| Chỉ hệ hiện tại đúng | 89G103316 | 89G103316 | 89G13316 |
| Chỉ hệ hiện tại đúng | 29H184359 | 29H184359 | 70H148359 |
| Chỉ hệ hiện tại đúng | 29V336771 | 29V336771 | 29H336771 |
| Chỉ nhánh mới đúng | 35B188067 | 35B388067 | 35B188067 |
| Chỉ nhánh mới đúng | 35G102993 | 3G12993 | 35G102993 |

## 5. Tái lập và tài nguyên

```powershell
.\.venv\Scripts\python.exe scripts/run_pca_fld_hmm_comparison.py --run reports/my-pca-fld-hmm-run --repeats 3
.\.venv\Scripts\python.exe scripts/build_pca_fld_hmm_report.py --run reports/my-pca-fld-hmm-run
```

Notebook phân tích chỉ đọc số liệu đã lưu; có thể thực thi bằng `python scripts/run_notebook.py 05_pca_fld_hmm_pipeline_comparison` sau khi cài kernel `plate-ml-314` như mục tái lập môi trường.

Cần phục hồi Registry và các manifest lịch sử theo [quy trình tái lập](reproduction.md). Có thể chỉ định `--vnlp-root`, `--ocr-root`, `--pca-components` và `--reference-notebook`. Thay cấu hình tạo thí nghiệm mới; không chọn cấu hình tốt nhất bằng test C.

Run cục bộ: `reports/pca-fld-hmm-180-20261026-v2`. Gồm protocol khóa trước test, audit train/split, manifest có hash, snapshot nguồn, candidate.joblib, dự đoán từng ảnh cho bốn nhánh và crop GT, thời gian từng lượt, chẩn đoán glyph và summary.json. Notebook phân tích được sinh tại `notebooks/05_pca_fld_hmm_pipeline_comparison.ipynb`.

Protocol SHA-256: `c233378c61da8563d5b853be644be372d26a9e131d731ebb8deab94107154f7f`.
