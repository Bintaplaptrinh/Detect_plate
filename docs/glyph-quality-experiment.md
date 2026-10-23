# Thí nghiệm 04: lọc chất lượng ký tự trước OCR

## 1. Mục tiêu và vị trí xử lý

Thí nghiệm ngày 22 tháng 10 năm 2026 bổ sung bộ lọc crop sau phân đoạn và trước chuẩn hóa HOG. Detector, bộ OCR, đặc trưng và trọng số xếp hạng của `research_pipeline/v3` được giữ nguyên. Mã lần chạy là `20261022T104205345901Z-glyph-filter`; phiên bản riêng là `research_glyph_filter/v1`.

Thứ tự xử lý gồm phát hiện vùng, cắt và hiệu chỉnh, tạo các giả thuyết phân đoạn, lọc chất lượng từng crop, chuẩn hóa deskew, ảnh 24 x 40 trên canvas 32 x 48 pixel, HOG 540 chiều, OCR và xếp hạng. Bộ lọc sử dụng bằng chứng hình học, không sử dụng nhãn hoặc ký tự dự đoán để quyết định loại crop.

## 2. Các trường hợp phát triển

| Nhãn biển | Chuỗi thô trước lọc | Chuỗi thô sau lọc |
|---|---|---|
| `36A40669` | `36A406691` | `36A40669` |
| `29X2534` | `29X254` | `29X2534` |

Ở biển `36A40669`, thành phần ngoại thất bên phải có kích thước 3 x 20 pixel, trong khi ký tự thật cao từ 82 đến 86 pixel. SVM phân loại thành `1` với độ tin cậy khoảng 0,215. Bộ lọc chiều cao tương đối loại thành phần này trước OCR.

Ở biển `29X2534`, ký tự `3` nối với viền trong ảnh nhị phân CCA. Thành phần 361 x 55 pixel bị loại do tỷ lệ và đặc điểm khung. Nhánh phép chiếu histogram đã tách được ký tự `3` với độ tin cậy khoảng 0,894, nhưng chứa nhiễu mép trái và nét viền phải làm giảm điểm giả thuyết. Lọc các crop mép giúp nhánh histogram được chọn với bảy ký tự. Chuỗi đúng được tạo bằng phân đoạn và OCR, không được bổ sung theo nhãn biển.

Hai ảnh này phục vụ phát triển và kiểm tra hồi quy. Hiệu quả độc lập được đánh giá trên tập C.

## 3. Quy tắc bộ lọc

| Quy tắc | Điều kiện |
|---|---|
| Ký tự thấp | Có ít nhất 3 ký tự hỗ trợ trong hàng; chiều cao crop dưới 0,5 lần trung vị chiều cao hàng |
| Nhiễu mép phân mảnh | Sát mép trái hoặc phải trong khoảng lớn hơn giữa 2 pixel và 0,025 chiều rộng ảnh; có ít nhất hai thành phần đáng kể; thành phần lớn nhất chiếm dưới 65% điểm nét; lệch tâm hàng trên 0,18 lần chiều cao tham chiếu |
| Nét viền đặc | Sát mép; tỷ lệ rộng trên cao dưới 0,18; mật độ nét trên 75%; cách cụm chữ trên 0,35 lần chiều cao; lệch hàng trên 0,08 lần chiều cao |

Đường hàng sử dụng trung vị độ dốc giữa các cặp ký tự neo ổn định; mỗi hàng được ước lượng riêng. Độ hẹp đơn lẻ chưa đủ để loại một ký tự. Các kiểm tra giữ ký tự `1` hoặc `I` đủ chiều cao và cùng hàng trong những trường hợp kiểm thử đã xác định. Bộ lọc là tập quy tắc hình học, chưa phải bộ phân loại ký tự và nhiễu được huấn luyện.

Nhật ký lưu vùng, hàng, mật độ nét, tỷ lệ thành phần lớn nhất, chiều cao tương đối, độ lệch hàng, khoảng cách và lý do giữ hoặc loại. Sau lọc, danh sách ký tự, thứ tự và xác suất được tính lại đồng bộ; ảnh nhị phân gốc được giữ cho kiểm tra.

## 4. Chọn cấu hình và chia tập

Ba phương án gồm pipeline 03 chưa lọc, chỉ lọc chiều cao và bộ lọc đầy đủ. Cấu hình đầy đủ được khóa khi sửa được hai ảnh phát triển và không tăng CER thô trên 90 ảnh validation B. Hai phương án lọc có cùng kết quả validation; các quy tắc mép cần thiết cho trường hợp `29X2534`.

Tập test B gồm 180 ảnh của thí nghiệm 03 được dùng làm kiểm tra hồi quy. Tập C mới gồm 180 ảnh, 60 ảnh mỗi bố cục, sử dụng hạt giống 2026102204 và loại toàn bộ định danh từng xuất hiện ở thí nghiệm 02 và 03. Tập C vẫn cùng nguồn dữ liệu, chưa phải tập độc lập theo camera.

## 5. Kết quả

| Validation B, 90 ảnh | Biển đúng sau hậu xử lý | CER thô | CER sau hậu xử lý |
|---|---:|---:|---:|
| Pipeline 03 | 39/90 | 20,79% | 20,92% |
| Chỉ lọc chiều cao | 44/90 | 19,97% | 19,70% |
| Lọc đầy đủ | 44/90 | 19,97% | 19,70% |

Trên validation B, bộ lọc sửa 5 trường hợp và không làm sai trường hợp đang đúng. Các quy tắc mép chưa có phép đo đóng góp độc lập trên tập này.

| Kiểm tra hồi quy B, 180 ảnh | Biển đúng sau hậu xử lý | CER sau hậu xử lý |
|---|---:|---:|
| Pipeline 03 | 80/180 | 25,46% |
| Lọc đầy đủ | 87/180 | 24,64% |

Bộ lọc sửa 7 trường hợp và không làm sai trường hợp đang đúng trên tập B. Tập đã quan sát này không được dùng làm ước lượng cải thiện độc lập.

| Test C mới, 180 ảnh | Biển đúng sau hậu xử lý | CER thô | CER sau hậu xử lý |
|---|---:|---:|---:|
| Pipeline 03 | 70/180 | 29,99% | 29,64% |
| Lọc đầy đủ | 74/180 | 29,03% | 28,76% |

Trên tập C, bộ lọc sửa 5 trường hợp và làm sai 1 trường hợp, tăng ròng 4 biển đúng. Exact match tăng 2,22 điểm phần trăm; CER sau hậu xử lý giảm 0,89 điểm phần trăm.

| Bố cục tập C | Pipeline 03 đúng | Sau lọc đúng | CER sau lọc |
|---|---:|---:|---:|
| Một hàng | 35/60 | 36/60 | 17,83% |
| Hai hàng ô tô | 15/60 | 15/60 | 44,16% |
| Hai hàng xe máy | 20/60 | 23/60 | 24,71% |

Trường hợp hồi quy có nhãn `35B188067`: kết quả sau hậu xử lý thay đổi thành `35B388067`. Vùng detector giữ nguyên. Trước lọc, CCA Otsu tạo chuỗi thô `358188067` và hậu xử lý thành nhãn đúng. Sau lọc, loại một thành phần thấp làm nhánh adaptive được ưu tiên; nhánh này đọc ký tự `1` thành `3`. Chính sách lọc chưa được điều chỉnh thêm theo trường hợp test C này.

## 6. Kiểm chứng và giới hạn

Nạp lại model cho kết quả chuỗi và xác suất khớp trên mẫu validation. Bộ kiểm thử tại thời điểm thí nghiệm có 37 trường hợp đạt yêu cầu. Tập C cho thấy cải thiện vừa phải, đồng thời ghi nhận nguy cơ thay đổi giả thuyết phân đoạn theo hướng bất lợi. Kết quả chưa bảo đảm mọi ký tự hẹp hoặc đứt nét đều được giữ chính xác.

Bộ lọc được triển khai trong `src/plate_ml/glyph_quality.py`; điều phối trong `scripts/run_glyph_filter_experiment.py`. Cơ sở dữ liệu và phương pháp chung được dẫn tại [1], [2], [3], [5] và [6] trong [tài liệu tham khảo](references.md). Thứ tự thực thi có tại [quy trình tái lập](reproduction.md).
