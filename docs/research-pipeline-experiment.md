# Thí nghiệm 03: detector hai giai đoạn và phân đoạn kết hợp

## 1. Mục tiêu và tài liệu nền

Thí nghiệm ngày 22 tháng 10 năm 2026 đánh giá detector và phân đoạn mới khi giữ nguyên OCR kết hợp của thí nghiệm 02. Mã lần chạy là `20261022T091902580679Z-research`. Phiên bản model hoàn tất là `research_pipeline/v3`.

Mazouz và cộng sự [3] cung cấp cơ sở cho mô hình hai giai đoạn HOG SVM: kiểm chứng vùng biển và nhận dạng ký tự, kết hợp tiền xử lý và phân đoạn histogram. Bài nghiên cứu sử dụng biển sau nền vàng của Algeria, dạng dài và chỉ có chữ số. Ngưỡng nền, bố cục và bảng ký tự của VNLP được xác định riêng.

Bài [3] huấn luyện vùng với 410 ảnh biển và 410 ảnh nền. Bảng đánh giá 1.000 crop có 493 biển và 507 vùng nền; 492 dương đúng và 483 âm đúng tương ứng accuracy 97,5%. Accuracy chữ số trung bình 99,46% tính trên các bài toán nhị phân có mẫu âm. Hai chỉ số của bài báo không tương đương exact match từ ảnh phương tiện của dự án.

Ayele và cộng sự [4] định hướng tiền xử lý độ sáng, giảm nhiễu, giãn ảnh và biên theo hai hướng. Bài báo đề cập bộ 62 ký tự huấn luyện nhưng chưa cung cấp quy mô và cách chia tập kiểm tra đủ để đối chiếu định lượng. Thí nghiệm hiện tại sử dụng đánh giá riêng trên VNLP [1], [2]. Thông tin xuất bản và liên kết chính thức có tại [tài liệu tham khảo](references.md).

## 2. Cấu hình thực hiện

| Thành phần | Thông số |
|---|---|
| Kích thước ảnh sinh vùng | Cạnh dài tối đa 640 pixel |
| Vùng trắng HSV | V tối thiểu 80, 130, 180 hoặc 220; S nhỏ hơn 115 |
| Vùng vàng HSV | H từ 12 đến 42; S tối thiểu 60; V tối thiểu 70 |
| Kernel đóng ảnh Sobel | 9 x 3, 17 x 3, 27 x 5, 41 x 7, 13 x 13 và 23 x 21 |
| Ngân sách vùng thô | Tối đa 220 vùng, loại trùng ở IoU 0,85 |
| Kiểm chứng vùng | CLAHE, HOG 3.780 chiều trên ảnh 128 x 64, StandardScaler, LinearSVC C = 0,01, cân bằng lớp |
| Phân đoạn | CCA với Otsu và adaptive threshold; phép chiếu histogram một hoặc hai hàng |
| Hiệu chỉnh trước phép chiếu | Nền Gaussian, góc âm 24 đến dương 24 độ, bước 4 độ, loại viền |
| OCR cố định | 4.812 ký tự, 35 lớp; deskew, ảnh 24 x 40 trên canvas 32 x 48, HOG 540 chiều, SVM RBF C = 3 |
| Chọn vùng | Tối đa 5 vùng; ngưỡng SVM -1,0; trọng số chất lượng OCR 2,0 |

Các giả thuyết phân đoạn được chọn bằng độ tin cậy OCR, độ đồng đều hình học, ràng buộc mềm độ dài 7 đến 9 ký tự và mức tương thích cú pháp. Số hàng hoặc số ký tự chuẩn không được truyền vào suy luận. Điểm chọn vùng chưa được hiệu chỉnh thành xác suất chấp nhận.

## 3. Chia tập và huấn luyện

Giữ 951 ảnh train của thí nghiệm 02. Tất cả định danh từng xuất hiện ở train, validation hoặc test của thí nghiệm 02 được loại trước khi lấy tập validation và test mới. Hạt giống chia tập là 20261022; hạt giống huấn luyện là 42.

| Tập | Một hàng | Hai hàng ô tô | Hai hàng xe máy | Tổng |
|---|---:|---:|---:|---:|
| Train | 318 | 317 | 316 | 951 |
| Validation B | 30 | 30 | 30 | 90 |
| Test B | 60 | 60 | 60 | 180 |

Bộ kiểm chứng học từ 12.439 crop, gồm 3.883 dương và 8.556 âm. Mẫu dương gồm vùng chuẩn, vùng nhiễu vị trí và vùng ứng viên có IoU tối thiểu 0,65. Mẫu âm gồm tối đa 8 vùng ứng viên mỗi ảnh có IoU nhỏ hơn 0,15 và vùng nền không chồng biển. Bốn mẫu âm khó trong tập crop đã lấy mẫu được tăng trọng số và fit lại; chưa thực hiện khai thác mẫu âm bằng cách quét lại toàn bộ ứng viên sau fit.

Tỷ lệ ảnh train có ít nhất một vùng thô đạt IoU 0,5 là 92,14% cho một hàng, 75,39% cho hai hàng ô tô và 91,46% cho hai hàng xe máy. Thời gian tạo crop và huấn luyện bộ kiểm chứng là khoảng 43,2 giây. Các tỷ lệ này mô tả train.

## 4. Lựa chọn bằng validation

Quy trình đối chiếu detector cơ sở, detector mới, phép chiếu histogram, CCA kết hợp histogram và xếp hạng bằng OCR. Ngưỡng vùng được thử ở -1,0, -0,5 và 0; trọng số OCR được thử ở 0, 0,5, 1 và 2. Tổng cộng 29 cấu hình cuối được đánh giá trên validation.

Tiêu chí chọn là CER chuỗi thô từ ảnh đầy đủ, tiếp theo là exact match. Phương án `hybrid_seg_t-1.0_w2.0` được khóa trước test, đạt CER thô 20,79% và 39/90 biển đúng sau hậu xử lý. Phương án 02 trên cùng validation có CER thô 79,48% và 12/90 biển đúng. Phân đoạn kết hợp được phát triển theo validation khi phương án chỉ dùng histogram đạt kết quả thấp.

## 5. Kết quả trên 180 ảnh test B

| Phương án | Biển đúng sau hậu xử lý | CER thô | CER sau hậu xử lý | Recall vùng được chọn ở IoU 0,5 |
|---|---:|---:|---:|---:|
| Phương án 02 | 21/180 | 80,87% | 80,87% | 17,78% |
| Chỉ đổi detector | 59/180 | 36,01% | 35,60% | 66,67% |
| Chỉ đổi sang histogram | 7/180 | 84,68% | 84,68% | 17,78% |
| Detector mới và histogram, vùng đứng đầu | 42/180 | 40,50% | 40,37% | 66,67% |
| Detector mới và phân đoạn kết hợp, vùng đứng đầu | 78/180 | 31,04% | 30,77% | 66,67% |
| Phân đoạn kết hợp và xếp hạng bằng OCR | 80/180 | 25,94% | 25,46% | 71,67% |

Phương án được chọn tăng 59 biển đúng, tương ứng 32,78 điểm phần trăm exact match. Có 66 trường hợp được sửa, 7 trường hợp đang đúng bị làm sai, 14 trường hợp cả hai phương án cùng đúng và 93 trường hợp cùng sai. Histogram phù hợp làm một giả thuyết bổ sung; thay toàn bộ CCA bằng histogram cho kết quả thấp hơn.

Trên crop chuẩn, phân đoạn kết hợp đạt 75/180 biển đúng và CER 21,78%; CCA đạt 59/180 và CER 27,43%. Vùng chuẩn là hình chữ nhật theo trục ảnh; detector có hiệu chỉnh phối cảnh tứ giác nên phép đo crop chuẩn chưa phải giới hạn trên tuyệt đối của pipeline.

| Bố cục | Phương án 02 đúng | Phương án 03 đúng | CER phương án 03 |
|---|---:|---:|---:|
| Một hàng | 14/60 | 39/60 | 11,04% |
| Hai hàng ô tô | 1/60 | 15/60 | 42,86% |
| Hai hàng xe máy | 6/60 | 26/60 | 22,87% |

## 6. Phân tích lỗi và tốc độ

Trong 100 biển còn sai, có 43 trường hợp vùng được chọn có IoU dưới 0,5; 47 trường hợp vùng đạt IoU nhưng số ký tự không khớp; 10 trường hợp vùng và số lượng đều phù hợp nhưng chuỗi sai. Đây là phân nhóm lỗi gián tiếp theo thứ tự, chưa thay thế được nhãn ký tự để quy trách nhiệm cho từng module.

Recall của danh sách tối đa 12 vùng sau kiểm chứng là 77,22%, với precision 6,66% trên toàn bộ danh sách. Vùng cuối có precision và recall 71,67%, tương ứng 129 dương đúng, 51 dương sai và 51 biển bị bỏ sót. Tập test chỉ gồm ảnh có biển, nên chưa đo khả năng từ chối trên ảnh phương tiện không có biển.

Đo trên 30 ảnh, gồm 10 ảnh mỗi bố cục, cho độ trễ trung bình 181 ms, trung vị 176 ms và phân vị 95 là 259 ms. Phép đo dùng CPU, một luồng OpenCV, bao gồm phát hiện, phân đoạn và OCR, không tính đọc ảnh. Các số đo chưa đại diện cho tốc độ camera hoặc phần cứng khác.

## 7. Tính tái lập và giới hạn

Hash mã nguồn được kiểm tra tại thời điểm khóa cấu hình. Sau đánh giá test, các thay đổi phục vụ tuần tự hóa tài nguyên OpenCV và cách lấy mẫu đo tốc độ; model và lựa chọn cấu hình được giữ nguyên. Nạp lại phiên bản cho kết quả chuỗi và xác suất khớp trên validation; bộ kiểm thử lịch sử có 31 trường hợp đạt yêu cầu.

Tập B độc lập theo định danh so với thí nghiệm 02 nhưng vẫn cùng nguồn VNLP. Hai hàng ô tô còn đạt exact match 25,00%. Nguồn ký tự kế thừa chưa có đầy đủ quan hệ tăng cường. Khi các lỗi test B đã được dùng để phát triển phương án tiếp theo, tập B trở thành tập kiểm tra hồi quy. Lệnh thực thi và các phụ thuộc được trình bày tại [quy trình tái lập](reproduction.md).
