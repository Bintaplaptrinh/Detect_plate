# Kiến trúc hệ thống

## 1. Các cấu hình nghiên cứu

| Cấu hình | Phát hiện vùng | Phân đoạn và OCR | Phiên bản lưu trữ |
|---|---|---|---|
| Thí nghiệm 01 | Haar, Sobel, hình thái học, tứ giác và bộ kiểm chứng HOG SVM | CCA được chọn bằng validation; OCR từ ký tự kế thừa | `pipeline/v3` |
| Thí nghiệm 02 | Giữ detector của thí nghiệm 01 | CCA cố định; OCR từ ký tự VNLP hoặc dữ liệu kết hợp | `pipeline_vnlp_experiment/v1` |
| Thí nghiệm 03 | Vùng ứng viên theo màu và Sobel, bộ kiểm chứng HOG SVM hai giai đoạn | CCA và phép chiếu histogram, chọn giả thuyết bằng OCR | `research_pipeline/v3` |
| Thí nghiệm 04 | Giữ detector của thí nghiệm 03 | Lọc chất lượng crop trước OCR trong từng giả thuyết | `research_glyph_filter/v1` |
| Kiểm tra màu | Nhánh thử nghiệm theo cạnh và tứ giác | Thử hai cực tính chữ nền với model cố định | Chưa đăng ký làm phiên bản thay thế |

Pipeline 03 và 04 xử lý ảnh phương tiện qua sinh vùng, kiểm chứng, hiệu chỉnh phối cảnh, phân đoạn, lọc ký tự nếu được bật, chuẩn hóa HOG, OCR, hậu xử lý và xếp hạng vùng. Hàm suy luận không sử dụng nhãn biển hoặc tọa độ chuẩn. Các nhãn này chỉ được dùng trong huấn luyện và đánh giá.

## 2. Module

| Module | Trách nhiệm |
|---|---|
| `config`, `types` | Cấu hình và vùng chữ nhật dạng `(x, y, width, height)` |
| `imaging` | Đọc ảnh, ảnh xám, CLAHE, hình thái học, hiệu chỉnh phối cảnh, IoU và NMS |
| `features` | Chuẩn hóa cực tính, deskew theo moment và trích xuất HOG ký tự |
| `data`, `vnlp` | Kiểm tra ảnh, nhóm nguồn, loại trùng pixel, đọc nhãn và chia tập |
| `detection`, `segmentation` | Detector và các phương án phân đoạn cơ sở |
| `models`, `training` | SVM, AdaBoost SAMME, mô hình chuyên gia, tuning và checkpoint |
| `heuristics`, `pipeline` | Ràng buộc vị trí ký tự, kết quả thô và sau hiệu chỉnh, đóng gói pipeline |
| `domain_ocr` | Tạo ký tự VNLP có nhãn yếu và đối chiếu nguồn huấn luyện OCR |
| `research_pipeline`, `research_experiment` | Detector hai giai đoạn, các giả thuyết phân đoạn và đánh giá ghép cặp |
| `glyph_quality` | Lọc crop ký tự theo hình học hàng và cấu trúc nét |
| `contrast_proposals` | Vùng ứng viên theo độ tương phản và phân đoạn hai cực tính |
| `evaluation`, `experiments` | Chỉ số ký tự, vùng biển và chuỗi biển |
| `registry` | Checkpoint, phiên bản model bất biến, dấu kiểm tra và nguồn gốc |

## 3. Chuẩn đặc trưng

Ký tự được chuyển về nét trắng trên nền đen, cắt vùng nét, thêm đệm và deskew theo moment. Ảnh được đổi kích thước thành 24 x 40 pixel rồi đặt giữa canvas 32 x 48 pixel. Cấu hình hiện tại không giữ tỷ lệ chiều rộng và chiều cao của vùng nét. OpenCV HOG sử dụng block 16 x 16, bước block 8 x 8, cell 8 x 8 và 9 khoảng hướng, tạo vector 540 chiều.

Vùng biển sử dụng đặc trưng riêng: CLAHE, ảnh 128 x 64 pixel và HOG 3.780 chiều. Bộ kiểm chứng vùng là StandardScaler kết hợp LinearSVC với C = 0,01 và trọng số lớp cân bằng. Bộ OCR của phương án kết hợp là StandardScaler và SVM RBF với C = 3. HOG vùng biển và HOG ký tự có mục tiêu, kích thước và model độc lập. Cơ sở HOG và SVM được dẫn tại [5] và [6] trong [tài liệu tham khảo](references.md).

## 4. Detector và vai trò của cascade

Cấu hình cơ sở dùng cascade biển Nga do OpenCV cung cấp, cùng các nguồn vùng ứng viên theo Sobel, hình thái học và tứ giác. Cascade này là mô hình chuyển miền để đối chiếu; bộ kiểm chứng HOG SVM được huấn luyện trên vùng VNLP. URL, phiên bản OpenCV 4.13.0, SHA-256 và giấy phép cascade được lưu trong `assets/cascades/`.

Cấu hình 03 và 04 dùng `TwoStageDetector`. `PanelProposer` kết hợp ngưỡng nền trắng và vàng với Sobel nhiều kernel; bộ kiểm chứng chọn các vùng giống biển số. Khối sinh vùng này có thể đưa cả thân xe, đèn hoặc lưới tản nhiệt vào danh sách ứng viên. Màu nền chưa đủ để xác nhận vùng biển số. Cascade vẫn tồn tại trong bundle cơ sở để lưu nguồn gốc, nhưng không được gọi bởi detector hai giai đoạn đang hoạt động.

Bộ kiểm chứng ở thí nghiệm 03 học từ 12.439 crop, gồm 3.883 vùng dương và 8.556 vùng âm. Bốn mẫu âm khó trong tập crop đã lấy mẫu được tăng trọng số ở lần fit tiếp theo. Quy trình này chưa quét lại toàn bộ vùng ứng viên sau huấn luyện để khai thác mẫu âm khó.

## 5. Phân đoạn và chọn giả thuyết

CCA tìm các thành phần liên thông của ảnh Otsu và adaptive threshold. Phương án phép chiếu histogram bổ sung cân bằng chiếu sáng bằng nền Gaussian, tìm góc trong khoảng âm 24 đến dương 24 độ với bước 4 độ, loại viền và tách theo một hoặc hai hàng. Phương án kết hợp chọn giữa các giả thuyết bằng chất lượng OCR, độ đồng đều hình học và ràng buộc mềm về độ dài.

Bộ lọc chất lượng được đặt sau phân đoạn và trước chuẩn hóa HOG. Các quy tắc dựa trên chiều cao tương đối của hàng, phân mảnh ở mép và nét viền đặc. Ảnh ký tự được giữ không bị đổi phép chuẩn hóa. Danh sách crop, thứ tự đọc và xác suất được tính đồng bộ sau lọc.

Tối đa 5 vùng được xếp hạng bằng điểm SVM vùng cộng 2,0 lần điểm chất lượng OCR. Ngưỡng SVM vùng được chọn là -1,0. Điểm xếp hạng chưa được hiệu chỉnh thành xác suất chấp nhận biển số. Chuỗi trước và sau ràng buộc vị trí luôn được lưu riêng.

## 6. Các phương án đối chiếu

CCA nâng cao là phương án nội bộ, sử dụng điểm sigmoid từ chiều cao, tỷ lệ và mật độ nét; vùng quá rộng có thể được chia tại điểm thấp của histogram. Phương án này được đánh giá như một ứng viên, không được mặc định ưu tiên hơn CCA cơ bản.

AdaBoost SAMME huấn luyện từng bộ phân loại yếu bằng trọng số mẫu, dừng theo macro F1 validation và giữ cấu hình tốt nhất. SVM sử dụng bộ giải tối ưu hội tụ. Phương án chuyên gia học SVM riêng cho từng nhóm lớp dễ nhầm, dùng logistic gate để phối trộn phân phối với SVM toàn cục. Phương án này tham khảo ý tưởng mô hình chuyên gia [8], với quy trình huấn luyện có giám sát riêng của dự án. AdaBoost đa lớp được dẫn tại [7].

## 7. Quy trình đánh giá

Scaler, bộ phân loại, gate và bộ kiểm chứng chỉ học trên tập huấn luyện. Validation dùng chọn cấu hình, early stopping, ngưỡng vùng, phương án phân đoạn và trọng số xếp hạng. Cấu hình được khóa trước khi chấm tập kiểm tra. Mỗi báo cáo phải ghi rõ tập kiểm tra còn độc lập hay đã được dùng làm tập kiểm tra hồi quy sau phát triển.

| Mức đánh giá | Chỉ số |
|---|---|
| Ký tự đã cắt | Accuracy, macro F1, log loss, ma trận nhầm lẫn |
| Vùng biển | Ghép một đối một theo IoU tối thiểu 0,5, precision, recall và F1 |
| Toàn biển | Exact match và CER, tính cả ca không phát hiện được |
| Crop chuẩn | Phân đoạn và OCR trên vùng chữ nhật được gán nhãn |
| Kiểm tra âm | Số ảnh không có biển nhưng vẫn xuất chuỗi nhận diện |

Crop chuẩn là hình chữ nhật theo trục ảnh, trong khi detector có thể hiệu chỉnh tứ giác; số đo trên crop chuẩn chưa phải giới hạn trên tuyệt đối của pipeline đã hiệu chỉnh. Khớp số lượng ký tự chỉ là chỉ báo gián tiếp cho phân đoạn. Các mẫu khuôn cú pháp hiện tại chưa bao phủ đầy đủ mọi loại biển và không dùng để suy loại đăng ký theo màu.
