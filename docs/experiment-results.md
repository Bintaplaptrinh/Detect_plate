# Thí nghiệm 01: phương án HOG và SVM cơ sở

## 1. Mục tiêu và cấu hình

Thí nghiệm ngày 21 tháng 10 năm 2026 thiết lập phương án cơ sở cho nhận diện biển số bằng học máy cổ điển. Môi trường gồm Python 3.14.2, OpenCV 4.13.0.92, scikit-learn 1.8.0 và hạt giống ngẫu nhiên 42. Mã lần chạy là `20261021T121906198186Z`. Các phiên bản hoàn tất là `ocr/v3`, `detector/v3` và `pipeline/v3`.

Bộ sinh vùng kết hợp cascade OpenCV biển Nga, Sobel, hình thái học và contour tứ giác. Bộ kiểm chứng vùng HOG SVM được huấn luyện bằng các vùng VNLP thuộc tập train. CCA cơ bản được chọn trên validation khi đối chiếu với phép chiếu histogram và CCA nâng cao. OCR sử dụng deskew theo moment, ảnh 24 x 40 trên canvas 32 x 48 pixel và HOG 540 chiều; SVM được chọn bằng macro F1 validation sau khi so sánh với AdaBoost và mô hình chuyên gia.

Cơ sở HOG, SVM và các phương án đối chiếu được dẫn tại [5] đến [8] trong [tài liệu tham khảo](references.md). Dữ liệu VNLP sử dụng các trích dẫn [1] và [2].

## 2. Dữ liệu và quy trình

| Bài toán | Huấn luyện | Validation | Kiểm tra |
|---|---:|---:|---:|
| OCR, 35 lớp | 2.100 ký tự | 700 ký tự | 700 ký tự |
| Ảnh VNLP, ba bố cục | 54 ảnh | 18 ảnh | 18 ảnh |

OCR lấy tối đa 100 ảnh mỗi lớp, tổng cộng 3.500 ảnh. Ảnh trùng pixel được nhóm trước khi chia gần tỷ lệ 60, 20 và 20 phần trăm. Quan hệ giữa các biến thể tăng cường chưa đầy đủ, nên kiểm tra trùng pixel chưa chứng nhận tính độc lập theo ảnh ký tự gốc.

VNLP lấy 30 ảnh mỗi bố cục, tổng cộng 90 ảnh. Các tập được chia theo định danh biển số. Bộ kiểm chứng vùng và mẫu âm khó chỉ sử dụng ảnh train; validation được dùng để chọn ngưỡng và cấu hình phân đoạn. Tập kiểm tra này đã được quan sát trong phát triển và tiếp tục được dùng ở thí nghiệm 02.

## 3. Kết quả OCR

| Chỉ số trên 700 ký tự kiểm tra | Giá trị |
|---|---:|
| Accuracy | 97,57% |
| Macro F1 | 97,59% |
| Log loss | 0,3523 |

Đây là kết quả trên ký tự đã cắt của nguồn dữ liệu kế thừa. Phạm vi đánh giá khác với ký tự phân đoạn từ ảnh VNLP và nhận diện toàn biển.

## 4. Phát hiện vùng và nhận diện toàn biển

| Chỉ số trên 18 ảnh kiểm tra | Giá trị |
|---|---:|
| Precision vùng ở IoU tối thiểu 0,5 | 60,00% |
| Recall vùng ở IoU tối thiểu 0,5 | 16,67% |
| F1 vùng ở IoU tối thiểu 0,5 | 26,09% |
| Biển đúng trên crop chuẩn, sau hậu xử lý | 2/18 |
| CER trên crop chuẩn, sau hậu xử lý | 44,22% |
| Biển đúng từ ảnh đầy đủ, sau hậu xử lý | 1/18 |
| CER từ ảnh đầy đủ, sau hậu xử lý | 79,59% |

Các chỉ số vùng trên áp dụng cho danh sách vùng dự đoán của detector. Kết quả chuỗi sử dụng vùng đứng đầu theo điểm mô hình. Trường hợp không phát hiện được biển số được ghi nhận là chuỗi rỗng và vẫn tính vào mẫu số.

## 5. Nhận xét và giới hạn

Accuracy ký tự cao chưa chuyển thành độ chính xác toàn biển tương ứng. Recall vùng thấp, phân đoạn bị ảnh hưởng bởi nghiêng và viền dính nét, cùng khác biệt phân phối giữa ký tự kế thừa và crop VNLP là các hạn chế cần kiểm tra tiếp. Tập kiểm tra chỉ có 18 ảnh và chưa độc lập theo camera.

Khả năng nạp lại bundle được kiểm tra bằng xác suất ký tự và chuỗi sau hậu xử lý. Bộ kiểm thử tại thời điểm thực nghiệm có 21 trường hợp đạt yêu cầu; con số này mô tả trạng thái lịch sử của mã. Số liệu tổng hợp được lưu tại [bảng kết quả máy đọc được](results/experiment-summary.json). Lệnh tạo và thực thi thí nghiệm từ terminal có tại [quy trình tái lập](reproduction.md).
