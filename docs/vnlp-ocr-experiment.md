# Thí nghiệm 02: huấn luyện OCR từ ký tự VNLP

## 1. Mục tiêu

Thí nghiệm ngày 22 tháng 10 năm 2026 đánh giá ảnh hưởng của nguồn dữ liệu ký tự khi giữ cố định detector, phân đoạn, đặc trưng và hậu xử lý. Ba phương án gồm OCR kế thừa, OCR chỉ dùng ký tự VNLP và OCR kết hợp hai nguồn. Mã lần chạy là `20261022T085302376534Z-vnlp`; hạt giống ngẫu nhiên là 42.

Tiền xử lý thống nhất gồm chuẩn hóa cực tính, cắt vùng nét, deskew có đệm, ảnh 24 x 40 trên canvas 32 x 48 pixel và HOG 540 chiều. HOG dùng block 16 x 16, bước 8 x 8, cell 8 x 8 và 9 khoảng hướng. Cấu hình 28 x 28 hoặc đặc trưng zoning không tham gia thí nghiệm này.

Dữ liệu VNLP được sử dụng với các trích dẫn [1] và [2]; cơ sở HOG và SVM được dẫn tại [5] và [6] trong [tài liệu tham khảo](references.md).

## 2. Dữ liệu và nhãn ký tự

Giữ 18 ảnh validation và 18 ảnh kiểm tra của thí nghiệm 01. Tập train mở rộng từ 54 ảnh bằng cách lấy thêm tối đa 300 ảnh mỗi bố cục, loại các định danh thuộc validation và test. Tập train cuối gồm 951 ảnh, tương ứng 930 định danh biển số. Kiểm tra trùng pixel toàn ảnh không phát hiện ảnh trùng giữa các tập trong lần chạy này.

Ký tự được gán nhãn theo thứ tự hàng và chuỗi biển khi số lượng crop phù hợp. Điều kiện chấp nhận áp dụng cho toàn biển; quy trình không thêm hoặc bớt crop để ép số lượng. Đánh giá chuỗi vẫn sử dụng toàn bộ ảnh của tập tương ứng.

| Tập | Số ảnh | Biển đủ điều kiện tạo nhãn ký tự | Ký tự thu được |
|---|---:|---:|---:|
| Huấn luyện | 951 | 332 | 2.712 |
| Validation | 18 | 5 | 41 |
| Kiểm tra | 18 | 9 | 74 |

Tỷ lệ biển train được chấp nhận là 34,91%. Có 574 biển sai số lượng crop, 42 biển sai số hàng hoặc định dạng và 3 biển một hàng sai số hàng. Ký tự VNLP có 31 lớp, thiếu I, J, Q và W. Phương án kết hợp thêm 2.100 ký tự train của tập kế thừa, đạt 4.812 mẫu thuộc 35 lớp.

Nhãn tạo theo thứ tự crop là nhãn yếu. Kiểm tra train ghi nhận crop mang nhãn `6` của biển `36A42617` chứa cả nét `36`, và crop mang nhãn `1` của biển `36A41591` có hình dạng gần `4`. Khớp số lượng và số hàng chưa bảo đảm nhãn từng crop chính xác.

## 3. Lựa chọn cấu hình

StandardScaler và SVM chỉ được fit trên train. Các cấu hình thử gồm C bằng 0,3, 1 và 3; kernel tuyến tính hoặc RBF. Tiêu chí chọn là CER chuỗi thô trên toàn bộ crop chuẩn validation, sau đó xét exact match. Cấu hình được khóa trước khi chấm test.

| Phương án | C | Kernel | CER thô trên crop chuẩn validation |
|---|---:|---|---:|
| OCR kế thừa | 3 | RBF | 51,37% |
| Chỉ ký tự VNLP | 1 | RBF | 26,03% |
| Ký tự kế thừa và VNLP | 3 | RBF | 23,97% |

Phương án kết hợp được chọn cho các thí nghiệm tiếp theo. Detector và CCA được giữ nguyên theo `pipeline/v3`; các nhánh sử dụng cùng đặc trưng crop chuẩn và vùng đứng đầu của detector.

## 4. Kết quả trên cùng 18 ảnh kiểm tra

| Chỉ số | Kế thừa | Chỉ VNLP | Kết hợp |
|---|---:|---:|---:|
| Crop chuẩn, chuỗi thô đúng | 2/18 | 7/18 | 6/18 |
| Crop chuẩn, CER thô | 46,94% | 25,17% | 25,17% |
| Crop chuẩn, chuỗi sau hậu xử lý đúng | 2/18 | 7/18 | 7/18 |
| Crop chuẩn, CER sau hậu xử lý | 44,22% | 24,49% | 24,49% |
| Ảnh đầy đủ, chuỗi sau hậu xử lý đúng | 1/18 | 2/18 | 2/18 |
| Ảnh đầy đủ, CER sau hậu xử lý | 79,59% | 76,19% | 76,19% |

Accuracy trên cùng tập con 74 ký tự có nhãn yếu lần lượt là 62,16%, 94,59% và 93,24%. Tập con này chỉ gồm 9 biển đủ điều kiện gán nhãn và khác nguồn với 700 ký tự kiểm tra ở thí nghiệm 01.

CER trên crop chuẩn giảm 19,73 điểm phần trăm, trong khi CER từ ảnh đầy đủ giảm 3,40 điểm phần trăm. Khi detector và phân đoạn giữ nguyên, kết quả cho thấy dữ liệu ký tự VNLP có ích đối với OCR. Mức cải thiện toàn biển còn bị giới hạn bởi vùng phát hiện và chất lượng phân đoạn.

## 5. Phiên bản và giới hạn

Các model được lưu riêng tại `ocr_vnlp_only/v1`, `ocr_legacy_plus_vnlp/v1` và `pipeline_vnlp_experiment/v1`. Kiểm tra nạp lại cho kết quả xác suất và chuỗi khớp trên crop validation. Bộ kiểm thử tại thời điểm thí nghiệm có 27 trường hợp đạt yêu cầu.

Tập test có 18 ảnh, đã được quan sát ở thí nghiệm 01. Nhãn ký tự là nhãn yếu và phân bố chữ cái còn thiếu cân bằng. Kết quả hỗ trợ hướng bổ sung ký tự VNLP, nhưng chưa xác nhận khả năng tổng quát trên camera mới. [Quy trình tái lập](reproduction.md) trình bày các tham số lấy mẫu và thứ tự chuẩn bị model; [số liệu tổng hợp](results/experiment-summary.json) giữ các phép đo độc lập với tài nguyên đầu ra cục bộ.
