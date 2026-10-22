# Kiểm tra ảnh hưởng màu thân xe và cực tính chữ nền

## 1. Mục tiêu và phạm vi

Kiểm tra ngày 22 tháng 10 năm 2026 đối chiếu ảnh hưởng của màu thân xe, màu nền biển và cực tính chữ nền. Mã lần chạy là `20261022T111323677150Z-color-robustness`. Các model OCR và kiểm chứng vùng được giữ cố định theo `research_glyph_filter/v1`; thí nghiệm này không huấn luyện hoặc đăng ký model thay thế.

Detector của thí nghiệm 03 và 04 kết hợp vùng trắng, vàng và Sobel. Màu thân xe, đèn và ngoại thất có thể tạo vùng ứng viên giả. Phân đoạn thiên về chữ tối trên nền sáng còn có thể mất ký tự trắng trước khi chuẩn hóa HOG. Chuẩn hóa cực tính ở cuối chưa khôi phục được ký tự đã mất ở bước phân đoạn.

## 2. Các phương án kiểm tra

| Phương án | Sinh vùng | Phân đoạn |
|---|---|---|
| `baseline_04` | Vùng màu và Sobel của phương án 04 | Cực tính cơ sở |
| `dual_polarity` | Giữ nguồn vùng của phương án 04 | Ảnh gốc và ảnh đảo cường độ |
| `structural_dual` | Sobel và tứ giác Canny trên ảnh độ sáng | Ảnh gốc và ảnh đảo cường độ |

`ContrastProposer` dùng CLAHE, Sobel theo 6 kernel và Canny với ngưỡng 50 và 150. Cạnh dài ảnh tối đa là 640 pixel; ngân sách tối đa 220 vùng được chia giữa nguồn nét và tứ giác. Nhánh này không dùng ngưỡng hue. `DualPolaritySegmenter` đánh giá hai cực tính trước các giả thuyết và bộ lọc hiện có. OCR giữ chuẩn deskew, ảnh 24 x 40 trên canvas 32 x 48 pixel và HOG 540 chiều.

Các tên màu trong thí nghiệm mô tả giá trị ảnh dựng. Chúng không biểu diễn hoặc xác nhận loại đăng ký, cơ quan hay quy định màu biển.

## 3. Dữ liệu kiểm soát

Tập dựng gồm 50 ảnh có biển: 5 màu thân xe, 5 cặp màu nền và chữ, 2 bố cục. Màu thân xe gồm vàng, trắng, xanh lam, đỏ và xanh lá. Hai chuỗi được vẽ là `29A12345` cho một hàng và `29A112345` cho hai hàng, dùng font OpenCV Hershey Simplex, tỷ lệ 1,1 và nét dày 2 pixel. Thêm 10 ảnh không có biển, gồm 5 màu thân xe, mỗi màu có hai trường hợp có và không có lưới tản nhiệt.

Detector và phân đoạn không nhận tọa độ hoặc số lượng ký tự chuẩn. Tọa độ chỉ dùng để chấm vùng và đánh giá riêng trên crop chuẩn. Ảnh dựng chưa phản ánh đầy đủ bóng, phản quang, nhòe, góc nhìn và phông chữ ngoài thực tế.

## 4. Kết quả trên ảnh dựng

| Màu nền và chữ | Crop chuẩn, cơ sở | Crop chuẩn, hai cực tính | Crop chuẩn, cạnh và hai cực tính |
|---|---:|---:|---:|
| Trắng và đen | 10/10 | 10/10 | 10/10 |
| Vàng và đen | 10/10 | 10/10 | 10/10 |
| Xanh lam và trắng | 0/10 | 10/10 | 10/10 |
| Đỏ và trắng | 0/10 | 10/10 | 10/10 |
| Xanh lá và trắng | 0/10 | 10/10 | 10/10 |

Từ ảnh dựng đầy đủ, phương án cơ sở đúng 20/50 biển. Hai phương án hai cực tính cùng đúng 49/50; một ảnh nền xanh lam vẫn bị chọn sai vùng. Kết quả xác định hạn chế cực tính trong phạm vi dữ liệu kiểm soát, chưa xác nhận độ chính xác trên biển màu thực tế.

| Phương án | Ảnh không có biển nhưng xuất chuỗi OCR |
|---|---:|
| Cơ sở 04 | 4/10 |
| Hai cực tính | 4/10 |
| Cạnh và hai cực tính | 2/10 |

Hai kết quả giả còn lại của phương án cạnh thuộc ảnh thân xe vàng, xuất chuỗi `361`. Phương án cơ sở xuất `IA` ở các ảnh vàng tương ứng. Thay nguồn vùng màu bằng cạnh ảnh chưa loại hết nhầm lẫn ngoại thất. Đây là số lỗi trên 10 ảnh dựng, chưa phải tỷ lệ báo giả trên tập ảnh giao thông.

## 5. Kiểm tra hồi quy trên VNLP validation B

| Phương án | Biển đúng | Exact match | CER sau hậu xử lý |
|---|---:|---:|---:|
| Cơ sở 04 | 44/90 | 48,89% | 19,70% |
| Hai cực tính | 44/90 | 48,89% | 19,84% |
| Cạnh và hai cực tính | 28/90 | 31,11% | 40,90% |

Tập này đã được dùng làm validation ở các thí nghiệm trước. Thay phân phối vùng ứng viên khi giữ bộ kiểm chứng cố định làm giảm kết quả. Phiên bản thử nghiệm chỉ dùng cạnh chưa được chọn để thay thế model 03 hoặc 04.

## 6. Giới hạn và hướng đánh giá

Hướng tiếp theo gồm giữ nguồn vùng cấu trúc độc lập với màu, đánh giá hai cực tính ở giai đoạn phân đoạn, huấn luyện lại bộ kiểm chứng trên phân phối ứng viên mới và bổ sung mẫu âm từ thân xe, đèn, lưới tản nhiệt hoặc chữ quảng cáo. Đánh giá cần ảnh thật được gán nhãn theo cặp màu chữ nền, bố cục và ánh sáng, cùng ảnh không có biển để xác định ngưỡng từ chối.

Bộ kiểm thử của mã ở thời điểm kiểm tra có 53 trường hợp đạt yêu cầu, gồm 15 tổ hợp màu thân xe và biển cùng kiểm tra đặc trưng HOG sau đảo cực tính. Kết quả kiểm thử phần mềm và độ chính xác thực nghiệm có phạm vi riêng. Mã thực hiện ở `src/plate_ml/contrast_proposals.py` và `scripts/check_color_robustness.py`; [quy trình tái lập](reproduction.md) cung cấp điều kiện đầu vào và lệnh thực thi.
