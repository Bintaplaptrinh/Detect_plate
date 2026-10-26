# Ứng dụng thử nghiệm nhận diện biển số

Webapp cục bộ, giao diện nền trắng. Rust phục vụ giao diện và API bằng Axum, quản lý một tiến trình Python nạp sẵn các mô hình đã thực nghiệm. Phần suy luận tái sử dụng OpenCV và scikit-learn; các mô hình chưa được chuyển sang thực thi thuần Rust.

## Chạy trên máy hiện tại

Tại thư mục gốc `Detect_plate`:

```powershell
powershell -ExecutionPolicy Bypass -File app/run.ps1
```

Hoặc mở `app/run.cmd`. Khi dòng `PlateDetect: http://127.0.0.1:8787` xuất hiện, truy cập địa chỉ đó trong trình duyệt. Giữ tiến trình đang chạy; nhấn Ctrl+C để dừng. Lần đầu Cargo tải và biên dịch các thư viện Rust. Không huấn luyện lại mô hình khi khởi chạy hoặc nhận dạng.

Đổi cổng khi cần:

```powershell
powershell -ExecutionPolicy Bypass -File app/run.ps1 -Port 8788
```

## Sử dụng

1. Chọn SVM hoặc PCA + FLD + HMM.
2. Chọn ảnh từ máy hoặc dùng ảnh mẫu nhiều biển số.
3. Bật công tắc nhận diện nhiều biển số nếu cần. Khi dùng ảnh mẫu, công tắc này tự bật.
4. Nhấn Nhận diện. Kết quả gồm thời gian theo giây, ảnh có khung và chuỗi biển số trên từng khung. Có thể tải ảnh kết quả.

Ảnh mẫu được đọc từ `multi_dectec_sample.jpg` tại gốc dự án. Nếu bản sao dự án không có ảnh này, chức năng chọn ảnh riêng vẫn sử dụng được. Ảnh tải lên được xử lý trong bộ nhớ, không tự lưu vào ổ đĩa. Ứng dụng chỉ lắng nghe trên địa chỉ cục bộ 127.0.0.1.

## Mô hình và môi trường bắt buộc

Máy cần Rust, Cargo và môi trường Python có các thư viện trong `requirements-tested.txt`. Ứng dụng ưu tiên Python trong `.venv`, có thể thay bằng biến môi trường `PLATE_PYTHON` trỏ tới tệp thực thi Python.

Hai tài nguyên đã có trên máy thực nghiệm:

1. Bản phát hành SVM tại `Registration/Registry/research_glyph_filter/v1`, cùng tài nguyên Registry liên quan.
2. Mô hình `reports/pca-fld-hmm-no-svm-180-20261026/no_svm_candidate.joblib` và cấu hình khóa `protocol_before_test.json` trong cùng thư mục. Dấu SHA-256 được kiểm tra trước khi nạp.

Xem `docs/reproduction.md` để phục hồi dữ liệu và mô hình khi chuyển máy. Chỉ nạp các tài nguyên mô hình nội bộ đáng tin cậy. Dữ liệu lớn và mô hình vẫn được Git bỏ qua.

## Hai chế độ nhận diện

Khi tắt công tắc nhiều biển, ứng dụng chạy đúng chuỗi xử lý một ảnh của mô hình đã lưu và trả một vùng được xếp hạng cao nhất. PCA + FLD + HMM sử dụng phát xạ Gaussian trong không gian Fisher; bộ kiểm chứng vùng SVM được dùng chung cho cả hai lựa chọn.

Khi bật công tắc, ứng dụng quét toàn ảnh với cạnh dài tối đa 1600 pixel, bổ sung các ô tối đa 960 pixel có chồng lấn. Tọa độ ô được chuyển về ảnh gốc. Các vùng trùng được nhóm trước OCR; mỗi nhóm giữ tối đa ba cách cắt từ toàn ảnh và các ô, có thêm cách cắt theo trục khi cần. Chỉ nhận các chuỗi đạt điều kiện cấu trúc, điểm mô hình quan sát và điểm kiểm chứng vùng. Khung trùng được loại sau OCR, còn các biển có cùng chuỗi ở hai vị trí khác nhau vẫn được giữ. Có giới hạn 32 nhóm vùng, 96 lượt OCR và 24 kết quả để kiểm soát thời gian.

Đây là bộ điều phối mới phục vụ ảnh nhiều biển, không thay cấu hình hoặc ghi đè mô hình của các thí nghiệm lịch sử. Chưa có đánh giá độc lập trên tập ảnh nhiều biển gán nhãn. Biển nhỏ, mờ, che khuất có thể bị bỏ sót hoặc đọc sai; chương trình hiển thị kết quả mô hình thực tế, không sử dụng chuỗi biển đã biết của ảnh mẫu.

Thời gian hiển thị bao gồm phát hiện, phân đoạn, OCR, gộp khung và vẽ kết quả. Không gồm khởi động Python, nạp mô hình, tải ảnh, giải mã ảnh đầu vào hay nén JPEG đầu ra. Hai mô hình được nạp một lần khi ứng dụng khởi động, suy luận nối tiếp để tránh tranh chấp tài nguyên. Một yêu cầu quá 60 giây làm tiến trình suy luận được khởi tạo lại ở lần tiếp theo.

Giới hạn ảnh là 20 MB, 16 triệu pixel và mỗi chiều ít nhất 16 pixel. Đầu ra JSON có tọa độ khung dạng `x, y, width, height` theo ảnh gốc. Điểm quan sát trong JSON là điểm mô hình ký tự, không phải xác suất đã hiệu chỉnh rằng cả biển được nhận đúng.

## Kiểm tra

```powershell
cargo test --manifest-path app/Cargo.toml
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe scripts/check_app.py
```

Lệnh cuối yêu cầu máy chủ đang chạy. Kiểm tra cả hai thuật toán, công tắc bật và tắt, tọa độ và ảnh đầu ra, dữ liệu không hợp lệ cùng khả năng xử lý tiếp sau một yêu cầu lỗi. Kết quả kiểm tra được lưu cục bộ trong `reports/app-demo`.
