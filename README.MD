# Nhận diện biển số xe Việt Nam bằng SVM và PCA, FLD, HMM

Hệ thống nhận diện biển số từ ảnh phương tiện bằng xử lý ảnh và học máy cổ điển.

Hai phương án chính là HOG với SVM và HOG với PCA, FLD, phát xạ Gaussian, HMM. PCA, FLD và HMM là kiến thức trong chương trình; SVM được dùng làm phương pháp đối chiếu. Bộ kiểm chứng vùng SVM dùng chung cho cả hai phương án.

## Sản phẩm

| Sản phẩm | Vị trí |
|---|---|
| Báo cáo kỹ thuật, gồm bảng phân công | [PDF báo cáo](docs/report/main.pdf) |
| Slide thuyết trình | [PowerPoint](docs/slide/slide.pptx) |
| Ứng dụng thử nghiệm | [Hướng dẫn ứng dụng](app/README.md) |
| Mô tả dữ liệu và nguồn | [Dữ liệu](docs/data.md) |
| Hướng dẫn tái lập thực nghiệm | [Tái lập](docs/reproduction.md) |

## Cài đặt và chạy

Thực hiện các lệnh tại thư mục `Detect_plate` chứa `pyproject.toml`. Môi trường đã kiểm tra gồm Python 3.14.2, OpenCV 4.13.0.92, scikit-learn 1.8.0 và Rust 1.96.1.

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-tested.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

Dữ liệu lớn và mô hình đã huấn luyện được lưu cục bộ. Cần phục hồi các bản phát hành SVM và PCA, FLD, HMM theo [hướng dẫn ứng dụng](app/README.md) và [quy trình tái lập](docs/reproduction.md) trước khi nhận dạng. Một bản clone chỉ có mã nguồn chưa chứa các mô hình này.

Sau khi có mô hình và Rust, khởi chạy webapp:

```powershell
powershell -ExecutionPolicy Bypass -File app/run.ps1
```

Mở [http://127.0.0.1:8787](http://127.0.0.1:8787), chọn ảnh, thuật toán và chế độ một hoặc nhiều biển số. Kết quả gồm thời gian theo giây, khung vùng biển và chuỗi dự đoán trên ảnh. Có thể tải ảnh kết quả. Rust phục vụ giao diện và API; tiến trình Python nạp sẵn các mô hình và thực hiện suy luận.

Chức năng dòng lệnh nhận một địa chỉ ảnh:

```powershell
.\.venv\Scripts\python.exe run.py "https://example.com/car.jpg" --cpu
```

Thay địa chỉ ví dụ bằng ảnh thực tế. Lệnh này dùng bản phát hành SVM, trả JSON gồm biển số, trạng thái và thời gian theo ms. Chi tiết tham số và chế độ nhiều biển nằm trong [hướng dẫn ứng dụng](app/README.md).

## Kiểm tra

```powershell
.\.venv\Scripts\python.exe -m pytest -q
cargo test --manifest-path app/Cargo.toml
```

Khi webapp đang chạy và có ảnh mẫu cục bộ, kiểm tra cả hai thuật toán, hai chế độ và xử lý yêu cầu lỗi:

```powershell
.\.venv\Scripts\python.exe scripts/check_app.py
```

## Kết quả thực nghiệm chính

Đối sánh dùng cùng 4.812 ký tự huấn luyện và 180 ảnh kiểm thử C. Thời gian đo trên CPU một luồng, trung bình của trung vị ba lượt theo ảnh, không gồm nạp mô hình hoặc đọc ảnh.

| Phương án | Đúng toàn biển | Exact match | CER | ms/ảnh |
|---|---:|---:|---:|---:|
| SVM | 74/180 | 41,11% | 28,76% | 175,92 |
| PCA, FLD, SVM, HMM | 70/180 | 38,89% | 30,33% | 146,09 |
| PCA, FLD, Gaussian, HMM | 70/180 | 38,89% | 34,90% | 132,81 |

Tập C đã được quan sát trong lịch sử phát triển, chưa độc lập theo camera. Chế độ nhiều biển của webapp là chức năng thử nghiệm bổ sung, chưa được đánh giá độc lập trên tập ảnh nhiều biển có nhãn.

| Tài liệu | Nội dung |
|---|---|
| [Kiến trúc](docs/architecture.md) | Phát hiện vùng, phân đoạn, HOG và mô hình nhận dạng |
| [Đối sánh bốn nhánh](docs/pca-fld-hmm-comparison.md) | Tách tác động giảm chiều và HMM |
| [Đối sánh Gaussian và HMM](docs/pca-fld-hmm-no-svm-comparison.md) | Kết quả chính, độ trễ và giới hạn |
| [Kết quả phát triển ban đầu](docs/experiment-results.md) | Thiết lập phương án cơ sở |
| [Nguồn ký tự VNLP](docs/vnlp-ocr-experiment.md) | Dữ liệu kết hợp và nhãn yếu |
| [Detector và phân đoạn](docs/research-pipeline-experiment.md) | Chuỗi xử lý hai giai đoạn |
| [Bộ lọc ký tự](docs/glyph-quality-experiment.md) | Phân tích và lọc nhiễu trước OCR |
| [Kiểm tra màu](docs/color-robustness.md) | Vùng giả và cực tính ảnh |
| [Tài liệu tham khảo](docs/references.md) | Nguồn dữ liệu, thuật toán và thư viện |

## Dữ liệu

VNLP được công bố tại [trang dữ liệu của tác giả](https://github.com/fict-labs/VNLP). Bản kiểm kê cục bộ có 37.297 ảnh, gồm 19.086 biển một hàng, 12.618 biển hai hàng ô tô và 5.593 biển hai hàng xe máy. Đồ án sử dụng các tập con đã lưu danh sách mẫu, chia theo danh tính biển trước khi tạo ảnh cắt.

Bộ ký tự nội bộ có 35.500 ảnh thuộc 35 lớp, dùng để bổ sung lớp thiếu mẫu. Tập huấn luyện kết hợp gồm 2.100 ký tự nội bộ và 2.712 ký tự VNLP. Cách kiểm tra trùng, nhãn yếu, thông tin nhóm nguồn và giới hạn dữ liệu được mô tả trong [tài liệu dữ liệu](docs/data.md).

Khi sử dụng VNLP, trích dẫn cả hai công trình:

1. Trinh, Pham và Hoang (2022). Layout-invariant license plate detection and recognition. [DOI](https://doi.org/10.1109/MAPR56351.2022.9924802).
2. Pham (2023). Effective deep neural networks for license plate detection and recognition. [DOI](https://doi.org/10.1007/s00371-021-02375-0).

## Cấu trúc repository

| Thư mục | Vai trò |
|---|---|
| `src/plate_ml/` | Logic xử lý ảnh, mô hình, đánh giá và quản lý bản phát hành |
| `app/` | Máy chủ Rust, giao diện web, cấu hình Cargo và hướng dẫn chạy |
| `scripts/` | Thực thi thí nghiệm, tạo notebook và kiểm tra ứng dụng |
| `tests/` | Kiểm thử Python |
| `assets/cascades/` | Cascade của detector cơ sở, giấy phép và nguồn gốc |
| `Registration/` | Hướng dẫn quản lý mô hình; mô hình nhị phân được giữ cục bộ |
| `data/` | Mẫu khai báo nhóm nguồn; dữ liệu thực tế được giữ cục bộ |
| `docs/` | Hướng dẫn, kết quả tóm tắt và tài liệu nộp |

Thư mục `docs/report` chỉ công bố `main.pdf`. Mã LaTeX, công cụ biên soạn, ảnh lịch sử, dữ liệu lớn, mô hình, cache, kết quả chạy và nhánh nghiên cứu trong `storage` được Git bỏ qua. Chi tiết tại [quy tắc quản lý repository](docs/repository-policy.md).

## Thành viên

| Họ và tên | Mã số sinh viên |
|---|---|
| Nguyễn Hữu Tuấn Phát | 2351267274 |
| Trần Mạnh Hùng | 2351267264 |
| Nguyễn Phước Toàn | 2351267278 |
