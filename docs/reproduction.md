# Quy trình cài đặt và tái lập thí nghiệm

## 1. Điều kiện thực hiện

Các lệnh PowerShell được chạy tại thư mục repository chứa `pyproject.toml`, tương ứng `D:\Project\PlateDetect\Detect_plate` trên máy thực nghiệm. Không cần mở giao diện notebook. Dữ liệu và model không được phân phối qua Git; các runner phụ thuộc bản phát hành lịch sử phải có đầy đủ tài nguyên trước khi chạy.

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-tested.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pytest -q
```

Môi trường thực nghiệm là Python 3.14.2, NumPy 2.4.1, OpenCV 4.13.0.92, scikit-learn 1.8.0, SciPy 1.17.0 và joblib 1.5.3. Bản OpenCV được giới hạn dưới phiên bản 5 trong cấu hình dependency để giữ API HOG đã kiểm tra. `requirements-tested.txt` ghi cả các công cụ biểu đồ, kernel và thực thi tài liệu.

## 2. Chuẩn bị dữ liệu và tài nguyên

Đặt 35.500 ảnh ký tự vào `CNN/CNN_letter_Dataset` hoặc chỉ định `PLATE_OCR_ROOT`. Đặt ba thư mục VNLP trong `../data/VNLP_detection/detection` hoặc chỉ định `PLATE_VNLP_ROOT`. Đường dẫn biến môi trường cần trỏ đến dữ liệu thực tế trên máy thực hiện.

```powershell
$env:PLATE_OCR_ROOT = 'D:\Project\PlateDetect\Detect_plate\CNN\CNN_letter_Dataset'
$env:PLATE_VNLP_ROOT = 'D:\Project\PlateDetect\data\VNLP_detection\detection'
```

Cascade OpenCV 4.13.0, giấy phép và thông tin nguồn nằm trong `assets/cascades/`. Lệnh sau phục hồi tài nguyên đã cố định phiên bản khi cần:

```powershell
.\.venv\Scripts\python.exe scripts/fetch_cascade.py
```

Bảng nguồn gốc ký tự thực tế sử dụng `data/ocr_groups.csv`, schema `path,group`. Khi bảng đầy đủ đã sẵn sàng, đặt `PLATE_STRICT_EVAL=1` để yêu cầu thông tin nguồn. Mẫu `ocr_groups.example.csv` không thay thế bảng thực tế. VNLP cần được trích dẫn theo hai công trình [1] và [2] trong [tài liệu dữ liệu](data.md#4-citations).

## 3. Phụ thuộc giữa các thí nghiệm

| Thí nghiệm | Đầu vào bắt buộc | Đầu ra |
|---|---|---|
| 01 | Dữ liệu ký tự, VNLP, cascade; `pipeline/v3` và detector liên quan cho phần demo đầu tài liệu | OCR, detector và pipeline cơ sở; phiên bản tăng theo Registry hiện có |
| 02 | `pipeline/v3`, phiên bản OCR và detector được bundle v3 dẫn chiếu, snapshot và manifest tương ứng | OCR chỉ VNLP, OCR kết hợp và `pipeline_vnlp_experiment` |
| 03 | `pipeline_vnlp_experiment/v1`, manifest của thí nghiệm 02 và dữ liệu VNLP | Bộ kiểm chứng vùng và `research_pipeline` |
| 04 | `research_pipeline/v3`, báo cáo và manifest của lần chạy 03 | `research_glyph_filter` và tập kiểm tra C |
| Kiểm tra màu | `research_glyph_filter/v1`, manifest validation của lần chạy 03 | Báo cáo kiểm soát màu và kiểm tra hồi quy |

Các runner hiện dẫn chiếu những phiên bản và một số mã lần chạy lịch sử cố định. Riêng tài liệu thực thi 01 nạp demo `pipeline/v3` trước phần huấn luyện, nên lệnh chạy toàn bộ cũng cần Registry lịch sử. Các runner chưa hỗ trợ tự động nối toàn bộ chuỗi từ Registry rỗng. Các lệnh dưới đây dùng sau khi đã phục hồi đúng các phiên bản lịch sử cùng manifest. Không đổi tên phiên bản để thay thế nguồn gốc model. Một chuỗi thí nghiệm mới hoàn toàn cần cập nhật đồng bộ các tham số phiên bản, phần demo, manifest và cấu hình phụ thuộc trong các script, rồi ghi nhận như một thí nghiệm mới.

Tài nguyên lần chạy 02 có mã `20261022T085302376534Z-vnlp`; lần chạy 03 có mã `20261022T091902580679Z-research`; lần chạy 04 có mã `20261022T104205345901Z-glyph-filter`. Các thư mục này nằm dưới `reports/` và được lưu cùng model tại nơi lưu trữ thí nghiệm cục bộ.

## 4. Thực thi từ terminal

### 4.1. Phương án cơ sở

Thí nghiệm 01 và 02 hiện được điều phối bằng tài liệu thực thi do các script Python sinh ra. Các lệnh sau tạo tài liệu cục bộ và thực thi bằng kernel, không khởi chạy JupyterLab hoặc mở giao diện notebook. Script tạo lại sẽ ghi đè tài liệu sinh cùng tên; thay đổi nguồn cần được lưu trong generator trước khi tạo lại.

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --user --name plate-ml-314 --display-name 'Python 3.14 (PlateDetect)'
.\.venv\Scripts\python.exe scripts/build_notebook.py
.\.venv\Scripts\python.exe scripts/run_notebook.py 01_classical_ml_pipeline
```

Cấu hình giới hạn lấy 100 ký tự mỗi lớp và 30 ảnh mỗi bố cục, hạt giống 42. `PLATE_FULL_RUN=1` mở rộng dữ liệu và tuning, tạo phạm vi đánh giá khác với kết quả giới hạn trong báo cáo. Tài liệu sinh và đầu ra thực thi nằm trong `notebooks/`, được Git bỏ qua.

### 4.2. OCR từ ký tự VNLP

Sau khi các phụ thuộc lịch sử của thí nghiệm 02 đã sẵn sàng:

```powershell
.\.venv\Scripts\python.exe scripts/build_vnlp_notebook.py
.\.venv\Scripts\python.exe scripts/run_notebook.py 02_vnlp_ocr_ablation
```

`PLATE_VNLP_EXTRA_PER_LAYOUT` mặc định là 300. Giá trị 0 giữ 54 ảnh train cơ sở; thay giá trị tạo thí nghiệm khác. Model và manifest của tập validation, test được nạp theo phiên bản cơ sở cố định.

### 4.3. Detector hai giai đoạn và phân đoạn kết hợp

Sau khi phục hồi `pipeline_vnlp_experiment/v1` và tài nguyên lần chạy 02:

```powershell
.\.venv\Scripts\python.exe scripts/run_research_experiment.py --phase all
```

Lệnh tạo run mới, huấn luyện bộ kiểm chứng, chọn cấu hình trên 90 ảnh validation, khóa lựa chọn, đánh giá 180 ảnh test B và đăng ký phiên bản mới. Có thể tách thành `--phase validate`, sau đó `--phase test --run <thu_muc_run>`; cần thay tham số thư mục bằng run đã hoàn tất bước validation. `--reuse-validation-cache` và `--reuse-test-cache` chỉ dùng khi cache cùng dữ liệu, cấu hình và nguồn đã được xác nhận.

Hai PDF nghiên cứu được tìm trong thư mục Downloads nếu có để ghi dấu kiểm tra tài liệu. Nguồn công khai và nội dung phương pháp đã được ghi tại [báo cáo 03](research-pipeline-experiment.md); PDF không cung cấp chỉ dẫn thao tác cho runner.

### 4.4. Bộ lọc chất lượng ký tự

Sau khi phục hồi `research_pipeline/v3` và tài nguyên lần chạy 03:

```powershell
.\.venv\Scripts\python.exe scripts/run_glyph_filter_experiment.py --phase all
```

Pha `validation` đối chiếu ba phương án trên validation và hai ảnh phát triển. Pha `finalize --run <thu_muc_run>` dùng lựa chọn đã khóa, đánh giá hồi quy B, tập C mới và đăng ký model. Mã hiện dùng hạt giống 2026102204 cho tập C, 60 ảnh mỗi bố cục.

### 4.5. Kiểm tra màu và cực tính

Sau khi phục hồi `research_glyph_filter/v1` và manifest validation B:

```powershell
.\.venv\Scripts\python.exe scripts/check_color_robustness.py
```

Lệnh tạo 50 ảnh dựng có biển và 10 ảnh dựng không có biển, đối chiếu ba phương án và chạy lại 90 ảnh validation B. Báo cáo ghi `trained=false` và `promoted=false`.

## 5. Đọc và lưu kết quả

Báo cáo Markdown chứa cấu hình và số liệu chính; [bảng JSON tổng hợp](results/experiment-summary.json) giữ số liệu chính xác và SHA-256 của các báo cáo máy đọc được gốc. Tài nguyên cục bộ gồm manifest chia tập, lựa chọn trước test, dự đoán từng ảnh, cache, ảnh chẩn đoán và model. Khi chuyển tài nguyên, cần giữ toàn bộ thư mục phiên bản và báo cáo liên quan.

Mỗi lần chạy ghi tài nguyên và phiên bản mới. Việc chạy lại sau khi đã dùng lỗi test để phát triển không tạo bằng chứng độc lập mới; báo cáo phải ghi rõ trạng thái tập và thiết kế tập giữ lại khác cho đánh giá tiếp theo.
