# Quy tắc quản lý repository

## 1. Phạm vi mã nguồn công bố

Repository mã nguồn được quản lý tại thư mục `Detect_plate`, nơi chứa `pyproject.toml` và thư mục `.git`. Khi kiểm tra ngày 22 tháng 10 năm 2026, thư mục cha `PlateDetect` có một Git riêng chưa có commit và chứa dữ liệu VNLP bên ngoài repository mã nguồn. Các thao tác commit và push của dự án cần thực hiện tại `Detect_plate`; kiểm tra đúng gốc trước khi thao tác.

```powershell
git rev-parse --show-toplevel
git status --short
```

Nội dung công bố gồm mã Python, test, cấu hình cài đặt, script điều phối, tài liệu Markdown, BibTeX, số liệu tổng hợp nhỏ, mẫu CSV và tài nguyên cascade có giấy phép. Ảnh sơ đồ trong `img/` và ảnh minh họa lịch sử trong `docs/legacy-evidence/` được giữ như tài liệu báo cáo.

## 2. Nội dung lưu cục bộ

| Nhóm | Vị trí hoặc định dạng |
|---|---|
| Môi trường và cache | `.venv/`, `__pycache__/`, `.pytest_cache/`, metadata cài đặt |
| Dữ liệu ký tự và thư mục triển khai cũ | `CNN/`, `CNN_threshold/`, `KNN/`, `SVM/` |
| Dữ liệu và bảng nguồn gốc thực tế | `data/`, ngoại trừ `ocr_groups.example.csv` |
| Tài liệu sinh và đầu ra thực thi | `notebooks/` |
| Đầu ra thí nghiệm | `reports/` |
| Model và checkpoint | `Registration/Checkpoints/`, `Registration/Registry/`, `Registration/RuntimeAssets/`, các định dạng model nhị phân |
| Tài nguyên cá nhân | Biến môi trường cục bộ, khóa truy cập, ảnh thử tại gốc repository, tệp nén và tệp tạm |

Các thư mục cũ đã ngừng chứa triển khai hoạt động; dữ liệu ký tự vẫn tồn tại cục bộ. Mã xử lý hiện tại nằm trong `src/plate_ml/`. Script tạo tài liệu thực thi được lưu trong `scripts/`, giúp tái tạo notebook cục bộ khi cần chạy từ terminal.

## 3. Tệp đã được Git theo dõi

`.gitignore` chỉ ngăn Git tự thêm tệp chưa được theo dõi. Với dữ liệu đã nằm trong index, thao tác `git rm --cached` gỡ theo dõi và giữ tệp trên máy. Ví dụ áp dụng cho tập ký tự:

```powershell
git rm -r --cached --ignore-unmatch -- CNN/CNN_letter_Dataset
```

Lần chuẩn hóa repository ngày 22 tháng 10 năm 2026 gỡ các tài nguyên khớp `.gitignore` khỏi index, gồm 35.500 ảnh ký tự. Thao tác này không xóa dữ liệu cục bộ và không viết lại lịch sử Git. Các commit cũ vẫn có thể chứa dữ liệu hoặc model; kích thước lịch sử không giảm chỉ nhờ quy tắc bỏ qua.

## 4. Kiểm tra trước khi commit

```powershell
git ls-files -ci --exclude-standard
git status --short
git diff --check
git diff --cached --stat
```

Lệnh đầu cần không còn tệp đã theo dõi khớp quy tắc bỏ qua. Kiểm tra danh sách thay đổi để bảo đảm có mã nguồn, tài liệu và số liệu tổng hợp cần công bố; các tệp đầu ra lớn không được thêm bằng tùy chọn ép bỏ qua. Chỉ stage những tệp đã được rà soát. Commit và push là thao tác công bố riêng sau khi nội dung đã được kiểm tra.

## 5. Chuẩn trình bày tài liệu

Tài liệu sử dụng tiếng Việt học thuật, mô tả mục tiêu, cấu hình, dữ liệu, quy trình và giới hạn. Các kết quả phải ghi cỡ mẫu, tập dữ liệu, chỉ số, trạng thái độc lập và phiên bản model. Thay lời dẫn mơ hồ về tệp đầu vào bằng kích thước ảnh, số mẫu, tham số hoặc kết quả cụ thể. Đường dẫn chỉ dùng để xác định tài nguyên và lệnh thực hiện.

Không dùng biểu tượng trang trí, dòng phân cách, dấu cảm thán, dấu chấm rời hoặc ngôn ngữ hội thoại. Markdown, dấu câu thông thường, dấu thập phân, tên tệp, cú pháp lệnh và ký hiệu BibTeX được giữ theo đúng ý nghĩa kỹ thuật. Tài liệu nguồn và giấy phép của bên thứ ba được giữ nguyên nội dung.
