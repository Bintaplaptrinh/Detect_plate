# Quy tắc quản lý repository

## Phạm vi và gốc Git

Repository mã nguồn nằm tại `Detect_plate`, chứa `pyproject.toml` và thư mục `.git`. Thư mục cha `PlateDetect` có Git riêng và dữ liệu VNLP ngoài repository. Thực hiện commit tại đúng gốc mã nguồn:

```powershell
git rev-parse --show-toplevel
git status --short --branch
```

## Nội dung được đưa lên Git

| Nhóm | Nội dung |
|---|---|
| Logic Python | `src/plate_ml/`, `tests/`, `run.py`, cấu hình và phiên bản thư viện |
| Webapp Rust | Mã Rust, HTML, cấu hình Cargo, `Cargo.lock` và lệnh khởi chạy |
| Thực nghiệm | Script tái lập, script tạo notebook, báo cáo Markdown và số liệu tổng hợp nhỏ |
| Tài nguyên bắt buộc | Cascade, giấy phép, dấu kiểm tra và mẫu nhóm nguồn ký tự |
| Sản phẩm nộp | `docs/report/main.pdf` và slide thuyết trình |
| Hướng dẫn | Cài đặt, dữ liệu, mô hình, tái lập và tài liệu tham khảo |

`Cargo.lock` được giữ để cố định phiên bản thư viện ứng dụng. Các script phát triển phương án cơ sở được giữ vì mô hình hiện tại và các thí nghiệm PCA, FLD, HMM phụ thuộc chúng.

## Nội dung chỉ lưu cục bộ

| Nhóm | Vị trí hoặc định dạng |
|---|---|
| Biên soạn báo cáo | Toàn bộ `docs/report/`, ngoại trừ `main.pdf`; script tạo hình báo cáo |
| Mã LaTeX và BibTeX | `.tex`, `.sty`, `.cls`, `.bib`, `.bst` cùng cấu hình và tệp biên dịch liên quan |
| Tài liệu lịch sử | `img/`, `docs/legacy-evidence/`, báo cáo và biên nhận chuyển đổi cũ |
| Chuyển đổi đã hoàn tất | Script di chuyển triển khai cũ |
| Nhánh nghiên cứu ngoài phạm vi | `storage/` |
| Môi trường và cache | `.venv/`, `__pycache__/`, `.pytest_cache/`, `app/target/` |
| Dữ liệu lớn | Dữ liệu ký tự, VNLP, ảnh thử và bảng nhóm nguồn thực tế |
| Mô hình | Checkpoint, Registry, RuntimeAssets và các định dạng nhị phân của mô hình |
| Đầu ra thực nghiệm | `reports/`, `notebooks/`, `tmp/` |
| Cấu hình cá nhân | Biến môi trường, khóa truy cập và thiết lập trình soạn thảo |

Các tệp bị bỏ theo dõi vẫn được giữ trên máy. Khi cần chỉnh sửa hoặc biên dịch báo cáo, dùng thư mục biên soạn cục bộ; Git chỉ nhận PDF hoàn chỉnh. Việc giữ PDF và slide không thay thế các mô hình và dữ liệu cần phục hồi trước khi chạy chương trình.

## Tệp đã được theo dõi

Thêm quy tắc ignore không tự gỡ tệp đã nằm trong index. Dùng `git rm --cached` đối với các tệp cần giữ trên máy nhưng không công bố. Quy tắc riêng cho báo cáo là:

```gitignore
/docs/report/*
!/docs/report/main.pdf
```

Việc này chỉ thu gọn nội dung của các commit mới, không viết lại lịch sử hoặc xóa tệp ở các commit trước.

## Kiểm tra trước khi commit

```powershell
git ls-files -ci --exclude-standard
git ls-files docs/report
git diff --check
git diff --cached --stat
```

Lệnh đầu cần không trả về tệp nào; danh sách trong `docs/report` chỉ có `main.pdf`. Kiểm tra các liên kết tài liệu bằng danh sách tệp sẽ commit để không dẫn đến nguồn LaTeX hoặc ảnh lịch sử đã bỏ theo dõi. Chạy kiểm thử Python và Rust trước khi hoàn tất commit. Chỉ push khi người quản lý repository yêu cầu.
