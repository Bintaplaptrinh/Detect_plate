# Biên dịch báo cáo

Báo cáo dùng XeLaTeX trong MiKTeX và phông Times New Roman của Windows. Nội dung chính cỡ 13, giấy A4, lề bốn phía 2,5 cm, số trang ở góc dưới bên phải. Các công thức dùng phông toán học LaTeX.

Tại thư mục gốc dự án, chạy:

```powershell
powershell -ExecutionPolicy Bypass -File docs/report/build_report.ps1
```

Tài liệu chính là `main.tex`, kết quả là `main.pdf`. Các phần bìa và lời cảm ơn được ghép vào tài liệu chính, không biên dịch riêng. MiKTeX được phép tải các gói LaTeX còn thiếu khi biên dịch.

Lớp đã xác nhận là S26-65TTNT. Mã số sinh viên để trống trong `cover_page.tex` để nhóm điền sau. Bảng phân công trong `phan_cong.tex` phân trách nhiệm hoàn thiện và bảo vệ theo ba nhóm công việc; nhóm cần đối chiếu với đóng góp thực tế trước khi nộp.

Bảng và hình được tạo từ kết quả thực nghiệm đã lưu, không dùng số liệu mô phỏng. Nếu đầy đủ mô hình và dữ liệu cục bộ, có thể tái tạo bằng:

```powershell
.venv/Scripts/python.exe scripts/build_course_report_assets.py
```

Thao tác này không huấn luyện lại mô hình; phần minh họa nạp mô hình đã lưu để nhận dạng lại một mẫu. Dấu kiểm tra nguồn số liệu và kết quả minh họa được ghi trong `asset_audit.json`.

Phạm vi tài liệu là các phương pháp cổ điển SVM, PCA, FLD, Gaussian và HMM. Bộ dữ liệu ký tự nội bộ được mô tả theo vai trò dự phòng và bổ sung các lớp thiếu mẫu. Webapp là giao diện dự kiến, chưa được triển khai trong mã nguồn hiện tại. Báo cáo không khẳng định tập kiểm thử lịch sử là tập độc lập mới và không tự tạo nhận xét của giảng viên.
