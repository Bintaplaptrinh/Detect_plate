# Dữ liệu và nguồn gốc mẫu

## 1. Tập ký tự kế thừa

Tập ký tự gồm 35.500 ảnh JPG thuộc 35 lớp: chữ số 0 đến 9, chữ cái A đến N và P đến Z. Thư mục cục bộ mặc định là `CNN/CNN_letter_Dataset`; biến `PLATE_OCR_ROOT` cho phép chỉ định vị trí khác. Tên thư mục được giữ để tương thích với các đường dẫn thí nghiệm đã lưu.

Thí nghiệm 01 lấy 100 ảnh mỗi lớp, tổng cộng 3.500 ảnh, chia thành 2.100 ảnh huấn luyện, 700 ảnh validation và 700 ảnh kiểm tra. Ảnh trùng pixel được nhóm trước khi chia tập. Quan hệ giữa các ảnh tăng cường và ảnh gốc chưa được khôi phục đầy đủ, nên các phép đo chỉ có giá trị thăm dò đối với khả năng tổng quát hóa theo nguồn ảnh ký tự.

## 2. Nhóm nguồn ký tự

Bảng nguồn gốc sử dụng hai cột `path` và `group`. Đường dẫn tương đối tính từ thư mục gốc tập ký tự. Tất cả biến thể của cùng ký tự gốc phải có cùng nhóm.

```csv
path,group
A/aug1_0.jpg,source-glyph-001
A/aug2_1.jpg,source-glyph-001
```

`data/ocr_groups.example.csv` mô tả cấu trúc và chứa giá trị minh họa. Bảng thực tế là `data/ocr_groups.csv`, phải bao phủ đầy đủ ảnh được sử dụng và được chuẩn bị từ nguồn tạo dữ liệu. Hệ thống đọc bảng nhóm trước khi lấy mẫu; các nhóm có ảnh trùng pixel được hợp nhất. Chế độ `PLATE_STRICT_EVAL=1` yêu cầu bảng nguồn gốc. Tên ảnh dạng `aug123_4.jpg` chưa đủ để suy ra nhóm gốc đáng tin cậy.

## 3. Tập VNLP

| Bố cục | Thư mục | Số ảnh |
|---|---|---:|
| Một hàng | `one_row` | 19.086 |
| Hai hàng ô tô | `two_rows` | 12.618 |
| Hai hàng xe máy | `two_rows_label_xe_may` | 5.593 |
| Tổng | | 37.297 |

Thư mục mặc định là `../data/VNLP_detection/detection`, tính từ repository. Biến `PLATE_VNLP_ROOT` cho phép chỉ định thư mục chứa ba nhóm bố cục. Tên ảnh chứa chuỗi biển và bốn tọa độ theo thứ tự `x1, y1, x2, y2`. Ví dụ `1_1010_1_36A38587_102_344_227_413.jpg` có nhãn `36A38587`, góc trái trên `(102, 344)` và góc phải dưới `(227, 413)`, tương ứng vùng rộng 125 pixel, cao 69 pixel. Biểu diễn nội bộ là `(x, y, width, height)`.

Bộ đọc kiểm tra tọa độ với kích thước ảnh và ghi định dạng vào bản kiểm kê. Mô tả width và height trong README nguồn dữ liệu không khớp ví dụ tọa độ đã kiểm tra; quy ước được dùng ở dự án là `xyxy`.

Các tập được chia theo chuỗi biển số trước khi sinh crop hoặc tạo nhãn ký tự. Nhãn ký tự VNLP ở thí nghiệm 02 được suy ra từ thứ tự crop, số hàng và chuỗi biển. Trong 951 ảnh huấn luyện, 332 biển thỏa điều kiện số lượng và thứ tự, tạo 2.712 ký tự thuộc 31 lớp. Đây là nhãn yếu, vẫn có khả năng chứa nhiều ký tự trong một crop hoặc gán sai vị trí. Tập kết hợp bổ sung 2.100 ký tự huấn luyện kế thừa, đạt 4.812 mẫu và 35 lớp.

Chia theo biển số và kiểm tra trùng pixel ngăn một số dạng giao thoa dữ liệu. Các tập vẫn cùng nguồn dữ liệu; chưa có phân chia độc lập theo camera hoặc phiên ghi hình. Danh sách split, hạt giống ngẫu nhiên và quy mô từng tập được mô tả tại [quy trình tái lập](reproduction.md) và các báo cáo thí nghiệm.

## 4. Citations

Khi sử dụng dữ liệu VNLP, cần trích dẫn cả hai công trình sau:

[1] Thi-Anh-Loan Trinh, The Anh Pham, và Van-Dung Hoang (2022). Layout-invariant license plate detection and recognition. International Conference on Multimedia Analysis and Pattern Recognition, MAPR 2022, trang 1 đến 6. [DOI: 10.1109/MAPR56351.2022.9924802](https://doi.org/10.1109/MAPR56351.2022.9924802).

[2] The-Anh Pham (2023). Effective deep neural networks for license plate detection and recognition. The Visual Computer, 39(3), trang 927 đến 941. [DOI: 10.1007/s00371-021-02375-0](https://doi.org/10.1007/s00371-021-02375-0).

Mục năm 2023 của bài báo thứ hai là năm xuất bản tập tạp chí; trang nhà xuất bản ghi ngày công bố trực tuyến là 21 tháng 1 năm 2022. Khóa BibTeX `PhamTA23` và thông tin cả hai công trình được lưu tại [references.bib](references.bib).

## 5. Lưu trữ và quyền sử dụng

Ảnh gốc và bảng nguồn gốc thực tế được lưu cục bộ, ngoài nội dung công bố của Git. Mẫu CSV được giữ trong repository. Giấy phép và thông tin ghi công của từng nguồn dữ liệu phải được giữ cùng dữ liệu; dự án không thay đổi quyền sử dụng của nguồn cung cấp. Thông tin kiểm kê trước đây được giữ trong [biên nhận chuyển đổi](migration-receipt.json).
