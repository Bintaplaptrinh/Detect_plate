# Nhận diện biển số xe Việt Nam bằng HOG và SVM

[![Python 3.14](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![OpenCV 4.13](https://img.shields.io/badge/OpenCV-4.13-green.svg)](https://opencv.org/)
[![scikit-learn 1.8](https://img.shields.io/badge/scikit--learn-1.8-orange.svg)](https://scikit-learn.org/1.8/)
[![Jupyter Notebooks](https://img.shields.io/badge/Jupyter-Notebooks-orange.svg)](https://jupyter.org/)

Dự án nghiên cứu nhận diện biển số từ ảnh phương tiện bằng xử lý ảnh và học máy cổ điển. Hệ thống gồm phát hiện vùng biển số, hiệu chỉnh hình học, phân đoạn ký tự, trích xuất đặc trưng HOG và phân loại SVM. Báo cáo này tổng hợp các thí nghiệm thực hiện ngày 21 và 22 tháng 10 năm 2026.

Pipeline hai giai đoạn đạt 80 biển đúng trên 180 ảnh kiểm tra, tương ứng 44,44%. Bộ lọc chất lượng ký tự đạt 74 biển đúng trên một tập kiểm tra khác gồm 180 ảnh, so với 70 biển đúng của cùng pipeline khi chưa lọc. Hai kết quả thuộc hai tập ảnh khác nhau. Hệ thống hiện phục vụ nghiên cứu; độ chính xác toàn biển và khả năng từ chối vùng giả còn hạn chế.

## 1. Dữ liệu và phạm vi đánh giá

| Nguồn dữ liệu | Quy mô | Vai trò |
|---|---:|---|
| Tập ký tự kế thừa | 35.500 ảnh, 35 lớp | Huấn luyện và đối chiếu OCR |
| VNLP, biển một hàng | 19.086 ảnh | Phát hiện và nhận diện toàn biển |
| VNLP, biển hai hàng ô tô | 12.618 ảnh | Phát hiện và nhận diện toàn biển |
| VNLP, biển hai hàng xe máy | 5.593 ảnh | Phát hiện và nhận diện toàn biển |
| Tổng VNLP được kiểm kê | 37.297 ảnh | Nguồn lấy mẫu cho các thí nghiệm |

VNLP được sử dụng với hai trích dẫn [1] và [2] tại mục [Citations](#4-citations). Các ảnh VNLP được chia theo chuỗi định danh biển số trước khi tạo crop. Tập ký tự kế thừa gồm chữ số 0 đến 9, chữ cái A đến N và P đến Z. Quan hệ giữa các ảnh tăng cường và ảnh ký tự gốc chưa đầy đủ; kết quả OCR trên tập này được xem là đánh giá thăm dò.

Các thí nghiệm sử dụng tập con của dữ liệu. Việc chia theo biển số và kiểm tra trùng pixel chưa thay thế được đánh giá độc lập theo camera hoặc phiên ghi hình. Quy tắc nhãn, tọa độ và nguồn gốc dữ liệu được trình bày trong [tài liệu dữ liệu](docs/data.md).

## 2. Phương pháp

Phương pháp kế thừa cấu trúc phát hiện và nhận diện hai giai đoạn của Mazouz và cộng sự [3], cùng hướng tiền xử lý độ sáng và biên ảnh của Ayele và cộng sự [4]. Cấu hình triển khai được điều chỉnh cho dữ liệu VNLP; các phép đo của dự án được tính trực tiếp trên các tập ảnh đã xác định.

| Giai đoạn | Cấu hình |
|---|---|
| Sinh vùng ứng viên | Ngưỡng nền trắng và vàng kết hợp Sobel theo nhiều kích thước kernel |
| Kiểm chứng vùng biển | CLAHE, ảnh 128 x 64 pixel, HOG 3.780 chiều, StandardScaler, LinearSVC với C = 0,01 |
| Hiệu chỉnh và phân đoạn | Hiệu chỉnh phối cảnh, cân bằng chiếu sáng, CCA và phép chiếu histogram theo hàng |
| Lọc ký tự ở thí nghiệm 04 | Chiều cao tương đối, mức phân mảnh, vị trí mép và độ lệch hàng |
| Chuẩn hóa ký tự | Nhị phân hóa, thống nhất cực tính, deskew theo moment, ảnh 24 x 40 trên canvas 32 x 48 pixel |
| OCR | HOG 540 chiều, StandardScaler, SVM RBF với C = 3; huấn luyện trên 4.812 ký tự thuộc 35 lớp |
| Chọn kết quả | Tối đa 5 vùng, điểm kiểm chứng vùng kết hợp chất lượng OCR với trọng số 2,0, ngưỡng vùng -1,0 |
| Hậu xử lý | Ràng buộc vị trí ký tự, lưu đồng thời chuỗi thô và chuỗi sau hiệu chỉnh |

HOG dựa trên Dalal và Triggs [5]; SVM dựa trên Cortes và Vapnik [6]. Chuẩn hóa ký tự và HOG được giữ đồng nhất giữa dữ liệu kế thừa, ký tự VNLP, huấn luyện và suy luận.

Cascade `haarcascade_russian_plate_number.xml` là mô hình tiền huấn luyện cho biển số Nga của OpenCV 4.13.0, được sử dụng trong phương án cơ sở ở thí nghiệm 01 và 02. Detector hai giai đoạn của thí nghiệm 03 và 04 sử dụng bộ kiểm chứng HOG và SVM, không gọi cascade này. Nguồn, giấy phép và SHA-256 của tài nguyên được lưu trong [thông tin cascade](assets/cascades/provenance.json).

Ngưỡng màu chỉ cung cấp vùng ứng viên và có thể nhầm thân xe, đèn hoặc lưới tản nhiệt với biển số. Thử nghiệm bổ sung theo cạnh ảnh và hai cực tính chữ nền chưa được chọn làm mô hình thay thế do làm giảm kết quả trên VNLP validation. Chi tiết thiết kế có tại [kiến trúc hệ thống](docs/architecture.md).

## 3. Kết quả thực nghiệm

Exact match là tỷ lệ chuỗi biển số đúng hoàn toàn. CER là tổng khoảng cách chỉnh sửa chia cho tổng số ký tự nhãn. Bảng dưới sử dụng chuỗi sau hậu xử lý, tính trên toàn bộ ảnh phương tiện; trường hợp không phát hiện được biển số được tính là chuỗi rỗng.

| Thí nghiệm | Phương án | Tập kiểm tra | Biển đúng | Exact match | CER |
|---|---|---|---:|---:|---:|
| 01 | OCR ký tự kế thừa, detector cơ sở | A, 18 ảnh | 1/18 | 5,56% | 79,59% |
| 02 | OCR kết hợp ký tự kế thừa và VNLP | A, 18 ảnh | 2/18 | 11,11% | 76,19% |
| 03 | Phương án 02 trên tập B | B, 180 ảnh | 21/180 | 11,67% | 80,87% |
| 03 | Detector hai giai đoạn, phân đoạn kết hợp | B, 180 ảnh | 80/180 | 44,44% | 25,46% |
| 04 | Phương án 03 chưa lọc ký tự | C, 180 ảnh | 70/180 | 38,89% | 29,64% |
| 04 | Phương án 03 có lọc ký tự | C, 180 ảnh | 74/180 | 41,11% | 28,76% |

Tập A được giữ nguyên giữa thí nghiệm 01 và 02. Tập B loại các định danh đã dùng ở thí nghiệm 02. Tập C loại các định danh đã dùng ở thí nghiệm 02 và 03. So sánh hiệu quả từng thay đổi được thực hiện trong cùng một tập kiểm tra.

Thí nghiệm 01 đạt accuracy ký tự 97,57% trên 700 ảnh ký tự kế thừa. Chỉ số này đo một bài toán và một nguồn dữ liệu khác với nhận diện toàn biển. Trên tập B, phương án 03 đạt exact match 65,00% cho biển một hàng, 25,00% cho biển hai hàng ô tô và 43,33% cho biển hai hàng xe máy. Độ trễ trung bình là 181 ms trên 30 ảnh, sử dụng CPU và một luồng OpenCV, không tính thời gian đọc ảnh.

Bộ lọc ở thí nghiệm 04 sửa 5 trường hợp và làm sai 1 trường hợp trên tập C. Phép thử màu trên 50 ảnh dựng và 10 ảnh dựng không có biển chỉ kiểm tra cơ chế xử lý; các kết quả đó không đại diện cho độ chính xác trên ảnh giao thông thực tế. Phiên bản chỉ dùng cạnh và hình học đạt 28/90 biển đúng trên validation, thấp hơn 44/90 của phương án 04.

| Báo cáo | Nội dung |
|---|---|
| [Thí nghiệm 01](docs/experiment-results.md) | OCR cơ sở, detector và đánh giá toàn biển |
| [Thí nghiệm 02](docs/vnlp-ocr-experiment.md) | Huấn luyện OCR bằng ký tự VNLP và đối chiếu nguồn dữ liệu |
| [Thí nghiệm 03](docs/research-pipeline-experiment.md) | Phương pháp hai giai đoạn, phân đoạn và xếp hạng vùng |
| [Thí nghiệm 04](docs/glyph-quality-experiment.md) | Lọc ký tự nhiễu và đánh giá trên tập C |
| [Kiểm tra màu](docs/color-robustness.md) | Màu thân xe, cực tính chữ nền và vùng giả |
| [Số liệu tổng hợp](docs/results/experiment-summary.json) | Kết quả máy đọc được, kèm dấu kiểm tra báo cáo gốc |
| [Báo cáo lịch sử](docs/legacy-report.md) | Các phương án KNN, CNN và SVM năm 2025 |

## 4. Citations

Khi sử dụng dữ liệu VNLP, cần trích dẫn cả hai công trình sau:

[1] Thi-Anh-Loan Trinh, The Anh Pham, và Van-Dung Hoang (2022). Layout-invariant license plate detection and recognition. International Conference on Multimedia Analysis and Pattern Recognition, MAPR 2022, trang 1 đến 6. [DOI: 10.1109/MAPR56351.2022.9924802](https://doi.org/10.1109/MAPR56351.2022.9924802).

[2] The-Anh Pham (2023). Effective deep neural networks for license plate detection and recognition. The Visual Computer, 39(3), trang 927 đến 941. [DOI: 10.1007/s00371-021-02375-0](https://doi.org/10.1007/s00371-021-02375-0).

[3] Lakhdar Djelloul Mazouz, Abdelkrim Meche, Abdelaziz Ouamri, và Abdel Wahab Ait Darna (2024). Two-stage HOG/SVM for license plate detection and recognition. Indonesian Journal of Electrical Engineering and Computer Science, 34(1), trang 210 đến 223. [DOI: 10.11591/ijeecs.v34.i1.pp210-223](https://doi.org/10.11591/ijeecs.v34.i1.pp210-223).

[4] Temesegan Walelign Ayele, Eyouel Getachew Eshete, Melaku Eneayehu Getnet, Dires Negash Fente, và Rutvik Mehta (2018). License Plate Recognition Using Machine Learning. Journal of Emerging Technologies and Innovative Research, 5(5), trang 375 đến 380. [JETIR1805485](https://www.jetir.org/papers/JETIR1805485.pdf).

[5] Navneet Dalal và Bill Triggs (2005). Histograms of Oriented Gradients for Human Detection. CVPR, tập 1, trang 886 đến 893. [DOI: 10.1109/CVPR.2005.177](https://doi.org/10.1109/CVPR.2005.177).

[6] Corinna Cortes và Vladimir Vapnik (1995). Support-vector networks. Machine Learning, 20, trang 273 đến 297. [DOI: 10.1007/BF00994018](https://doi.org/10.1007/BF00994018).

Các tài liệu về AdaBoost, mô hình chuyên gia, U-Net lịch sử và thư viện triển khai được ghi tại [danh mục tham khảo](docs/references.md). Trích dẫn có định dạng BibTeX được cung cấp trong [references.bib](docs/references.bib), bao gồm khóa `PhamTA23`.

## 5. Cài đặt và tái lập

Môi trường đã kiểm tra gồm Python 3.14.2, NumPy 2.4.1, OpenCV 4.13.0.92 và scikit-learn 1.8.0. Các lệnh được thực hiện tại thư mục gốc của repository, nơi chứa `pyproject.toml`.

```powershell
python -m pip install -r requirements-tested.txt
python -m pip install -e .
python -m pytest -q
```

Dữ liệu, model và đầu ra thí nghiệm được lưu cục bộ và loại khỏi Git. Một bản clone mã nguồn cần chuẩn bị dữ liệu và các phiên bản model phụ thuộc trước khi chạy thí nghiệm tiếp theo. [Quy trình tái lập](docs/reproduction.md) trình bày thứ tự các thí nghiệm, biến môi trường, điều kiện đầu vào và lệnh terminal. Các lệnh không yêu cầu mở giao diện notebook.

### Nhận diện một ảnh từ URL

Sau khi cài package bằng `python -m pip install -e .`, chạy tại thư mục chứa `run.py`:

```powershell
python run.py "https://example.com/car.jpg" --cpu
```

Thay URL ví dụ bằng liên kết HTTP/HTTPS trả về ảnh xe thực tế. Cú pháp tổng quát: `python run.py <image_url> [--cpu | --gpu]`; mặc định là `--cpu`. Có thể dùng tương đương `python -m plate_ml.inference <image_url> --cpu`.

Lệnh nạp model cục bộ `Registration/Registry/research_glyph_filter/v1` và không huấn luyện lại. Model không có trong Git; cần phục hồi bản phát hành theo [quy trình tái lập](docs/reproduction.md). Có thể đổi thư mục Registry và phiên bản bằng `--registry "D:\models\Registration" --version v1`. Chỉ sử dụng Registry từ nguồn tin cậy.

Đầu ra JSON minh họa (biển số và thời gian thực tế phụ thuộc ảnh, máy và mạng):

```json
{"plate": "29A12345", "response_ms": 1250.321, "inference_ms": 181.234, "device": "cpu", "status": "recognized"}
```

`plate` là chuỗi biển số được xếp hạng cao nhất. `response_ms` tính từ lúc bắt đầu nạp model, tải và giải mã ảnh đến khi nhận diện xong; `inference_ms` chỉ tính detector, phân đoạn và OCR. Các giá trị thời gian dùng đơn vị ms, không bao gồm khởi động Python/import thư viện. Nếu không nhận diện được, `plate` là chuỗi rỗng và `status` là `no_plate`. URL lỗi, nội dung không phải ảnh hoặc thiếu model sẽ báo lỗi qua stderr và trả exit code 1. Tải ảnh có timeout 30 giây và giới hạn 20 MiB.

`--gpu` đã có trong cú pháp nhưng sẽ báo chưa hỗ trợ và trả exit code 1: pipeline HOG/SVM hiện dùng OpenCV và scikit-learn trên CPU, chưa có backend suy luận GPU.

## 6. Project structure

Cấu trúc chính của mã nguồn và tài liệu được công bố trong Git:

```text
Detect_plate/
    assets/cascades/                       # Cascade OpenCV, giấy phép và thông tin nguồn
    data/
        ocr_groups.example.csv             # Mẫu nhóm nguồn ký tự
    docs/
        legacy-evidence/                   # Ảnh minh họa thí nghiệm lịch sử
        results/
            experiment-summary.json        # Số liệu tổng hợp và dấu kiểm tra báo cáo
        architecture.md                    # Kiến trúc và cấu hình hệ thống
        experiment-results.md              # Kết quả phương án cơ sở
        vnlp-ocr-experiment.md             # OCR từ ký tự VNLP
        research-pipeline-experiment.md    # Detector hai giai đoạn và phân đoạn kết hợp
        glyph-quality-experiment.md        # Lọc chất lượng ký tự
        color-robustness.md                # Kiểm tra màu và cực tính
        reproduction.md                    # Quy trình cài đặt và tái lập
        repository-policy.md               # Quy tắc quản lý Git
        references.md                      # Tài liệu tham khảo
        references.bib                     # Trích dẫn BibTeX
    img/                                   # Sơ đồ kiến trúc lịch sử
    Registration/
        README.md                          # Cấu trúc checkpoint và phiên bản model
    scripts/                               # Điều phối thí nghiệm và tạo tài liệu thực thi
    src/plate_ml/                          # Xử lý ảnh, HOG, huấn luyện, đánh giá và Registry
    tests/                                 # Kiểm thử dữ liệu, pipeline và khả năng nạp model
    .gitignore                             # Quy tắc bỏ qua dữ liệu và tài nguyên cục bộ
    pyproject.toml                         # Cấu hình package và dependency
    requirements-tested.txt                # Phiên bản thư viện đã kiểm tra
    README.MD                              # Báo cáo tổng quan
```

`.gitignore` loại môi trường Python, dữ liệu ảnh, notebook sinh tự động, model, checkpoint, cache và đầu ra cục bộ. Báo cáo Markdown, số liệu tổng hợp nhỏ và ảnh minh họa lịch sử trong `docs/legacy-evidence/` được giữ làm tài liệu nghiên cứu. [Quy tắc quản lý repository](docs/repository-policy.md) trình bày cách kiểm tra nội dung trước khi commit.

## Contact

For more details, reach out to:

- **Nguyễn Hữu Tuấn Phát** - [tuanphatnguyenhuu@gmail.com](mailto:tuanphatnguyenhuu@gmail.com)
- **Trần Mạnh Hùng**
- **Nguyễn Phước Toàn**
