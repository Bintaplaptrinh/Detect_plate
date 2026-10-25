# Fine-tune MTCNN và YOLO26n 10 epoch trên VNLP — 25/10/2026

MTCNN: đúng toàn biển từ **0 lên 0/100**, recall box được chọn từ **3.00% lên 64.00%**. YOLO26n + EasyOCR: đúng toàn biển từ **1 lên 20/100**, recall từ **53.00% lên 99.00%**. OCR và ngưỡng suy luận được giữ nguyên; không huấn luyện LPRNet/EasyOCR.

## Dữ liệu và chống giao tập

Dùng đúng split train hiện có của dự án: **951 ảnh train, 90 ảnh validation**. Không mở rộng thành toàn bộ ảnh VNLP còn lại. Hai detector dùng cùng các scene train/validation. 100 ảnh test giữ nguyên byte manifest từ lần đối sánh trước: 34 biển một hàng, 33 biển hai hàng ô tô, 33 biển xe máy. Loại trừ theo đường dẫn, định danh biển và kiểm tra SHA-256 pixel giải mã; train/validation/test không giao định danh hoặc pixel. Test không được đưa vào YAML huấn luyện YOLO hoặc tạo patch/hard negative MTCNN.

Tập test đã được đánh giá trước, nên đây là phép đối sánh trước/sau trên test đã biết, không phải bằng chứng độc lập mới. Nguồn ký tự kế thừa của HOG/SVM và quan hệ với dữ liệu huấn luyện checkpoint bên ngoài chưa được xác minh đầy đủ theo camera/phiên ghi hình.

## Huấn luyện đã khóa trước khi chạy

Seed `2026102510`, GPU **NVIDIA GeForce RTX 3060**. Dùng checkpoint pretrained của đúng hai repo trong [đối sánh gốc](comparison-experiment.md). Không chọn checkpoint hoặc ngưỡng theo test. Đánh giá **checkpoint epoch 10 cuối cùng**, không chọn best theo test hoặc best epoch validation. Log và checkpoint được lưu từng epoch.

**MTCNN:** fine-tune PNet đủ 10 epoch, sau đó ONet đủ 10 epoch (mỗi stage 10, không phải 5+5). Khởi tạo bằng PNet/ONet upstream. Adam lr=0,0001, batch 128, loss đúng mask/trọng số upstream: 0,02 CE trên positive/background và 0,6 MSE trên positive/partial. Mỗi scene tạo 12 positive, 8 partial và 32 background; IoU ≥0,65 là positive, 0,4–0,65 là partial, <0,3 là background, khoảng 0,3–0,4 bỏ qua. Jitter quanh box và lấy nền trong ảnh; chuẩn hóa RGB (x−127,5)/128. PNet input 47×12, ONet 94×24. Sau PNet epoch10, khai thác thêm **15216 patch** từ proposal trên train, tối đa 16/ảnh; ONet dùng tổng **64668 patch**. Validation không được cập nhật gradient. STN/LPRNet, bảng ký tự Trung Quốc và greedy decoder giữ nguyên.

**YOLO26n:** bắt đầu từ `models/best.pt` của project Esteebaan23; **10 epoch**, input 640, batch16, AdamW lr0=0,001, lrf=0,01, warm-up1 epoch, patience=0 (tắt early stopping), close_mosaic=0, AMP bật, workers0. Nhãn VNLP xywh chuyển thành YOLO center xywh chuẩn hóa, một lớp `License Plate`. Các augmentation còn lại theo Ultralytics. Dùng `yolo_train/weights/last.pt` sau đủ 10 epoch, không dùng best.pt để chọn kết quả thuận lợi. Vehicle segmentation và English EasyOCR không thay weight. [API train chính thức](https://docs.ultralytics.com/modes/train/).

Thời gian huấn luyện: PNet 12.2 giây; ONet 37.8 giây; YOLO 167.6 giây. Thời gian stage MTCNN gồm train/validation/checkpoint, chưa gồm sinh patch/hard mining.

## Kết quả trên đúng 100 ảnh test

| Hệ thống | Đúng toàn biển | CER ↓ | Precision box | Recall box | F1 box | AP50 all | Mean CPU ms/ảnh | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HOG/SVM hiện tại | 45/100 | 26.78% | 73.00% | 73.00% | 73.00% | 64.68% | 179.10 | 5.58 |
| MTCNN trước fine-tune | 0/100 | 98.03% | 50.00% | 3.00% | 5.66% | 3.96% | 100.62 | 9.94 |
| MTCNN sau 10 epoch | 0/100 | 95.95% | 83.12% | 64.00% | 72.32% | 62.08% | 259.96 | 3.85 |
| YOLO26n trước fine-tune | 1/100 | 94.35% | 77.94% | 53.00% | 63.10% | 50.77% | 105.72 | 9.46 |
| YOLO26n sau 10 epoch | 20/100 | 35.26% | 100.00% | 99.00% | 99.50% | 99.01% | 252.38 | 3.96 |

Precision/recall/F1 dùng **box cuối cùng được chọn**, ghép một-một IoU ≥0,5; AP dùng mọi box sau ngưỡng, nội suy 101 điểm, không phải toàn bộ COCO evaluator. Exact/CER tính trên đủ 100 ảnh; không phát hiện/không đủ confidence OCR là chuỗi rỗng. CER = tổng edit distance / tổng ký tự nhãn, thấp hơn tốt hơn. `summary.json` còn giữ AP50:95, kết quả theo bố cục, p50/p95 và các dự đoán chi tiết.

Đánh giá lại cả baseline lẫn checkpoint mới trong cùng lượt chạy trên CPU Ryzen 7 7700, một luồng OpenCV/PyTorch/BLAS, batch1, 3 lượt; warm-up 3 ảnh validation. Lấy median các lượt cho từng ảnh rồi tính mean trên 100 ảnh. Bỏ thời gian nạp model/đọc ảnh/ghi debug; giữ toàn bộ detector và OCR được gọi. Kiểm tra số luồng sau warm-up và từng inference. Các baseline tái tạo đúng text/box của lần đối sánh trước trên cả 100 ảnh; độ trễ có thể khác do thời điểm đo.

Giữ MTCNN min plate 50×15, PNet threshold0,6, ONet threshold0,8, OCR resize94×24. Nhánh YOLO giữ infer_scale0,5, plate input416, conf0,30, OCR gate0,75, ROI tắt và car-only tắt; vẫn chạy vehicle segmentation. Chọn confidence cao nhất cho hai detector mạng. Giữ bản sửa lỗi đọc y của quadrilateral trong ghép dòng EasyOCR đã dùng ở thí nghiệm trước. Không đổi ký tự O/0, I/1 theo nhãn; EasyOCR chỉ viết hoa/bỏ dấu phân cách. Số ảnh được gọi EasyOCR: **9 trước**, **91 sau**; thay đổi coverage là hệ quả confidence detector mới, không phải đổi ngưỡng.

## So sánh ghép cặp sau trừ trước

Exact delta và CI dùng điểm phần trăm; bootstrap 10.000 lần theo cặp ảnh. McNemar exact hai phía. Hai cột “chỉ sau/trước đúng” đếm ảnh được sửa và ảnh bị làm sai; không chọn lại mẫu.

| Detector | Exact delta | CI95% | Chỉ sau đúng | Chỉ trước đúng | McNemar p |
|---|---:|---|---:|---:|---:|
| mtcnn | 0.00 | [0.00; 0.00] | 0 | 0 | 1 |
| yolo | 19.00 | [12.00; 27.00] | 19 | 0 | 3.8147e-06 |

Đây là fine-tune **detector**. MTCNN nhận diện toàn biển còn phụ thuộc STN/LPRNet pretrained Trung Quốc và giả định bố cục; cải thiện phát hiện không bảo đảm cải thiện chuỗi biển Việt Nam. YOLO còn phụ thuộc EasyOCR và gate0,75 của project. Không suy diễn rằng kết quả này là giới hạn của các kiến trúc hoặc của pipeline đã fine-tune cả OCR.

## Tái lập và tài nguyên

Trong thư mục `Detect_plate`, dùng môi trường đã có PyTorch CUDA, Ultralytics, EasyOCR và Registry/data lịch sử:

```powershell
.\.venv\Scripts\python.exe scripts/run_finetune_experiment.py
.\.venv\Scripts\python.exe scripts/build_finetune_report.py --run "<thu_muc_run_vua_in>"
```

Có thể tách `--phase prepare`, `mtcnn`, `yolo`, `evaluate` cùng `--run` đã prepare. Runner kiểm tra hash source ở các pha tiếp tục. Giữ các checkpoint, manifest và source snapshot cùng nhau; không ghi đè Registry triển khai. Thư mục local **`reports/finetune-10-20261025/`** chứa:

- `protocol_before_training.json`, `evaluation_protocol_before_test.json`, `split_audit.json`: cấu hình và kiểm tra chống giao tập.
- `train_manifest.json`, `validation_manifest.json`, `manifest.json`: danh sách cố định.
- `pnet_history.json`, `onet_history.json`, `yolo_train/results.csv`: đủ 10 epoch và validation.
- `pnet_epoch10.pth`, `onet_epoch10.pth`, `yolo_train/weights/last.pt`: checkpoint được đánh giá.
- `*_predictions.json`, `paired_predictions.csv`, `metrics.csv`, `summary.json`: dự đoán/timing/chênh lệch.
- `source_snapshot.json`, `environment-freeze.txt`: mã và môi trường tái lập.

Hash test manifest `0b0fd71b2fc311ef2dc7ab542098989a41a94114b3145abcf25c762b8cb87b0e` trùng manifest gốc. Hash evaluation protocol `33252c97b7105ca5085560b6cf32de042011ed3c7845f87083b8795ef05dca47`. Hash weight pretrained được kiểm tra không đổi trước test; OCR giữ nguyên. Trích dẫn VNLP theo [tài liệu dữ liệu](data.md): MAPR 2022, DOI [10.1109/MAPR56351.2022.9924802](https://doi.org/10.1109/MAPR56351.2022.9924802), và Pham 2023, DOI [10.1007/s00371-021-02375-0](https://doi.org/10.1007/s00371-021-02375-0).

Biểu đồ cục bộ: `reports/finetune-10-20261025/before_after.png`.

Đường học cục bộ: `reports/finetune-10-20261025/training_curves.png`.
