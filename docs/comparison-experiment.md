# Đối sánh hệ thống và OCR sau fine-tune trên 100 ảnh VNLP

Lần test hoàn tất: **25/10/2026 10:37:01 UTC+7**. Báo cáo dùng duy nhất HOG/SVM đang triển khai, MTCNN đã fine-tune PNet/ONet 10 epoch và YOLO26n đã fine-tune 10 epoch. Cả nhận diện toàn hệ thống và OCR trên crop nhãn thật đều được **chạy lại 3 lượt trên đủ 100 ảnh**, không lấy lại số liệu OCR từ báo cáo khác.

Đúng toàn biển khi chạy toàn hệ thống: **HOG/SVM 45/100**, **MTCNN + STN/LPRNet 0/100**, **YOLO26n + EasyOCR 20/100**. Khi cấp crop nhãn thật cho OCR: lần lượt **36/100**, **0/100**, **30/100**.

## Dữ liệu và mô hình được kiểm tra

Giữ đúng 100 ảnh test C: **34 biển một hàng, 33 biển hai hàng ô tô và 33 biển hai hàng xe máy**. Detector được fine-tune trên **951 ảnh train**, theo dõi bằng **90 ảnh validation**. Kiểm tra lại đường dẫn, định danh biển và hash pixel giải mã: train/validation/test không giao định danh hoặc pixel. 100 ảnh test không dùng để cập nhật trọng số hay chọn ngưỡng. Đây là tập test đã được sử dụng trong quá trình nghiên cứu, không phải holdout độc lập mới theo camera/phiên ghi hình.

| Nhánh | Trọng số được nạp để test | OCR |
|---|---|---|
| HOG/SVM hiện tại | `research_glyph_filter/v1` | Phân đoạn hybrid, lọc ký tự, HOG/SVM và hậu xử lý hiện tại |
| MTCNN fine-tune | `pnet_epoch10.pth` và `onet_epoch10.pth` | STN/LPRNet của repo xuexingyu24 |
| YOLO26n fine-tune | `yolo_train/weights/last.pt`, sau đủ epoch 10 | Tiền xử lý và English EasyOCR của project Esteebaan23 |

Checkpoint detector lấy tại epoch 10 cuối cùng. PNet và ONet mỗi mạng đã được huấn luyện đủ 10 epoch. **OCR không được fine-tune trong thí nghiệm detector**; bảng OCR bên dưới là phép đo mới của OCR đang gắn với các mô hình hiện tại. Vehicle segmentation của nhánh YOLO giữ nguyên. Hash của mọi trọng số hoạt động được kiểm tra trước và sau test trong protocol.

## Độ chính xác khi chạy toàn hệ thống

| Hệ thống | Ảnh | Biển đúng | Exact match | CER ↓ | Đúng cả vị trí và chuỗi |
| --- | ---: | ---: | ---: | ---: | ---: |
| HOG/SVM hiện tại | 100 | 45/100 | 45.00% | 26.78% | 42.00% |
| MTCNN fine-tune + STN/LPRNet | 100 | 0/100 | 0.00% | 95.95% | 0.00% |
| YOLO26n fine-tune + EasyOCR | 100 | 20/100 | 20.00% | 35.26% | 20.00% |

Exact match yêu cầu chuỗi đầu ra đúng hoàn toàn so với nhãn; CER = tổng khoảng cách Levenshtein / tổng ký tự nhãn. Ảnh không phát hiện hoặc không đủ điều kiện gọi OCR được tính là chuỗi rỗng. Cột cuối còn yêu cầu box biển được chọn đạt **IoU ≥ 0,5**, giúp phân biệt chuỗi đúng với nhận diện đúng cả vùng biển. Nhãn test không được truyền vào predictor để chọn box hoặc sửa chuỗi.

## Phát hiện vùng biển: TP, FP, FN, Precision, Recall, F1

Mỗi ảnh có một box nhãn. Bảng chính dùng **box cuối cùng được chọn**: HOG/SVM theo reranking, MTCNN/YOLO theo confidence detector. Box đúng ở IoU ≥ 0,5 là TP; box sai là FP và để lại một FN; không có box là FN. Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1 = 2PR/(P+R). Không có ảnh âm nên không báo TN, specificity hoặc accuracy nhị phân.

| Hệ thống | TP | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HOG/SVM hiện tại | 73 | 27 | 27 | 73.00% | 73.00% | 73.00% |
| MTCNN fine-tune + STN/LPRNet | 64 | 13 | 36 | 83.12% | 64.00% | 72.32% |
| YOLO26n fine-tune + EasyOCR | 99 | 0 | 1 | 100.00% | 99.00% | 99.50% |

Đối với **toàn bộ ứng viên sau ngưỡng**, ghép box một-một với nhãn. HOG/SVM giữ nhiều ứng viên để OCR rerank nên bảng này có thể có precision thấp hơn box cuối cùng. AP nội suy 101 điểm; AP50:95 trung bình 10 ngưỡng từ 0,50 đến 0,95, không phải toàn bộ COCO evaluator theo area/crowd/maxDets.

| Hệ thống | TP | FP | FN | Precision | Recall | F1 | AP50 | AP50:95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HOG/SVM hiện tại | 78 | 1073 | 22 | 6.78% | 78.00% | 12.47% | 64.68% | 33.80% |
| MTCNN fine-tune + STN/LPRNet | 66 | 47 | 34 | 58.41% | 66.00% | 61.97% | 62.08% | 32.40% |
| YOLO26n fine-tune + EasyOCR | 99 | 0 | 1 | 100.00% | 99.00% | 99.50% | 99.01% | 75.34% |

## Nhận diện đúng cả vị trí và chuỗi: TP, FP, FN

TP yêu cầu **box IoU ≥ 0,5 và chuỗi đúng hoàn toàn**. Một dự đoán có text nhưng sai vị trí hoặc sai chữ tạo FP và FN; không có text tạo FN. Chỉ tính kết quả biển cuối cùng được phát ra, không gộp mọi ứng viên OCR.

| Hệ thống | TP | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HOG/SVM hiện tại | 42 | 58 | 58 | 42.00% | 42.00% | 42.00% |
| MTCNN fine-tune + STN/LPRNet | 0 | 77 | 100 | 0.00% | 0.00% | 0.00% |
| YOLO26n fine-tune + EasyOCR | 20 | 70 | 80 | 22.22% | 20.00% | 21.05% |

## Đối sánh OCR trên crop nhãn thật, chạy lại độc lập

Cắt cùng box nhãn thật từ từng ảnh và đưa pixel crop vào OCR của cả ba nhánh. Không chạy detector, không dùng gate confidence 0,75 của YOLO trong phép đo này. OCR gồm các bước tiền xử lý/phân đoạn/nhận dạng/hậu xử lý của nhánh tương ứng. Bbox nhãn chỉ được dùng để tạo crop; chuỗi nhãn chỉ dùng sau suy luận để chấm điểm. Đây là chẩn đoán OCR, không được cộng vào kết quả toàn hệ thống.

| OCR của hệ thống | Crop | Biển đúng | Exact match | CER ↓ | Ký tự đúng / ký tự nhãn |
| --- | ---: | ---: | ---: | ---: | ---: |
| HOG/SVM hiện tại | 100 | 36/100 | 36.00% | 23.83% | 79.73% |
| MTCNN fine-tune + STN/LPRNet | 100 | 0/100 | 0.00% | 96.68% | 5.04% |
| YOLO26n fine-tune + EasyOCR | 100 | 30/100 | 30.00% | 23.83% | 79.61% |

Ký tự đúng được đếm sau căn chỉnh Levenshtein trên chuỗi đầu ra; tỷ lệ ở cột cuối là **C/N**, không phải 1−CER khi có ký tự chèn. Với căn chỉnh này: TP = ký tự khớp; FP = thay thế + chèn; FN = thay thế + xóa. Nếu có nhiều căn chỉnh tối ưu, ưu tiên đường chéo, rồi xóa, rồi chèn; quy tắc được giữ cố định cho mọi nhánh. Đây là micro P/R/F1 của ký tự theo căn chỉnh, không phải macro-F1 của bộ phân loại 35 lớp.

| Hệ thống | TP | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HOG/SVM hiện tại | 649 | 109 | 165 | 85.62% | 79.73% | 82.57% |
| MTCNN fine-tune + STN/LPRNet | 41 | 459 | 773 | 8.20% | 5.04% | 6.24% |
| YOLO26n fine-tune + EasyOCR | 648 | 149 | 166 | 81.30% | 79.61% | 80.45% |

Chuỗi EasyOCR được viết hoa và bỏ khoảng trắng/dấu phân cách để khớp định dạng nhãn VNLP. Không thay O/0, I/1 hay loại ký tự Trung Quốc để làm đẹp điểm. Raw OCR cũng được lưu riêng. Với EasyOCR, adapter giữ bản sửa lỗi tọa độ ghép dòng: lấy y trung bình của bốn góc thay cho phép trừ tuple trong mã nguồn project. STN/LPRNet giữ decoder upstream, bao gồm cách suppression timestep đầu.

## Toàn hệ thống theo từng bố cục test

| Bố cục | Hệ thống | Ảnh | Biển đúng | Exact match | CER ↓ | Đúng vị trí và chuỗi |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Một hàng | HOG/SVM hiện tại | 34 | 23/34 | 67.65% | 15.99% | 58.82% |
| Một hàng | MTCNN fine-tune + STN/LPRNet | 34 | 0/34 | 0.00% | 88.85% | 0.00% |
| Một hàng | YOLO26n fine-tune + EasyOCR | 34 | 15/34 | 44.12% | 21.93% | 44.12% |
| Hai hàng ô tô | HOG/SVM hiện tại | 33 | 10/33 | 30.30% | 39.08% | 30.30% |
| Hai hàng ô tô | MTCNN fine-tune + STN/LPRNet | 33 | 0/33 | 0.00% | 100.00% | 0.00% |
| Hai hàng ô tô | YOLO26n fine-tune + EasyOCR | 33 | 4/33 | 12.12% | 41.00% | 12.12% |
| Hai hàng xe máy | HOG/SVM hiện tại | 33 | 12/33 | 36.36% | 25.70% | 36.36% |
| Hai hàng xe máy | MTCNN fine-tune + STN/LPRNet | 33 | 0/33 | 0.00% | 98.94% | 0.00% |
| Hai hàng xe máy | YOLO26n fine-tune + EasyOCR | 33 | 1/33 | 3.03% | 42.61% | 3.03% |

## Phát hiện vùng biển theo từng bố cục

| Bố cục | Hệ thống | TP | FP | FN | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Một hàng | HOG/SVM hiện tại | 25 | 9 | 9 | 73.53% | 73.53% | 73.53% |
| Một hàng | MTCNN fine-tune + STN/LPRNet | 33 | 0 | 1 | 100.00% | 97.06% | 98.51% |
| Một hàng | YOLO26n fine-tune + EasyOCR | 34 | 0 | 0 | 100.00% | 100.00% | 100.00% |
| Hai hàng ô tô | HOG/SVM hiện tại | 21 | 12 | 12 | 63.64% | 63.64% | 63.64% |
| Hai hàng ô tô | MTCNN fine-tune + STN/LPRNet | 10 | 10 | 23 | 50.00% | 30.30% | 37.74% |
| Hai hàng ô tô | YOLO26n fine-tune + EasyOCR | 32 | 0 | 1 | 100.00% | 96.97% | 98.46% |
| Hai hàng xe máy | HOG/SVM hiện tại | 27 | 6 | 6 | 81.82% | 81.82% | 81.82% |
| Hai hàng xe máy | MTCNN fine-tune + STN/LPRNet | 21 | 3 | 12 | 87.50% | 63.64% | 73.68% |
| Hai hàng xe máy | YOLO26n fine-tune + EasyOCR | 33 | 0 | 0 | 100.00% | 100.00% | 100.00% |

## Nhận diện đúng vị trí và chuỗi theo từng bố cục

| Bố cục | Hệ thống | TP | FP | FN | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Một hàng | HOG/SVM hiện tại | 20 | 14 | 14 | 58.82% | 58.82% | 58.82% |
| Một hàng | MTCNN fine-tune + STN/LPRNet | 0 | 33 | 34 | 0.00% | 0.00% | 0.00% |
| Một hàng | YOLO26n fine-tune + EasyOCR | 15 | 14 | 19 | 51.72% | 44.12% | 47.62% |
| Hai hàng ô tô | HOG/SVM hiện tại | 10 | 23 | 23 | 30.30% | 30.30% | 30.30% |
| Hai hàng ô tô | MTCNN fine-tune + STN/LPRNet | 0 | 20 | 33 | 0.00% | 0.00% | 0.00% |
| Hai hàng ô tô | YOLO26n fine-tune + EasyOCR | 4 | 25 | 29 | 13.79% | 12.12% | 12.90% |
| Hai hàng xe máy | HOG/SVM hiện tại | 12 | 21 | 21 | 36.36% | 36.36% | 36.36% |
| Hai hàng xe máy | MTCNN fine-tune + STN/LPRNet | 0 | 24 | 33 | 0.00% | 0.00% | 0.00% |
| Hai hàng xe máy | YOLO26n fine-tune + EasyOCR | 1 | 31 | 32 | 3.12% | 3.03% | 3.08% |

## OCR trên crop nhãn thật theo từng bố cục

| Bố cục | OCR của hệ thống | Crop | Biển đúng | Exact match | CER ↓ | Ký tự đúng / nhãn |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Một hàng | HOG/SVM hiện tại | 34 | 21/34 | 61.76% | 12.64% | 89.59% |
| Một hàng | MTCNN fine-tune + STN/LPRNet | 34 | 0/34 | 0.00% | 91.08% | 10.78% |
| Một hàng | YOLO26n fine-tune + EasyOCR | 34 | 20/34 | 58.82% | 5.95% | 95.17% |
| Hai hàng ô tô | HOG/SVM hiện tại | 33 | 7/33 | 21.21% | 29.50% | 73.18% |
| Hai hàng ô tô | MTCNN fine-tune + STN/LPRNet | 33 | 0/33 | 0.00% | 98.47% | 1.92% |
| Hai hàng ô tô | YOLO26n fine-tune + EasyOCR | 33 | 7/33 | 21.21% | 26.82% | 76.63% |
| Hai hàng xe máy | HOG/SVM hiện tại | 33 | 8/33 | 24.24% | 29.23% | 76.41% |
| Hai hàng xe máy | MTCNN fine-tune + STN/LPRNet | 33 | 0/33 | 0.00% | 100.35% | 2.46% |
| Hai hàng xe máy | YOLO26n fine-tune + EasyOCR | 33 | 3/33 | 9.09% | 38.03% | 67.61% |

TP/FP/FN và Precision/Recall/F1 **ký tự OCR** theo từng bố cục:

| Bố cục | Hệ thống | TP | FP | FN | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Một hàng | HOG/SVM hiện tại | 241 | 9 | 28 | 96.40% | 89.59% | 92.87% |
| Một hàng | MTCNN fine-tune + STN/LPRNet | 29 | 132 | 240 | 18.01% | 10.78% | 13.49% |
| Một hàng | YOLO26n fine-tune + EasyOCR | 256 | 13 | 13 | 95.17% | 95.17% | 95.17% |
| Hai hàng ô tô | HOG/SVM hiện tại | 191 | 34 | 70 | 84.89% | 73.18% | 78.60% |
| Hai hàng ô tô | MTCNN fine-tune + STN/LPRNet | 5 | 146 | 256 | 3.31% | 1.92% | 2.43% |
| Hai hàng ô tô | YOLO26n fine-tune + EasyOCR | 200 | 59 | 61 | 77.22% | 76.63% | 76.92% |
| Hai hàng xe máy | HOG/SVM hiện tại | 217 | 66 | 67 | 76.68% | 76.41% | 76.54% |
| Hai hàng xe máy | MTCNN fine-tune + STN/LPRNet | 7 | 181 | 277 | 3.72% | 2.46% | 2.97% |
| Hai hàng xe máy | YOLO26n fine-tune + EasyOCR | 192 | 77 | 92 | 71.38% | 67.61% | 69.44% |

Các chỉ số trên được lưu đầy đủ trong `metrics_by_layout.csv` và JSON tổng hợp; mỗi dòng tương ứng một nhánh và một bố cục. Không loại ảnh lỗi khỏi mẫu số.

## Độ trễ đo mới trên CPU

CPU **AMD Ryzen 7 7700**, Windows, batch1, một luồng OpenCV/PyTorch/BLAS. Mỗi tác vụ warm-up bằng ba ảnh validation, chạy 3 lượt đủ 100 ảnh, đổi thứ tự nhánh giữa các lượt. Lấy median các lượt cho từng ảnh, sau đó tính mean/median/p95 trên 100 ảnh; FPS = 1000/mean_ms. Đọc ảnh, cắt crop GT và nạp checkpoint nằm ngoài timer. System timer gồm detector và các bước OCR thực sự được gọi; OCR timer chỉ gồm pipeline OCR trên crop GT. Kiểm tra số luồng sau warm-up và sau từng suy luận.

| Hệ thống | System mean ms | System median ms | System P95 ms | System FPS | OCR mean ms/crop | OCR P95 ms/crop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HOG/SVM hiện tại | 185.49 | 176.15 | 275.12 | 5.39 | 40.04 | 64.63 |
| MTCNN fine-tune + STN/LPRNet | 263.54 | 225.47 | 482.64 | 3.79 | 10.71 | 11.04 |
| YOLO26n fine-tune + EasyOCR | 262.01 | 229.64 | 512.74 | 3.82 | 166.22 | 340.18 |

Giữ cấu hình MTCNN min plate 50×15, ngưỡng PNet 0,6 / ONet 0,8, OCR resize 94×24. YOLO giữ infer_scale 0,5, plate input 416, conf 0,30, OCR gate 0,75 và crop ít nhất 12×30; ROI và car-only tắt; vẫn chạy vehicle segmentation. Trong toàn hệ thống, **91/100 ảnh** được gọi EasyOCR. Các lần không gọi OCR vẫn nằm trong trung bình system latency. YOLO/LPRNet dùng FP32, EasyOCR giữ quantization CPU mặc định. GPU không tham gia suy luận trong bảng này.

## Diễn giải và giới hạn

Kết quả vùng biển và chuỗi biển đo hai mức khác nhau. MTCNN đã thích ứng detector nhưng STN/LPRNet vẫn là OCR pretrained cho biển Trung Quốc; YOLO26n còn phụ thuộc English EasyOCR và các điều kiện gọi OCR. Cấp box đúng không bảo đảm OCR crop GT là cận trên toán học, vì pipeline toàn ảnh có thể hiệu chỉnh/rerank ra crop khác. Tập có một biển được gán nhãn mỗi ảnh và không có ảnh âm; chưa suy diễn khả năng tổng quát hóa camera độc lập hoặc tần suất báo giả ở cảnh không có biển.

## Tái lập và nguồn kết quả hiện tại

```powershell
.\.venv\Scripts\python.exe scripts/run_post_finetune_evaluation.py --training-run reports/finetune-10-20261025
.\.venv\Scripts\python.exe scripts/build_comparison_report.py --run "<thu_muc_retest_vua_in>"
```

Runner chỉ nạp ba nhánh hiện tại và không huấn luyện thêm. MTCNN, YOLO26n và OCR khởi tạo từ [xuexingyu24/License_Plate_Detection_Pytorch](https://github.com/xuexingyu24/License_Plate_Detection_Pytorch/tree/aa5778d6e6da0f382894048b4199cca7a1151e95) và [Esteebaan23/Vehicle-License-Plate-Detection-with-YOLO26](https://github.com/Esteebaan23/Vehicle-License-Plate-Detection-with-YOLO26/tree/25f9099a8b1af72c8d481e8bf1ec62e586b98887); checkpoint detector epoch 10 là kết quả fine-tune VNLP cục bộ. Không chọn lại checkpoint/ngưỡng theo test. PyTorch `2.10.0+cu130`, OpenCV `4.13.0`, NumPy `2.4.1`; môi trường đầy đủ ở `environment-freeze.txt`.

Đầu ra lần test này: `reports/post-finetune-retest-100-20261025/`.

- [JSON tổng hợp hiện tại](results/comparison-100-20261025.json): chỉ chứa ba nhánh đang được đối sánh, gồm OCR/system và mọi bố cục.
- [CSV toàn bộ chỉ số](../reports/post-finetune-retest-100-20261025/metrics.csv), [CSV theo bố cục](../reports/post-finetune-retest-100-20261025/metrics_by_layout.csv), [chuỗi dự đoán từng ảnh](../reports/post-finetune-retest-100-20261025/paired_predictions.csv).
- `*_predictions.json`: box/confidence, text và thời gian từng lượt của toàn hệ thống; `*_ocr_gt_predictions.json`: OCR crop GT mới.
- `protocol_before_test.json`, `split_audit.json`, `verification.json`: hash checkpoint/manifest/source, chống giao tập và kiểm tra suy luận; `source_snapshot.json` giữ mã nguồn của lần chạy.
- [Biểu đồ kết quả hiện tại](../reports/post-finetune-retest-100-20261025/comparison.png). Ảnh, trọng số và chi tiết trong `reports/` lưu cục bộ.

Hash protocol: `ab9f214b8c33681864672d1ccdee97b56325a310b9e900c538e8816286bf130a`. Hash manifest test: `0b0fd71b2fc311ef2dc7ab542098989a41a94114b3145abcf25c762b8cb87b0e`. Không dùng kết quả test đã lưu để điền các bảng mới; tất cả số liệu system và OCR đều tính từ lần chạy này.

VNLP: Trinh và cộng sự, MAPR2022, DOI [10.1109/MAPR56351.2022.9924802](https://doi.org/10.1109/MAPR56351.2022.9924802); Pham, The Visual Computer2023, DOI [10.1007/s00371-021-02375-0](https://doi.org/10.1007/s00371-021-02375-0). Xem [dữ liệu](data.md) và [BibTeX](references.bib).
