# Báo cáo lịch sử: nhận diện biển số năm 2025

## 1. Phạm vi và thời điểm

Báo cáo tổng hợp các phương án KNN, CNN đầu vào nhị phân, CNN kết hợp U-Net và SVM được ghi nhận ngày 28 tháng 12 năm 2025. Các triển khai này đã ngừng sử dụng trong pipeline hiện tại. Số liệu dưới đây được giữ để mô tả kết quả lịch sử; chưa được tái đánh giá theo các split VNLP năm 2026.

| STT | Họ và tên | Mã số sinh viên |
|---|---|---|
| 1 | Nguyễn Phước Toàn | 2351267278 |
| 2 | Nguyễn Hữu Tuấn Phát | 2351267274 |
| 3 | Trần Mạnh Hùng | 2351267254 |

## 2. Phương án KNN

Tiền xử lý sử dụng kênh V của HSV, top hat và black hat với kernel 3 x 3, Gaussian 5 x 5, adaptive threshold có block 19 và hằng số 9, sau đó Canny với ngưỡng 250 và 255. Vùng biển được tìm theo contour tứ giác và tỷ lệ hình học. Góc xoay được ước lượng từ cạnh dưới của vùng biển.

Ký tự được lọc theo tỷ lệ diện tích từ 0,01 đến 0,09 diện tích crop biển và tỷ lệ rộng trên cao từ 0,25 đến 0,7. Mỗi ký tự được đổi kích thước thành 20 x 30 pixel, trải thành 600 đặc trưng nhị phân và phân loại bằng KNN với k bằng 3. Nhãn ký tự được biểu diễn bằng mã ASCII.

Báo cáo lịch sử gồm ảnh minh họa nhận diện nhưng chưa có tập kiểm tra, cỡ mẫu hoặc bảng chỉ số đủ để kết luận định lượng về độ chính xác toàn biển.

## 3. Phương án CNN đầu vào nhị phân

Ảnh nhị phân được đưa về 64 x 64 pixel, 3 kênh. Kiến trúc gồm các lớp tích chập, BatchNorm, ReLU, pooling và lớp phân loại 35 đầu ra. Báo cáo ghi nguồn dữ liệu là License Plate Digits Classification Dataset trên Kaggle.

| Tham số | Giá trị đã ghi nhận |
|---|---|
| Tỷ lệ train và validation | 80% và 20% |
| Batch size | 32 |
| Optimizer | Adam, learning rate 0,001 |
| Hàm mất mát | Cross entropy |
| Số epoch tối đa | 50 |
| Early stopping | Patience 10 |

| Epoch | Train loss | Train accuracy | Validation loss | Validation accuracy |
|---|---:|---:|---:|---:|
| 1 | 2,1381 | 32,87% | 3,1243 | 15,39% |
| 10 | 0,9296 | 72,90% | 0,9443 | 72,63% |
| 20 | 0,4823 | 85,99% | 0,7102 | 78,00% |
| 27 | 0,3377 | 90,04% | 0,3186 | 90,25% |
| 36 | 0,2137 | 93,83% | 0,4937 | 84,41% |

Validation accuracy cao nhất được ghi là 90,25% tại epoch 27; huấn luyện dừng ở epoch 36. Một giá trị cuối 87,86% được ghi trong báo cáo gốc nhưng chưa xác định rõ tập đánh giá, nên không được trình bày như test accuracy độc lập. Mô tả 35 lớp trong báo cáo gốc chưa khớp danh sách ký hiệu loại trừ I, O và Q; bảng lớp lịch sử cần được xác minh bằng nhãn huấn luyện trước khi tái sử dụng.

## 4. Phương án CNN và U-Net

U-Net Small dự đoán mask biển số từ ảnh màu. Các khối DoubleConv gồm hai lớp tích chập 3 x 3 với BatchNorm và ReLU. Cấu hình kênh cơ sở là 32; các mức encoder là 32, 64 và 128 kênh, bottleneck 256 kênh, decoder tương ứng 128, 64 và 32 kênh với skip connection. Lớp đầu ra 1 x 1 tạo mask một kênh. Kiến trúc U-Net nền được dẫn tại [9] trong [tài liệu tham khảo](references.md).

Vùng biển được hiệu chỉnh phối cảnh. Ký tự được tìm bằng MSER, contour hoặc kết hợp hai phương án. CNN nhận ảnh màu 64 x 64 pixel, 3 kênh và có 35 đầu ra. Ràng buộc vị trí giới hạn hai ký tự đầu là chữ số và vị trí tiếp theo là chữ cái trong mẫu khuôn đã cấu hình.

Báo cáo có ảnh minh họa mask, vùng hiệu chỉnh và OCR, nhưng chưa có chỉ số trên tập kiểm tra độc lập để xác nhận hiệu quả của phát hiện hoặc nhận diện toàn biển.

## 5. Phương án SVM

Đặc trưng HOG lịch sử dùng ảnh 30 x 60 pixel, block 10 x 20, bước block 5 x 10, cell 5 x 10 và 9 khoảng hướng. Cấu hình này khác với HOG 540 chiều trên canvas 32 x 48 của pipeline năm 2026.

| Tham số hoặc kết quả | Giá trị đã ghi nhận |
|---|---|
| Số mẫu | 1.047 |
| Tỷ lệ train và test | 80% và 20% |
| Kernel | Tuyến tính |
| C | 1,0 |
| Random state | 42 |
| Số lớp trong bảng so sánh lịch sử | 31 |
| Accuracy ký tự được báo cáo | 97,14% |

Accuracy 97,14% áp dụng cho bài toán ký tự đã cắt. Báo cáo chưa ghi đầy đủ quan hệ nguồn, mẫu số test sau làm tròn hoặc phép đánh giá toàn biển. Chỉ số không được so sánh trực tiếp với kết quả toàn ảnh VNLP năm 2026.

## 6. Đối chiếu và giới hạn

| Phương án | Đầu vào OCR | Điểm cần xác minh |
|---|---|---|
| KNN | Nhị phân 20 x 30 | Chưa có phép đo trên tập kiểm tra được mô tả đầy đủ |
| CNN nhị phân | 64 x 64, 3 kênh | Danh sách lớp và phạm vi chỉ số cuối 87,86% |
| CNN và U-Net | Ảnh màu 64 x 64 | Chỉ số phát hiện, phân đoạn và toàn biển chưa được báo cáo |
| SVM | HOG trên ảnh 30 x 60 | Nguồn mẫu ký tự và phép đo toàn biển |

Bốn phương án chưa có một tập kiểm tra toàn biển chung nên báo cáo không xếp hạng định lượng giữa chúng. Hạn chế chung là phân đoạn phụ thuộc ngưỡng hình học, ảnh nhiễu và vùng biển nghiêng hoặc dính viền. Các hướng CRNN, detector neural hoặc xử lý video từng được đề xuất là hướng nghiên cứu, chưa phải thành phần đã được kiểm chứng trong báo cáo này.

## 7. Tài liệu và ảnh minh họa

Cơ sở HOG, SVM và U-Net được dẫn tại [5], [6] và [9] trong [danh mục tham khảo](references.md). Nguồn tập ký tự được ghi ở mục tài nguyên triển khai của danh mục đó; quan hệ đầy đủ giữa tập công khai và ảnh cục bộ chưa được xác minh.

| Sơ đồ lịch sử | Tài nguyên |
|---|---|
| Các bước nhận diện | [Sơ đồ tổng quát](../img/download.png) |
| Phát hiện và cắt biển | [Sơ đồ xử lý ảnh](../img/download%20%281%29.png) |
| CNN ký tự | [Kiến trúc CNN](../img/cnnarchitecture.png) |
| U-Net Small | [Kiến trúc U-Net](../img/unet_small.png) |
| CNN và hiệu chỉnh phối cảnh | [Pipeline CNN](../img/cnnpipeline.png) |
| HOG và SVM | [Pipeline SVM](../img/svmarchitecture.png) |

Ảnh minh họa từng bước được giữ tại các nhóm dưới đây. Ảnh minh họa thể hiện trường hợp cụ thể và không thay thế phép đo trên tập kiểm tra.

### KNN

1. [Ảnh 1](<legacy-evidence/KNN/image.png>)
2. [Ảnh 2](<legacy-evidence/KNN/Screenshot%202025-12-28%20125511.png>)
3. [Ảnh 3](<legacy-evidence/KNN/Screenshot%202025-12-28%20125559.png>)
4. [Ảnh 4](<legacy-evidence/KNN/Screenshot%202025-12-28%20125616.png>)
5. [Ảnh 5](<legacy-evidence/KNN/Screenshot%202025-12-28%20130004.png>)
6. [Ảnh 6](<legacy-evidence/KNN/Screenshot%202025-12-28%20130007.png>)
7. [Ảnh 7](<legacy-evidence/KNN/Screenshot%202025-12-28%20130012.png>)
8. [Ảnh 8](<legacy-evidence/KNN/Screenshot%202025-12-28%20130016.png>)
9. [Ảnh 9](<legacy-evidence/KNN/Screenshot%202025-12-28%20130021.png>)
10. [Ảnh 10](<legacy-evidence/KNN/Screenshot%202025-12-28%20130023.png>)
11. [Ảnh 11](<legacy-evidence/KNN/Screenshot%202025-12-28%20130051.png>)

### CNN_threshold

1. [Ảnh 1](<legacy-evidence/CNN_threshold/Screenshot%202025-12-28%20132105.png>)
2. [Ảnh 2](<legacy-evidence/CNN_threshold/Screenshot%202025-12-28%20132107.png>)
3. [Ảnh 3](<legacy-evidence/CNN_threshold/Screenshot%202025-12-28%20132109.png>)
4. [Ảnh 4](<legacy-evidence/CNN_threshold/Screenshot%202025-12-28%20132226.png>)
5. [Ảnh 5](<legacy-evidence/CNN_threshold/Screenshot%202025-12-28%20132242.png>)
6. [Ảnh 6](<legacy-evidence/CNN_threshold/Screenshot%202025-12-28%20132249.png>)
7. [Ảnh 7](<legacy-evidence/CNN_threshold/Screenshot%202025-12-28%20132257.png>)
8. [Ảnh 8](<legacy-evidence/CNN_threshold/Screenshot%202025-12-28%20132305.png>)
9. [Ảnh 9](<legacy-evidence/CNN_threshold/Screenshot%202025-12-28%20132311.png>)
10. [Ảnh 10](<legacy-evidence/CNN_threshold/Screenshot%202025-12-28%20132313.png>)

### CNN

1. [Ảnh 1](<legacy-evidence/CNN/Screenshot%202025-12-28%20130434.png>)
2. [Ảnh 2](<legacy-evidence/CNN/Screenshot%202025-12-28%20130438.png>)
3. [Ảnh 3](<legacy-evidence/CNN/Screenshot%202025-12-28%20130442.png>)
4. [Ảnh 4](<legacy-evidence/CNN/Screenshot%202025-12-28%20130444.png>)
5. [Ảnh 5](<legacy-evidence/CNN/Screenshot%202025-12-28%20130446.png>)
6. [Ảnh 6](<legacy-evidence/CNN/Screenshot%202025-12-28%20130448.png>)
7. [Ảnh 7](<legacy-evidence/CNN/Screenshot%202025-12-28%20131909.png>)
8. [Ảnh 8](<legacy-evidence/CNN/Screenshot%202025-12-28%20131914.png>)
9. [Ảnh 9](<legacy-evidence/CNN/Screenshot%202025-12-28%20131920.png>)
10. [Ảnh 10](<legacy-evidence/CNN/Screenshot%202025-12-28%20131925.png>)
11. [Ảnh 11](<legacy-evidence/CNN/Screenshot%202025-12-28%20131930.png>)
12. [Ảnh 12](<legacy-evidence/CNN/Screenshot%202025-12-28%20131943.png>)

### SVM

1. [Ảnh 1](<legacy-evidence/SVM/output.png>)
2. [Ảnh 2](<legacy-evidence/SVM/output1.png>)
3. [Ảnh 3](<legacy-evidence/SVM/output2.png>)
4. [Ảnh 4](<legacy-evidence/SVM/output3.png>)
5. [Ảnh 5](<legacy-evidence/SVM/output4.png>)
6. [Ảnh 6](<legacy-evidence/SVM/output5.png>)
7. [Ảnh 7](<legacy-evidence/SVM/output6.png>)
8. [Ảnh 8](<legacy-evidence/SVM/output7.png>)
9. [Ảnh 9](<legacy-evidence/SVM/output8.png>)
10. [Ảnh 10](<legacy-evidence/SVM/output9.png>)
