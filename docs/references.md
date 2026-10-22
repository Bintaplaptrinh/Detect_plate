# Tài liệu tham khảo

## 1. Dữ liệu VNLP

[1] Thi-Anh-Loan Trinh, The Anh Pham, và Van-Dung Hoang (2022). Layout-invariant license plate detection and recognition. International Conference on Multimedia Analysis and Pattern Recognition, MAPR 2022, trang 1 đến 6. [DOI: 10.1109/MAPR56351.2022.9924802](https://doi.org/10.1109/MAPR56351.2022.9924802).

[2] The-Anh Pham (2023). Effective deep neural networks for license plate detection and recognition. The Visual Computer, 39(3), trang 927 đến 941. [Trang nhà xuất bản](https://link.springer.com/article/10.1007/s00371-021-02375-0). DOI: 10.1007/s00371-021-02375-0.

Cả hai công trình được dẫn theo yêu cầu ghi công của dữ liệu VNLP. Bài [2] xuất bản trực tuyến ngày 21 tháng 1 năm 2022 và thuộc tập 39 năm 2023. Các mô hình neural trong hai công trình không được đưa vào pipeline HOG SVM hiện tại. Khóa BibTeX tương ứng là `TrinhPH22` và `PhamTA23`.

## 2. Phương pháp phát hiện và nhận diện biển số

[3] Lakhdar Djelloul Mazouz, Abdelkrim Meche, Abdelaziz Ouamri, và Abdel Wahab Ait Darna (2024). Two-stage HOG/SVM for license plate detection and recognition. Indonesian Journal of Electrical Engineering and Computer Science, 34(1), trang 210 đến 223. [Bài báo chính thức](https://ijeecs.iaescore.com/index.php/IJEECS/article/download/35854/18119). DOI: 10.11591/ijeecs.v34.i1.pp210-223.

Công trình [3] định hướng detector và OCR hai giai đoạn, kiểm chứng vùng HOG SVM, tiền xử lý và phân đoạn theo histogram. Thí nghiệm 03 điều chỉnh cho biển Việt Nam, chữ cái và bố cục hai hàng. Các chỉ số phân loại crop của bài báo được phân biệt với exact match từ ảnh phương tiện trong [báo cáo 03](research-pipeline-experiment.md).

[4] Temesegan Walelign Ayele, Eyouel Getachew Eshete, Melaku Eneayehu Getnet, Dires Negash Fente, và Rutvik Mehta (2018). License Plate Recognition Using Machine Learning. Journal of Emerging Technologies and Innovative Research, 5(5), trang 375 đến 380. [Bài báo JETIR1805485](https://www.jetir.org/papers/JETIR1805485.pdf).

Công trình [4] được tham khảo cho xử lý độ sáng, giảm nhiễu và biên trước phân đoạn. Báo cáo không sử dụng ảnh minh họa hoặc mô tả 62 ký tự của bài này làm chuẩn định lượng cho VNLP.

## 3. Đặc trưng và mô hình học máy

[5] Navneet Dalal và Bill Triggs (2005). Histograms of Oriented Gradients for Human Detection. IEEE Computer Society Conference on Computer Vision and Pattern Recognition, tập 1, trang 886 đến 893. [Bản tác giả tại INRIA](https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf). DOI: 10.1109/CVPR.2005.177.

[6] Corinna Cortes và Vladimir Vapnik (1995). Support-vector networks. Machine Learning, 20, trang 273 đến 297. [Trang nhà xuất bản](https://link.springer.com/article/10.1007/BF00994018). DOI: 10.1007/BF00994018.

[7] Ji Zhu, Hui Zou, Saharon Rosset, và Trevor Hastie (2009). Multi-class AdaBoost. Statistics and Its Interface, 2(3), trang 349 đến 360. [Bài báo tại nhà xuất bản](https://archive.intlpress.com/site/pub/files/_fulltext/journals/sii/2009/0002/0003/SII-2009-0002-0003-a008.pdf). DOI: 10.4310/SII.2009.v2.n3.a8.

[8] Robert A. Jacobs, Michael I. Jordan, Steven J. Nowlan, và Geoffrey E. Hinton (1991). Adaptive Mixtures of Local Experts. Neural Computation, 3(1), trang 79 đến 87. [Bản tác giả](https://www.cs.toronto.edu/~fritz/absps/jjnh91.html). DOI: 10.1162/neco.1991.3.1.79.

Tài liệu [5] và [6] là cơ sở cho HOG và SVM. Tài liệu [7] được dẫn cho phương án AdaBoost SAMME. Tài liệu [8] cung cấp ý tưởng phân chia chuyên gia; dự án sử dụng gate logistic và SVM theo nhóm lớp với cách huấn luyện riêng.

## 4. Phương pháp trong báo cáo lịch sử

[9] Olaf Ronneberger, Philipp Fischer, và Thomas Brox (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI 2015, Lecture Notes in Computer Science 9351, trang 234 đến 241. [Bản tác giả trên arXiv](https://arxiv.org/abs/1505.04597). DOI: 10.1007/978-3-319-24574-4_28.

Tài liệu [9] mô tả kiến trúc nền của phương án U-Net trong báo cáo năm 2025. Pipeline hiện tại sử dụng học máy cổ điển.

## 5. Thư viện và tài nguyên triển khai

| Nguồn | Vai trò |
|---|---|
| [OpenCV 4.13.0](https://docs.opencv.org/4.13.0/) | Xử lý ảnh, HOG và cascade |
| [Cascade biển Nga, tag 4.13.0](https://github.com/opencv/opencv/blob/4.13.0/data/haarcascades/haarcascade_russian_plate_number.xml) | Tài nguyên của detector cơ sở |
| [scikit-learn 1.8](https://scikit-learn.org/1.8/) | SVM, StandardScaler, AdaBoost và phép chia tập |
| [Tập ký tự License Plate Digits Classification](https://www.kaggle.com/datasets/aladdinss/license-plate-digits-classification-dataset) | Nguồn được ghi trong báo cáo CNN lịch sử; chưa xác nhận đầy đủ lineage của 35.500 ảnh cục bộ |

Giấy phép và hash cascade nằm tại `assets/cascades/`. Danh mục này xác định nguồn phương pháp và tài nguyên, không thay thế giấy phép của từng nguồn.

## 6. Tham khảo cách tổ chức README

[VIETNAMESE_LICENSE_PLATE](https://github.com/mrzaizai2k/VIETNAMESE_LICENSE_PLATE/blob/master/README.md) được tham khảo về cách trình bày các bước xử lý và giải thích phương pháp. [ai-job-search](https://github.com/ai-job-search-dev/ai-job-search/blob/master/README.md) được tham khảo về bố cục tổng quan, cài đặt, cấu trúc và tài liệu liên quan. Hai README này không được sử dụng làm nguồn số liệu thực nghiệm hoặc cơ sở chứng minh thuật toán.

## 7. BibTeX

[references.bib](references.bib) cung cấp bản trích dẫn cho chín công trình khoa học. Khoảng trang trong BibTeX sử dụng cú pháp hai dấu gạch nối theo quy ước của định dạng; phần văn bản báo cáo sử dụng cách ghi trang bằng từ.
