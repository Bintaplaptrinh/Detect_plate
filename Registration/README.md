# Quản lý phiên bản model

## 1. Cấu trúc lưu trữ

`ModelRegistry("Registration")` quản lý checkpoint và các bản phát hành theo cấu trúc:

```text
Registration/
  Checkpoints/<run_id>/<model>-NNN.joblib
  Checkpoints/<run_id>/<model>-NNN.json
  Registry/<model_name>/vX/model.joblib
  Registry/<model_name>/vX/metadata.json
  Registry/<model_name>/vX/split_manifest.json
  Registry/<model_name>/vX/experiments.json
  Registry/<model_name>/vX/source_snapshot.json
  RuntimeAssets/
```

Checkpoint được ghi khi chỉ số validation cải thiện. Phiên bản được tạo bằng thao tác tạo thư mục độc quyền và được xem là hoàn tất khi có `metadata.json`. Các phiên bản chưa hoàn tất bị bỏ qua khi tìm bản phát hành hợp lệ. Mỗi phiên bản giữ nguyên model, cấu hình, hạt giống ngẫu nhiên, kết quả, manifest chia tập, môi trường và SHA-256 của tài nguyên.

## 2. Các phiên bản của thí nghiệm

| Phiên bản | Nội dung | Cách nạp |
|---|---|---|
| `pipeline/v3` | Detector cơ sở và OCR ký tự kế thừa | `PlateRecognizer.from_bundle` |
| `ocr_vnlp_only/v1` | OCR từ 2.712 ký tự VNLP, 31 lớp | `ModelRegistry.load` |
| `ocr_legacy_plus_vnlp/v1` | OCR kết hợp 4.812 ký tự, 35 lớp | `ModelRegistry.load` |
| `pipeline_vnlp_experiment/v1` | Detector cơ sở và OCR kết hợp | `PlateRecognizer.from_bundle` |
| `research_pipeline/v3` | Detector hai giai đoạn và phân đoạn kết hợp | `load_research_release` |
| `research_glyph_filter/v1` | Phương án nghiên cứu có lọc ký tự | `load_filter_release` |

Hai hàm nạp cấu hình nghiên cứu được định nghĩa trong `scripts/run_research_experiment.py` và `scripts/run_glyph_filter_experiment.py`. Chúng dựng lại các module phù hợp với schema bundle. Bộ phát hành nghiên cứu chứa bundle cơ sở để lưu phụ thuộc; detector hoạt động được xác định bởi cấu hình lựa chọn, không suy ra từ việc bundle có cascade.

## 3. Tái lập và kiểm tra

Thông tin mã nguồn ghi Git revision, trạng thái thay đổi và hash các module. Snapshot Python và tài nguyên thực nghiệm được lưu để đối chiếu khi mã nguồn phát triển tiếp. Kết quả phụ thuộc đồng thời vào dữ liệu, manifest, cấu hình, phiên bản thư viện và snapshot đã sử dụng. Số phiên bản model giống nhau không bảo đảm kết quả tương đương khi các thành phần phụ thuộc khác nhau.

Khi nạp model, cần giữ toàn bộ thư mục phiên bản và sử dụng loader phù hợp. Ví dụ `registry.load("ocr", "v3", trusted=True)` yêu cầu tài nguyên từ nguồn đã được tin cậy, vì joblib có thể thực thi mã khi giải tuần tự hóa. SHA-256 kiểm tra tính toàn vẹn của tài nguyên.

`Checkpoints`, `Registry` và `RuntimeAssets` được loại khỏi Git. Thư mục này chỉ công bố tài liệu cấu trúc. Quy trình chuẩn bị model và thứ tự thí nghiệm được trình bày tại [quy trình tái lập](../docs/reproduction.md).
