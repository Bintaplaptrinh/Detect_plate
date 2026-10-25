"""Build the ten-epoch detector adaptation report from recorded predictions."""
import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = dict(current_hog_svm='HOG/SVM hiện tại', mtcnn_before='MTCNN trước fine-tune',
             mtcnn_after='MTCNN sau 10 epoch', yolo_before='YOLO26n trước fine-tune',
             yolo_after='YOLO26n sau 10 epoch')


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def build(run):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    summary = load(run / 'summary.json')
    protocol = load(run / 'protocol_before_training.json')
    audit = load(run / 'split_audit.json')
    metrics = summary['metrics']
    assert all(n == 10 for n in summary['epochs'].values())
    table, records = [], []
    for name, m in metrics.items():
        d, timing = m['detector_selected'], m['latency']
        table.append(f"| {NAMES[name]} | {m['exact_count']}/100 | {100*m['decoded']['character_error_rate']:.2f}% | "
                     f"{100*d['precision']:.2f}% | {100*d['recall']:.2f}% | {100*d['f1']:.2f}% | "
                     f"{100*m['ap50']:.2f}% | {timing['mean_ms']:.2f} | {timing['serial_fps']:.2f} |")
        records.append(dict(model=name, exact_count=m['exact_count'], exact_match=m['decoded']['exact_match'],
            cer=m['decoded']['character_error_rate'], precision=d['precision'], recall=d['recall'], f1=d['f1'],
            all_recall=m['detector_all']['recall'], ap50=m['ap50'], map50_95=m['map50_95'],
            mean_ms=timing['mean_ms'], median_ms=timing['median_ms'], p95_ms=timing['p95_ms'], fps=timing['serial_fps']))
    with (run / 'metrics.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    predictions = {k: load(run / f'{k}_predictions.json') for k in metrics}
    paired = []
    for i, scene in enumerate(load(run / 'manifest.json')):
        row = dict(path=scene['path'], layout=scene['layout'], truth=scene['text'])
        for key in metrics:
            assert predictions[key][i]['path'] == scene['path']
            row[key] = predictions[key][i]['text']
        paired.append(row)
    with (run / 'paired_predictions.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired[0])); writer.writeheader(); writer.writerows(paired)
    training = {name: load(run / f'{name}_training.json') for name in ['pnet', 'onet', 'yolo']}
    mining = load(run / 'hard_mining_audit.json')
    differences = []
    for name, value in summary['paired_after_minus_before'].items():
        ci = value['exact_match_delta_ci95']
        differences.append(f"| {name} | {100*value['exact_match_delta']:.2f} | [{100*ci[0]:.2f}; {100*ci[1]:.2f}] | "
                           f"{value['candidate_only_correct']} | {value['current_only_correct']} | {value['mcnemar_exact_p']:.6g} |")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout='constrained')
    x = np.arange(2)
    for ax, field, title in zip(axes, ['exact', 'recall', 'latency'],
                               ['Exact plate match (%)', 'Selected plate recall at IoU 0.5 (%)', 'CPU mean latency (ms/image)']):
        def value(name):
            m = metrics[name]
            return 100*m['decoded']['exact_match'] if field == 'exact' else (
                100*m['detector_selected']['recall'] if field == 'recall' else m['latency']['mean_ms'])
        for offset, stage, color in [(-.18, 'before', '#94A3B8'), (.18, 'after', '#177E89')]:
            bars = ax.bar(x+offset, [value(f'{name}_{stage}') for name in ['mtcnn', 'yolo']], .34, label=stage, color=color)
            ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=9)
        ax.axhline(value('current_hog_svm'), color='#DE7A22', linestyle='--', linewidth=1.2, label='HOG/SVM')
        ax.set_xticks(x, ['MTCNN + STN/LPRNet', 'YOLO26n + EasyOCR'])
        ax.tick_params(axis='x', labelsize=8)
        ax.set_title(title, fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_ylim(0, max(value(k) for k in metrics)*1.25+1)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle('10-epoch detector fine-tuning: same 100 VNLP test images, OCR unchanged', fontsize=12)
    fig.savefig(run / 'before_after.png', dpi=180, bbox_inches='tight'); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout='constrained')
    for ax, stage in zip(axes[:2], ['pnet', 'onet']):
        history = load(run / f'{stage}_history.json')
        for partition in ['train', 'validation']:
            ax.plot([r['epoch'] for r in history], [r[partition]['loss'] for r in history], marker='.', label=partition)
        ax.set_title(f'{stage.upper()}: weighted loss'); ax.set_xlabel('Epoch'); ax.legend(frameon=False)
    yolo_history = [{k.strip(): v for k, v in row.items()} for row in training['yolo']['history']]
    for metric in ['metrics/mAP50(B)', 'metrics/mAP50-95(B)']:
        if metric in yolo_history[0]:
            axes[2].plot(range(1, 11), [float(r[metric]) for r in yolo_history], marker='.', label=metric)
    axes[2].set_title('YOLO26n: validation'); axes[2].set_xlabel('Epoch'); axes[2].legend(frameon=False, fontsize=8)
    fig.savefig(run / 'training_curves.png', dpi=180, bbox_inches='tight'); plt.close(fig)
    m0, m1, y0, y1 = [metrics[k] for k in ['mtcnn_before', 'mtcnn_after', 'yolo_before', 'yolo_after']]
    report = f'''# Fine-tune MTCNN và YOLO26n 10 epoch trên VNLP — 25/10/2026

MTCNN: đúng toàn biển từ **{m0['exact_count']} lên {m1['exact_count']}/100**, recall box được chọn từ **{100*m0['detector_selected']['recall']:.2f}% lên {100*m1['detector_selected']['recall']:.2f}%**. YOLO26n + EasyOCR: đúng toàn biển từ **{y0['exact_count']} lên {y1['exact_count']}/100**, recall từ **{100*y0['detector_selected']['recall']:.2f}% lên {100*y1['detector_selected']['recall']:.2f}%**. OCR và ngưỡng suy luận được giữ nguyên; không huấn luyện LPRNet/EasyOCR.

## Dữ liệu và chống giao tập

Dùng đúng split train hiện có của dự án: **{audit['train']} ảnh train, {audit['validation']} ảnh validation**. Không mở rộng thành toàn bộ ảnh VNLP còn lại. Hai detector dùng cùng các scene train/validation. 100 ảnh test giữ nguyên byte manifest từ lần đối sánh trước: 34 biển một hàng, 33 biển hai hàng ô tô, 33 biển xe máy. Loại trừ theo đường dẫn, định danh biển và kiểm tra SHA-256 pixel giải mã; train/validation/test không giao định danh hoặc pixel. Test không được đưa vào YAML huấn luyện YOLO hoặc tạo patch/hard negative MTCNN.

Tập test đã được đánh giá trước, nên đây là phép đối sánh trước/sau trên test đã biết, không phải bằng chứng độc lập mới. Nguồn ký tự kế thừa của HOG/SVM và quan hệ với dữ liệu huấn luyện checkpoint bên ngoài chưa được xác minh đầy đủ theo camera/phiên ghi hình.

## Huấn luyện đã khóa trước khi chạy

Seed `{protocol['seed']}`, GPU **{protocol['gpu']}**. Dùng checkpoint pretrained của đúng hai repo trong [đối sánh gốc](comparison-experiment.md). Không chọn checkpoint hoặc ngưỡng theo test. Đánh giá **checkpoint epoch 10 cuối cùng**, không chọn best theo test hoặc best epoch validation. Log và checkpoint được lưu từng epoch.

**MTCNN:** fine-tune PNet đủ 10 epoch, sau đó ONet đủ 10 epoch (mỗi stage 10, không phải 5+5). Khởi tạo bằng PNet/ONet upstream. Adam lr=0,0001, batch 128, loss đúng mask/trọng số upstream: 0,02 CE trên positive/background và 0,6 MSE trên positive/partial. Mỗi scene tạo 12 positive, 8 partial và 32 background; IoU ≥0,65 là positive, 0,4–0,65 là partial, <0,3 là background, khoảng 0,3–0,4 bỏ qua. Jitter quanh box và lấy nền trong ảnh; chuẩn hóa RGB (x−127,5)/128. PNet input 47×12, ONet 94×24. Sau PNet epoch10, khai thác thêm **{mining['extra_patches']} patch** từ proposal trên train, tối đa 16/ảnh; ONet dùng tổng **{mining['onet_train_patches']} patch**. Validation không được cập nhật gradient. STN/LPRNet, bảng ký tự Trung Quốc và greedy decoder giữ nguyên.

**YOLO26n:** bắt đầu từ `models/best.pt` của project Esteebaan23; **10 epoch**, input 640, batch16, AdamW lr0=0,001, lrf=0,01, warm-up1 epoch, patience=0 (tắt early stopping), close_mosaic=0, AMP bật, workers0. Nhãn VNLP xywh chuyển thành YOLO center xywh chuẩn hóa, một lớp `License Plate`. Các augmentation còn lại theo Ultralytics. Dùng `yolo_train/weights/last.pt` sau đủ 10 epoch, không dùng best.pt để chọn kết quả thuận lợi. Vehicle segmentation và English EasyOCR không thay weight. [API train chính thức](https://docs.ultralytics.com/modes/train/).

Thời gian huấn luyện: PNet {training['pnet']['elapsed_seconds']:.1f} giây; ONet {training['onet']['elapsed_seconds']:.1f} giây; YOLO {training['yolo']['elapsed_seconds']:.1f} giây. Thời gian stage MTCNN gồm train/validation/checkpoint, chưa gồm sinh patch/hard mining.

## Kết quả trên đúng 100 ảnh test

| Hệ thống | Đúng toàn biển | CER ↓ | Precision box | Recall box | F1 box | AP50 all | Mean CPU ms/ảnh | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

Precision/recall/F1 dùng **box cuối cùng được chọn**, ghép một-một IoU ≥0,5; AP dùng mọi box sau ngưỡng, nội suy 101 điểm, không phải toàn bộ COCO evaluator. Exact/CER tính trên đủ 100 ảnh; không phát hiện/không đủ confidence OCR là chuỗi rỗng. CER = tổng edit distance / tổng ký tự nhãn, thấp hơn tốt hơn. `summary.json` còn giữ AP50:95, kết quả theo bố cục, p50/p95 và các dự đoán chi tiết.

Đánh giá lại cả baseline lẫn checkpoint mới trong cùng lượt chạy trên CPU Ryzen 7 7700, một luồng OpenCV/PyTorch/BLAS, batch1, 3 lượt; warm-up 3 ảnh validation. Lấy median các lượt cho từng ảnh rồi tính mean trên 100 ảnh. Bỏ thời gian nạp model/đọc ảnh/ghi debug; giữ toàn bộ detector và OCR được gọi. Kiểm tra số luồng sau warm-up và từng inference. Các baseline tái tạo đúng text/box của lần đối sánh trước trên cả 100 ảnh; độ trễ có thể khác do thời điểm đo.

Giữ MTCNN min plate 50×15, PNet threshold0,6, ONet threshold0,8, OCR resize94×24. Nhánh YOLO giữ infer_scale0,5, plate input416, conf0,30, OCR gate0,75, ROI tắt và car-only tắt; vẫn chạy vehicle segmentation. Chọn confidence cao nhất cho hai detector mạng. Giữ bản sửa lỗi đọc y của quadrilateral trong ghép dòng EasyOCR đã dùng ở thí nghiệm trước. Không đổi ký tự O/0, I/1 theo nhãn; EasyOCR chỉ viết hoa/bỏ dấu phân cách. Số ảnh được gọi EasyOCR: **{summary['yolo_ocr_images']['yolo_before']} trước**, **{summary['yolo_ocr_images']['yolo_after']} sau**; thay đổi coverage là hệ quả confidence detector mới, không phải đổi ngưỡng.

## So sánh ghép cặp sau trừ trước

Exact delta và CI dùng điểm phần trăm; bootstrap 10.000 lần theo cặp ảnh. McNemar exact hai phía. Hai cột “chỉ sau/trước đúng” đếm ảnh được sửa và ảnh bị làm sai; không chọn lại mẫu.

| Detector | Exact delta | CI95% | Chỉ sau đúng | Chỉ trước đúng | McNemar p |
|---|---:|---|---:|---:|---:|
{chr(10).join(differences)}

Đây là fine-tune **detector**. MTCNN nhận diện toàn biển còn phụ thuộc STN/LPRNet pretrained Trung Quốc và giả định bố cục; cải thiện phát hiện không bảo đảm cải thiện chuỗi biển Việt Nam. YOLO còn phụ thuộc EasyOCR và gate0,75 của project. Không suy diễn rằng kết quả này là giới hạn của các kiến trúc hoặc của pipeline đã fine-tune cả OCR.

## Tái lập và tài nguyên

Trong thư mục `Detect_plate`, dùng môi trường đã có PyTorch CUDA, Ultralytics, EasyOCR và Registry/data lịch sử:

```powershell
.\\.venv\\Scripts\\python.exe scripts/run_finetune_experiment.py
.\\.venv\\Scripts\\python.exe scripts/build_finetune_report.py --run "<thu_muc_run_vua_in>"
```

Có thể tách `--phase prepare`, `mtcnn`, `yolo`, `evaluate` cùng `--run` đã prepare. Runner kiểm tra hash source ở các pha tiếp tục. Giữ các checkpoint, manifest và source snapshot cùng nhau; không ghi đè Registry triển khai. Thư mục local **`{run.relative_to(ROOT).as_posix()}/`** chứa:

- `protocol_before_training.json`, `evaluation_protocol_before_test.json`, `split_audit.json`: cấu hình và kiểm tra chống giao tập.
- `train_manifest.json`, `validation_manifest.json`, `manifest.json`: danh sách cố định.
- `pnet_history.json`, `onet_history.json`, `yolo_train/results.csv`: đủ 10 epoch và validation.
- `pnet_epoch10.pth`, `onet_epoch10.pth`, `yolo_train/weights/last.pt`: checkpoint được đánh giá.
- `*_predictions.json`, `paired_predictions.csv`, `metrics.csv`, `summary.json`: dự đoán/timing/chênh lệch.
- `source_snapshot.json`, `environment-freeze.txt`: mã và môi trường tái lập.

Hash test manifest `{audit['test_manifest_sha256']}` trùng manifest gốc. Hash evaluation protocol `{summary['evaluation_protocol_sha256']}`. Hash weight pretrained được kiểm tra không đổi trước test; OCR giữ nguyên. Trích dẫn VNLP theo [tài liệu dữ liệu](data.md): MAPR 2022, DOI [10.1109/MAPR56351.2022.9924802](https://doi.org/10.1109/MAPR56351.2022.9924802), và Pham 2023, DOI [10.1007/s00371-021-02375-0](https://doi.org/10.1007/s00371-021-02375-0).

![Trước và sau](before_after.png)

![Đường học](training_curves.png)
'''
    (run / 'report.md').write_text(report.replace('(comparison-experiment.md)', '(../../docs/comparison-experiment.md)').replace(
        '(data.md)', '(../../docs/data.md)'), encoding='utf-8')
    published = report.replace('![Trước và sau](before_after.png)', f'Biểu đồ cục bộ: `{run.relative_to(ROOT).as_posix()}/before_after.png`.').replace(
        '![Đường học](training_curves.png)', f'Đường học cục bộ: `{run.relative_to(ROOT).as_posix()}/training_curves.png`.')
    (ROOT / 'docs/finetune-10-experiment.md').write_text(published, encoding='utf-8')
    (ROOT / 'docs/results/finetune-10-20261025.json').write_text(json.dumps(dict(summary,
        source_summary_sha256=sha256((run / 'summary.json').read_bytes()).hexdigest()), ensure_ascii=False, indent=2), encoding='utf-8')
    print(ROOT / 'docs/finetune-10-experiment.md')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    build(parser.parse_args().run.resolve())
