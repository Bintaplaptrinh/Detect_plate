"""Publish only freshly retested current/fine-tuned models to comparison-experiment.md."""
import argparse
import csv
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = dict(current_hog_svm='HOG/SVM hiện tại', mtcnn_finetuned='MTCNN fine-tune + STN/LPRNet',
             yolo26n_finetuned='YOLO26n fine-tune + EasyOCR')
LAYOUTS = dict(one_row='Một hàng', two_rows='Hai hàng ô tô', two_rows_label_xe_may='Hai hàng xe máy')


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def csv_write(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def percent(value):
    return f'{100*value:.2f}%'


def prf_cells(values):
    return [str(values[k]) for k in ['tp', 'fp', 'fn']]+[percent(values[k]) for k in ['precision', 'recall', 'f1']]


def table(headers, rows):
    alignment = ['---' if h in {'Hệ thống', 'Bố cục', 'OCR của hệ thống'} else '---:' for h in headers]
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(alignment)+' |']+
                     ['| '+' | '.join(map(str, row))+' |' for row in rows])


def flatten(model, layout, result):
    text, det, joint, ocr = result['text'], result['system']['detector_selected'], result['joint'], result['ocr_gt']
    row = dict(model=model, layout=layout, samples=text['samples'], system_exact_count=text['exact_count'],
               system_exact_match=text['exact_match'], system_cer=text['character_error_rate'],
               localized_exact_match=joint['localized_exact_match'], ocr_exact_count=ocr['exact_count'],
               ocr_exact_match=ocr['exact_match'], ocr_cer=ocr['character_error_rate'],
               ocr_character_accuracy=ocr['character_accuracy'])
    for prefix, values in [('detector', det), ('joint_recognition', joint), ('ocr_character', ocr['character_prf'])]:
        row.update({f'{prefix}_{k}': values[k] for k in ['tp', 'fp', 'fn', 'precision', 'recall', 'f1']})
    row.update({f'ocr_{k}': v for k, v in ocr['alignment'].items()})
    row.update(system_mean_ms=result['system']['latency']['mean_ms'], system_p95_ms=result['system']['latency']['p95_ms'],
               ocr_mean_ms=ocr['latency']['mean_ms'], ocr_p95_ms=ocr['latency']['p95_ms'])
    return row


def build(run):
    summary = load(run / 'summary.json')
    if summary.get('scope') != 'post_finetune_only' or set(summary.get('models', {})) != set(NAMES):
        raise ValueError('Use the fresh run_post_finetune_evaluation.py output containing only the three current models')
    protocol, audit = load(run / 'protocol_before_test.json'), load(run / 'split_audit.json')
    models = summary['models']
    assert sha256((run / 'protocol_before_test.json').read_bytes()).hexdigest() == summary['protocol_sha256']
    assert sha256((run / 'manifest.json').read_bytes()).hexdigest() == summary['test_manifest_sha256']
    overall, selected, all_boxes, joint, ocr_table, ocr_prf, timings = [], [], [], [], [], [], []
    layout_system, layout_ocr, layout_det, layout_ocr_prf, layout_joint = [], [], [], [], []
    csv_rows, layout_csv = [], []
    for name, label in NAMES.items():
        result = models[name]
        t, s, o, j = result['text'], result['system'], result['ocr_gt'], result['joint']
        overall.append([label, t['samples'], f"{t['exact_count']}/100", percent(t['exact_match']),
                        percent(t['character_error_rate']), percent(j['localized_exact_match'])])
        selected.append([label]+prf_cells(s['detector_selected']))
        all_boxes.append([label]+prf_cells(s['detector_all'])+[percent(s['ap50']), percent(s['map50_95'])])
        joint.append([label]+prf_cells(j))
        ocr_table.append([label, o['samples'], f"{o['exact_count']}/100", percent(o['exact_match']),
                          percent(o['character_error_rate']), percent(o['character_accuracy'])])
        ocr_prf.append([label]+prf_cells(o['character_prf']))
        timings.append([label]+[f'{v:.2f}' for v in [s['latency']['mean_ms'], s['latency']['median_ms'],
                       s['latency']['p95_ms'], s['latency']['serial_fps'], o['latency']['mean_ms'], o['latency']['p95_ms']]])
        csv_rows.append(flatten(name, 'all', result))
        for layout in LAYOUTS:
            v = result['by_layout'][layout]; a, b = v['text'], v['ocr_gt']
            layout_system.append([LAYOUTS[layout], label, a['samples'], f"{a['exact_count']}/{a['samples']}",
                                  percent(a['exact_match']), percent(a['character_error_rate']), percent(v['joint']['localized_exact_match'])])
            layout_ocr.append([LAYOUTS[layout], label, b['samples'], f"{b['exact_count']}/{b['samples']}",
                               percent(b['exact_match']), percent(b['character_error_rate']), percent(b['character_accuracy'])])
            layout_det.append([LAYOUTS[layout], label]+prf_cells(v['system']['detector_selected']))
            layout_ocr_prf.append([LAYOUTS[layout], label]+prf_cells(b['character_prf']))
            layout_joint.append([LAYOUTS[layout], label]+prf_cells(v['joint']))
            layout_csv.append(flatten(name, layout, v))
    # Group tables by layout for easy comparison of all three systems.
    for rows in [layout_system, layout_ocr, layout_det, layout_ocr_prf, layout_joint]:
        rows.sort(key=lambda row: list(LAYOUTS.values()).index(row[0]))
    csv_write(run / 'metrics.csv', csv_rows)
    csv_write(run / 'metrics_by_layout.csv', layout_csv)
    paired = []
    system_rows = {name: load(run / f'{name}_predictions.json') for name in NAMES}
    ocr_rows = {name: load(run / f'{name}_ocr_gt_predictions.json') for name in NAMES}
    for i, scene in enumerate(load(run / 'manifest.json')):
        row = dict(path=scene['path'], truth=scene['text'], layout=scene['layout'])
        for name in NAMES:
            assert system_rows[name][i]['path'] == ocr_rows[name][i]['path'] == scene['path']
            row[f'{name}_system'] = system_rows[name][i]['text']
            row[f'{name}_ocr_gt'] = ocr_rows[name][i]['text']
        paired.append(row)
    csv_write(run / 'paired_predictions.csv', paired)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), layout='constrained')
    labels = ['HOG/SVM', 'MTCNN +\nSTN/LPRNet', 'YOLO26n +\nEasyOCR']
    x = np.arange(3)
    for offset, task, color in [(-.18, 'system', '#177E89'), (.18, 'ocr_gt', '#4D64AD')]:
        values = [100*(models[k]['text']['exact_match'] if task == 'system' else models[k]['ocr_gt']['exact_match']) for k in NAMES]
        bars = axes[0].bar(x+offset, values, .34, label=task, color=color)
        axes[0].bar_label(bars, fmt='%.0f', padding=3, fontsize=9)
    axes[0].set_xticks(x, labels); axes[0].set_title('Exact match (%)'); axes[0].legend(frameon=False, fontsize=8)
    for ax, values, title in [(axes[1], [100*models[k]['system']['detector_selected']['recall'] for k in NAMES], 'Detection recall, IoU 0.5 (%)'),
                              (axes[2], [models[k]['system']['latency']['mean_ms'] for k in NAMES], 'Full-system CPU mean (ms/image)')]:
        bars = ax.bar(labels, values, color=['#177E89', '#DE7A22', '#4D64AD'], width=.62)
        ax.bar_label(bars, fmt='%.2f', padding=4, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, max(values+[1])*1.18)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False); ax.tick_params(axis='x', labelsize=9)
    axes[0].set_ylim(0, max(max(p.get_height() for p in axes[0].patches), 1)*1.2)
    fig.suptitle('Current models after detector fine-tuning: fresh 100-image retest', fontsize=12)
    fig.savefig(run / 'comparison.png', dpi=180, bbox_inches='tight'); plt.close(fig)
    date = datetime.fromisoformat(summary['completed_at_utc']).astimezone(timezone(timedelta(hours=7))).strftime('%d/%m/%Y %H:%M:%S UTC+7')
    c, m, y = [models[k] for k in NAMES]
    prf_header = ['Hệ thống', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1']
    report = f'''# Đối sánh hệ thống và OCR sau fine-tune trên 100 ảnh VNLP

Lần test hoàn tất: **{date}**. Báo cáo dùng duy nhất HOG/SVM đang triển khai, MTCNN đã fine-tune PNet/ONet 10 epoch và YOLO26n đã fine-tune 10 epoch. Cả nhận diện toàn hệ thống và OCR trên crop nhãn thật đều được **chạy lại {protocol['repeats']} lượt trên đủ 100 ảnh**, không lấy lại số liệu OCR từ báo cáo khác.

Đúng toàn biển khi chạy toàn hệ thống: **HOG/SVM {c['text']['exact_count']}/100**, **MTCNN + STN/LPRNet {m['text']['exact_count']}/100**, **YOLO26n + EasyOCR {y['text']['exact_count']}/100**. Khi cấp crop nhãn thật cho OCR: lần lượt **{c['ocr_gt']['exact_count']}/100**, **{m['ocr_gt']['exact_count']}/100**, **{y['ocr_gt']['exact_count']}/100**.

## Dữ liệu và mô hình được kiểm tra

Giữ đúng 100 ảnh test C: **34 biển một hàng, 33 biển hai hàng ô tô và 33 biển hai hàng xe máy**. Detector được fine-tune trên **{audit['train']} ảnh train**, theo dõi bằng **{audit['validation']} ảnh validation**. Kiểm tra lại đường dẫn, định danh biển và hash pixel giải mã: train/validation/test không giao định danh hoặc pixel. 100 ảnh test không dùng để cập nhật trọng số hay chọn ngưỡng. Đây là tập test đã được sử dụng trong quá trình nghiên cứu, không phải holdout độc lập mới theo camera/phiên ghi hình.

| Nhánh | Trọng số được nạp để test | OCR |
|---|---|---|
| HOG/SVM hiện tại | `research_glyph_filter/v1` | Phân đoạn hybrid, lọc ký tự, HOG/SVM và hậu xử lý hiện tại |
| MTCNN fine-tune | `pnet_epoch10.pth` và `onet_epoch10.pth` | STN/LPRNet của repo xuexingyu24 |
| YOLO26n fine-tune | `yolo_train/weights/last.pt`, sau đủ epoch 10 | Tiền xử lý và English EasyOCR của project Esteebaan23 |

Checkpoint detector lấy tại epoch 10 cuối cùng. PNet và ONet mỗi mạng đã được huấn luyện đủ 10 epoch. **OCR không được fine-tune trong thí nghiệm detector**; bảng OCR bên dưới là phép đo mới của OCR đang gắn với các mô hình hiện tại. Vehicle segmentation của nhánh YOLO giữ nguyên. Hash của mọi trọng số hoạt động được kiểm tra trước và sau test trong protocol.

## Độ chính xác khi chạy toàn hệ thống

{table(['Hệ thống', 'Ảnh', 'Biển đúng', 'Exact match', 'CER ↓', 'Đúng cả vị trí và chuỗi'], overall)}

Exact match yêu cầu chuỗi đầu ra đúng hoàn toàn so với nhãn; CER = tổng khoảng cách Levenshtein / tổng ký tự nhãn. Ảnh không phát hiện hoặc không đủ điều kiện gọi OCR được tính là chuỗi rỗng. Cột cuối còn yêu cầu box biển được chọn đạt **IoU ≥ 0,5**, giúp phân biệt chuỗi đúng với nhận diện đúng cả vùng biển. Nhãn test không được truyền vào predictor để chọn box hoặc sửa chuỗi.

## Phát hiện vùng biển: TP, FP, FN, Precision, Recall, F1

Mỗi ảnh có một box nhãn. Bảng chính dùng **box cuối cùng được chọn**: HOG/SVM theo reranking, MTCNN/YOLO theo confidence detector. Box đúng ở IoU ≥ 0,5 là TP; box sai là FP và để lại một FN; không có box là FN. Precision = TP/(TP+FP), Recall = TP/(TP+FN), F1 = 2PR/(P+R). Không có ảnh âm nên không báo TN, specificity hoặc accuracy nhị phân.

{table(prf_header, selected)}

Đối với **toàn bộ ứng viên sau ngưỡng**, ghép box một-một với nhãn. HOG/SVM giữ nhiều ứng viên để OCR rerank nên bảng này có thể có precision thấp hơn box cuối cùng. AP nội suy 101 điểm; AP50:95 trung bình 10 ngưỡng từ 0,50 đến 0,95, không phải toàn bộ COCO evaluator theo area/crowd/maxDets.

{table(prf_header+['AP50', 'AP50:95'], all_boxes)}

## Nhận diện đúng cả vị trí và chuỗi: TP, FP, FN

TP yêu cầu **box IoU ≥ 0,5 và chuỗi đúng hoàn toàn**. Một dự đoán có text nhưng sai vị trí hoặc sai chữ tạo FP và FN; không có text tạo FN. Chỉ tính kết quả biển cuối cùng được phát ra, không gộp mọi ứng viên OCR.

{table(prf_header, joint)}

## Đối sánh OCR trên crop nhãn thật, chạy lại độc lập

Cắt cùng box nhãn thật từ từng ảnh và đưa pixel crop vào OCR của cả ba nhánh. Không chạy detector, không dùng gate confidence 0,75 của YOLO trong phép đo này. OCR gồm các bước tiền xử lý/phân đoạn/nhận dạng/hậu xử lý của nhánh tương ứng. Bbox nhãn chỉ được dùng để tạo crop; chuỗi nhãn chỉ dùng sau suy luận để chấm điểm. Đây là chẩn đoán OCR, không được cộng vào kết quả toàn hệ thống.

{table(['OCR của hệ thống', 'Crop', 'Biển đúng', 'Exact match', 'CER ↓', 'Ký tự đúng / ký tự nhãn'], ocr_table)}

Ký tự đúng được đếm sau căn chỉnh Levenshtein trên chuỗi đầu ra; tỷ lệ ở cột cuối là **C/N**, không phải 1−CER khi có ký tự chèn. Với căn chỉnh này: TP = ký tự khớp; FP = thay thế + chèn; FN = thay thế + xóa. Nếu có nhiều căn chỉnh tối ưu, ưu tiên đường chéo, rồi xóa, rồi chèn; quy tắc được giữ cố định cho mọi nhánh. Đây là micro P/R/F1 của ký tự theo căn chỉnh, không phải macro-F1 của bộ phân loại 35 lớp.

{table(prf_header, ocr_prf)}

Chuỗi EasyOCR được viết hoa và bỏ khoảng trắng/dấu phân cách để khớp định dạng nhãn VNLP. Không thay O/0, I/1 hay loại ký tự Trung Quốc để làm đẹp điểm. Raw OCR cũng được lưu riêng. Với EasyOCR, adapter giữ bản sửa lỗi tọa độ ghép dòng: lấy y trung bình của bốn góc thay cho phép trừ tuple trong mã nguồn project. STN/LPRNet giữ decoder upstream, bao gồm cách suppression timestep đầu.

## Toàn hệ thống theo từng bố cục test

{table(['Bố cục', 'Hệ thống', 'Ảnh', 'Biển đúng', 'Exact match', 'CER ↓', 'Đúng vị trí và chuỗi'], layout_system)}

## Phát hiện vùng biển theo từng bố cục

{table(['Bố cục']+prf_header, layout_det)}

## Nhận diện đúng vị trí và chuỗi theo từng bố cục

{table(['Bố cục']+prf_header, layout_joint)}

## OCR trên crop nhãn thật theo từng bố cục

{table(['Bố cục', 'OCR của hệ thống', 'Crop', 'Biển đúng', 'Exact match', 'CER ↓', 'Ký tự đúng / nhãn'], layout_ocr)}

TP/FP/FN và Precision/Recall/F1 **ký tự OCR** theo từng bố cục:

{table(['Bố cục']+prf_header, layout_ocr_prf)}

Các chỉ số trên được lưu đầy đủ trong `metrics_by_layout.csv` và JSON tổng hợp; mỗi dòng tương ứng một nhánh và một bố cục. Không loại ảnh lỗi khỏi mẫu số.

## Độ trễ đo mới trên CPU

CPU **{protocol['cpu']}**, Windows, batch1, một luồng OpenCV/PyTorch/BLAS. Mỗi tác vụ warm-up bằng ba ảnh validation, chạy {protocol['repeats']} lượt đủ 100 ảnh, đổi thứ tự nhánh giữa các lượt. Lấy median các lượt cho từng ảnh, sau đó tính mean/median/p95 trên 100 ảnh; FPS = 1000/mean_ms. Đọc ảnh, cắt crop GT và nạp checkpoint nằm ngoài timer. System timer gồm detector và các bước OCR thực sự được gọi; OCR timer chỉ gồm pipeline OCR trên crop GT. Kiểm tra số luồng sau warm-up và sau từng suy luận.

{table(['Hệ thống', 'System mean ms', 'System median ms', 'System P95 ms', 'System FPS', 'OCR mean ms/crop', 'OCR P95 ms/crop'], timings)}

Giữ cấu hình MTCNN min plate 50×15, ngưỡng PNet 0,6 / ONet 0,8, OCR resize 94×24. YOLO giữ infer_scale 0,5, plate input 416, conf 0,30, OCR gate 0,75 và crop ít nhất 12×30; ROI và car-only tắt; vẫn chạy vehicle segmentation. Trong toàn hệ thống, **{summary['yolo_ocr_images']}/100 ảnh** được gọi EasyOCR. Các lần không gọi OCR vẫn nằm trong trung bình system latency. YOLO/LPRNet dùng FP32, EasyOCR giữ quantization CPU mặc định. GPU không tham gia suy luận trong bảng này.

## Diễn giải và giới hạn

Kết quả vùng biển và chuỗi biển đo hai mức khác nhau. MTCNN đã thích ứng detector nhưng STN/LPRNet vẫn là OCR pretrained cho biển Trung Quốc; YOLO26n còn phụ thuộc English EasyOCR và các điều kiện gọi OCR. Cấp box đúng không bảo đảm OCR crop GT là cận trên toán học, vì pipeline toàn ảnh có thể hiệu chỉnh/rerank ra crop khác. Tập có một biển được gán nhãn mỗi ảnh và không có ảnh âm; chưa suy diễn khả năng tổng quát hóa camera độc lập hoặc tần suất báo giả ở cảnh không có biển.

## Tái lập và nguồn kết quả hiện tại

```powershell
.\\.venv\\Scripts\\python.exe scripts/run_post_finetune_evaluation.py --training-run reports/finetune-10-20261025
.\\.venv\\Scripts\\python.exe scripts/build_comparison_report.py --run "<thu_muc_retest_vua_in>"
```

Runner chỉ nạp ba nhánh hiện tại và không huấn luyện thêm. MTCNN, YOLO26n và OCR khởi tạo từ [xuexingyu24/License_Plate_Detection_Pytorch](https://github.com/xuexingyu24/License_Plate_Detection_Pytorch/tree/aa5778d6e6da0f382894048b4199cca7a1151e95) và [Esteebaan23/Vehicle-License-Plate-Detection-with-YOLO26](https://github.com/Esteebaan23/Vehicle-License-Plate-Detection-with-YOLO26/tree/25f9099a8b1af72c8d481e8bf1ec62e586b98887); checkpoint detector epoch 10 là kết quả fine-tune VNLP cục bộ. Không chọn lại checkpoint/ngưỡng theo test. PyTorch `{protocol['packages']['torch']}`, OpenCV `{protocol['packages']['opencv']}`, NumPy `{protocol['packages']['numpy']}`; môi trường đầy đủ ở `environment-freeze.txt`.

Đầu ra lần test này: `{run.relative_to(ROOT).as_posix()}/`.

- [JSON tổng hợp hiện tại](results/comparison-100-20261025.json): chỉ chứa ba nhánh đang được đối sánh, gồm OCR/system và mọi bố cục.
- [CSV toàn bộ chỉ số](../{run.relative_to(ROOT).as_posix()}/metrics.csv), [CSV theo bố cục](../{run.relative_to(ROOT).as_posix()}/metrics_by_layout.csv), [chuỗi dự đoán từng ảnh](../{run.relative_to(ROOT).as_posix()}/paired_predictions.csv).
- `*_predictions.json`: box/confidence, text và thời gian từng lượt của toàn hệ thống; `*_ocr_gt_predictions.json`: OCR crop GT mới.
- `protocol_before_test.json`, `split_audit.json`, `verification.json`: hash checkpoint/manifest/source, chống giao tập và kiểm tra suy luận; `source_snapshot.json` giữ mã nguồn của lần chạy.
- [Biểu đồ kết quả hiện tại](../{run.relative_to(ROOT).as_posix()}/comparison.png). Ảnh, trọng số và chi tiết trong `reports/` lưu cục bộ.

Hash protocol: `{summary['protocol_sha256']}`. Hash manifest test: `{summary['test_manifest_sha256']}`. Không dùng kết quả test đã lưu để điền các bảng mới; tất cả số liệu system và OCR đều tính từ lần chạy này.

VNLP: Trinh và cộng sự, MAPR2022, DOI [10.1109/MAPR56351.2022.9924802](https://doi.org/10.1109/MAPR56351.2022.9924802); Pham, The Visual Computer2023, DOI [10.1007/s00371-021-02375-0](https://doi.org/10.1007/s00371-021-02375-0). Xem [dữ liệu](data.md) và [BibTeX](references.bib).
'''
    (ROOT / 'docs/comparison-experiment.md').write_text(report, encoding='utf-8')
    # Run-local copy uses paths relative to reports/<run>/ instead of docs/.
    local_report = report.replace('(results/', '(../../docs/results/').replace('(../reports/', '(../../reports/').replace(
        '(data.md)', '(../../docs/data.md)').replace('(references.bib)', '(../../docs/references.bib)')
    (run / 'report.md').write_text(local_report, encoding='utf-8')
    exported = dict(summary, source_summary_sha256=sha256((run / 'summary.json').read_bytes()).hexdigest())
    (ROOT / 'docs/results/comparison-100-20261025.json').write_text(json.dumps(exported, ensure_ascii=False, indent=2), encoding='utf-8')
    print(ROOT / 'docs/comparison-experiment.md')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    build(parser.parse_args().run.resolve())
