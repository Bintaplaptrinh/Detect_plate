"""Generate a separate report/notebook for the PCA + FLD + Gaussian HMM arm."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
LABELS = dict(current_svm='SVM hiện tại', pca_fld_svm_hmm='PCA + FLD + SVM + HMM',
              pca_fld_hmm='PCA + FLD + HMM (Gaussian, không SVM OCR)')
REPORT = ROOT / 'docs/pca-fld-hmm-no-svm-comparison.md'
RESULT = ROOT / 'docs/results/pca-fld-hmm-no-svm-180-20261026.json'
NOTEBOOK = '06_pca_fld_hmm_no_svm_comparison'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def pct(value):
    return f'{100*value:.2f}%'


def build(run):
    run = run.resolve()
    summary, protocol = read(run / 'summary.json'), read(run / 'protocol_before_test.json')
    training, split = read(run / 'training_audit.json'), read(run / 'split_audit.json')
    if sha256((run / 'protocol_before_test.json').read_bytes()).hexdigest() != summary['protocol_sha256']:
        raise ValueError('Protocol checksum mismatch')
    if sha256((run / 'no_svm_candidate.joblib').read_bytes()).hexdigest() != protocol['candidate_sha256']:
        raise ValueError('Candidate checksum mismatch')
    predictions = {name: read(run / f'{name}_predictions.json') for name in LABELS}
    reference = Path(protocol['reference_run'])
    for name in LABELS:
        if len(predictions[name]) != protocol['samples'] or any(len(r['latency_ms']) != protocol['repeats'] for r in predictions[name]):
            raise ValueError(f'Incomplete predictions: {name}')
        for a, b in zip(predictions['current_svm'], predictions[name]):
            if a['path'] != b['path'] or a['truth'] != b['truth'] or a['detections'] != b['detections']:
                raise ValueError(f'Unpaired test data/detector outputs: {name}')
    for name in ('current_svm', 'pca_fld_svm_hmm'):
        previous = read(reference / f'{name}_predictions.json')
        if len(previous) != protocol['samples'] or any(any(a[k] != b[k] for k in ('path', 'raw', 'text', 'box'))
                for a, b in zip(previous, predictions[name])):
            raise ValueError(f'Reference predictions changed: {name}')
    summary['reference_prediction_check'] = dict(samples=protocol['samples'], arms=['current_svm', 'pca_fld_svm_hmm'],
        mismatches=0, fields=['path', 'raw', 'text', 'box'], detector_equal_across_arms=True)
    lines = ['# Đối sánh riêng PCA + FLD + HMM không dùng SVM trong OCR', '',
        '## 1. Mô hình và phạm vi', '',
        'Nhánh mới thay bộ phân loại SVM bằng phát xạ Gaussian của HMM trong không gian Fisher: '
        '**HOG 540 → StandardScaler → PCA 48 → FLD 34 → Gaussian emissions → HMM/Viterbi**. '
        'PCA và FLD là phép chiếu; cần mô hình phát xạ để tính độ phù hợp của quan sát với từng ký tự. '
        'Ở đây mỗi ký tự có một vector trung bình riêng, dùng chung ma trận hiệp phương sai nội lớp đầy đủ. '
        'Likelihood Gaussian được chuẩn hóa theo hàng với prior ký tự đồng đều trước khi đưa vào decoder.', '',
        'Nhánh OCR mới không tạo, huấn luyện hoặc gọi SVC. Kiểm thử chặn SVC.fit vẫn huấn luyện được nhánh Gaussian. '
        '**Detector của toàn hệ thống vẫn giữ SVM kiểm chứng vùng hiện tại** để phép đối sánh chỉ thay khâu OCR. '
        'Vì vậy tên “không SVM” trong báo cáo này chỉ áp dụng cho OCR, không áp dụng cho detector.', '',
        f'Phát xạ Gaussian được fit trên đúng {training["total"]} glyph train của bản phát hành, '
        f'{len(training["classes"])} lớp: {training["legacy_glyphs"]} legacy + {training["vnlp_glyphs"]} VNLP. '
        'Scaler, PCA, FLD, trung bình và covariance chỉ học train. '
        'Covariance = tổng scatter nội lớp/(N−C), cộng 1e−6 × trace(covariance)/d vào đường chéo. '
        'Tham số này được khóa trước test, không chọn theo kết quả test.', '',
        f'Phép chiếu PCA/FLD được kiểm tra bằng số với nhánh có SVM của [thí nghiệm trước](pca-fld-hmm-comparison.md). '
        f'HMM dùng cùng bigram từ {protocol["hmm"]["training_sequences"]} chuỗi train, smoothing=1, '
        'template và ngưỡng mean log penalty=0.35. Nhánh Gaussian dùng likelihood thật của phân bố quan sát; '
        'nhánh SVM đối chiếu dùng posterior SVC như pseudo-emission. '
        'Giữ HOG, detector, glyph filter, giả thuyết phân đoạn, top-k và trọng số rerank. '
        'Chọn crop và giả thuyết vẫn dựa trên chất lượng OCR của từng nhánh, không dùng nhãn chuẩn.', '',
        f'Chạy mới cả ba nhánh trên {protocol["samples"]} ảnh test C lịch sử '
        f'(60 ảnh/bố cục), {protocol["repeats"]} lượt, CPU một luồng, warm-up ba ảnh validation. '
        'Không dùng lại thời gian của lần chạy trước. Độ trễ là trung bình median theo ảnh, '
        'bao gồm toàn bộ inference từ detector đến xếp hạng; không gồm nạp model, đọc ảnh và mạng. '
        'Không có giao danh tính/pixel scene giữa train, validation và test; provenance legacy chưa đầy đủ. '
        'Test C đã từng được quan sát, chưa phải holdout độc lập mới.', '',
        '## 2. Kết quả toàn pipeline', '',
        '| Nhánh | Đúng toàn biển | Exact match | CER ↓ | Mean ms/ảnh ↓ | P95 ms ↓ |',
        '|---|---:|---:|---:|---:|---:|']
    for name, label in LABELS.items():
        m = summary['metrics'][name]
        lines.append(f'| {label} | {m["exact_count"]}/{protocol["samples"]} | {pct(m["decoded"]["exact_match"])} | '
                     f'{pct(m["decoded"]["character_error_rate"])} | {m["latency"]["mean_ms"]:.2f} | {m["latency"]["p95_ms"]:.2f} |')
    lines += ['', '### Chênh lệch ghép cặp của nhánh không SVM OCR', '',
              '| So với | Chênh exact (điểm %) | CI95% exact | Chênh CER (điểm %) | CI95% CER | Tốc độ × | CI95% tốc độ |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for name in ('current_svm', 'pca_fld_svm_hmm'):
        c = summary['paired_comparisons'][f'pca_fld_hmm_minus_{name}']
        m, t = c['accuracy'], c['latency']
        elo, ehi = m['exact_match_delta_ci95']
        clo, chi = m['cer_delta_ci95']
        tlo, thi = t['speedup_ci95']
        lines.append(f'| {LABELS[name]} | {m["exact_match_delta"]*100:+.2f} | [{elo*100:+.2f}; {ehi*100:+.2f}] | '
                     f'{m["cer_delta"]*100:+.2f} | [{clo*100:+.2f}; {chi*100:+.2f}] | {t["speedup"]:.3f} | [{tlo:.3f}; {thi:.3f}] |')
    lines += ['', 'CI95% dùng bootstrap ghép cặp 10.000 lần theo ảnh; các lần đo được xoay thứ tự nhánh. '
              'Không có kiểm định không thua kém với biên sai số định trước. '
              'Ảnh không phát hiện được biển, tách sai hoặc xuất chuỗi rỗng đều được giữ khi chấm điểm.', '',
              '### Thời gian từng công đoạn (ms/ảnh)', '',
              '| Nhánh | Detector | Phân đoạn | HOG | Phát xạ/bộ phân loại, gồm PCA/FLD | Decoder |',
              '|---|---:|---:|---:|---:|---:|']
    for name, label in LABELS.items():
        stages = summary['metrics'][name]['stages_mean_ms']
        lines.append('| ' + label + ' | ' + ' | '.join(f'{stages[k]:.2f}' for k in ('detector', 'segmentation', 'hog', 'classifier', 'decoder')) + ' |')
    lines += ['', 'Cộng công đoạn trên mọi ứng viên và giả thuyết. Median từng công đoạn được lấy riêng; '
              'tổng các cột có thể khác độ trễ toàn pipeline, vốn gồm cả overhead và rerank.', '',
              '## 3. Chẩn đoán crop GT và glyph', '',
              '| Nhánh | Crop GT exact | Crop GT CER | Mean ms/crop |', '|---|---:|---:|---:|']
    for name, label in LABELS.items():
        m = summary['oracle_crop'][name]
        lines.append(f'| {label} | {pct(m["decoded"]["exact_match"])} | {pct(m["decoded"]["character_error_rate"])} | {m["latency"]["mean_ms"]:.2f} |')
    g = summary['glyph_diagnostic']
    lines += ['', f'Chẩn đoán ký tự dùng {g["acceptance"]["glyphs"]} glyph có nhãn yếu từ '
              f'{g["acceptance"]["accepted_plates"]}/{g["acceptance"]["plates"]} biển được segmentation gốc nhận. '
              'Đây là tập đã lọc theo khớp count/rows, không thay thế chấm chuỗi trên mọi ảnh. '
              f'Có {g["exact_train_glyph_duplicates"]} glyph trùng hash chuẩn hóa với train.', '',
              '| Mô hình quan sát (chưa HMM) | Accuracy | Macro F1 | Median ms/batch từ HOG |', '|---|---:|---:|---:|']
    for name, label in LABELS.items():
        m = g['models'][name]
        lines.append(f'| {label} | {pct(m["accuracy"])} | {pct(m["macro_f1"])} | {m["batch_median_ms"]:.2f} |')
    a, b = summary['metrics']['current_svm'], summary['metrics']['pca_fld_hmm']
    svm_hmm = summary['metrics']['pca_fld_svm_hmm']
    c = summary['paired_comparisons']['pca_fld_hmm_minus_current_svm']
    lines += ['', '## 4. Diễn giải', '',
              f'Nhánh không SVM OCR đúng {b["exact_count"]}/{protocol["samples"]} biển, '
              f'so với {a["exact_count"]} của SVM hiện tại; '
              f'tốc độ toàn pipeline đạt {c["latency"]["speedup"]:.3f}×. '
              'Việc bỏ SVM giảm chi phí tính điểm quan sát nhưng cũng thay đổi phân bố xác suất, '
              'vì vậy có thể ảnh hưởng lựa chọn phân đoạn và ứng viên ngoài thay đổi ký tự.', '',
              f'So với PCA + FLD + SVM + HMM, nhánh không SVM giảm độ trễ từ '
              f'{svm_hmm["latency"]["mean_ms"]:.2f} xuống {b["latency"]["mean_ms"]:.2f} ms/ảnh, '
              f'nhưng CER tăng từ {pct(svm_hmm["decoded"]["character_error_rate"])} lên '
              f'{pct(b["decoded"]["character_error_rate"])}. '
              'Số biển đúng bằng nhau không có nghĩa sai số chuỗi tương đương. '
              f'Accuracy glyph riêng giảm từ {pct(g["models"]["pca_fld_svm_hmm"]["accuracy"])} '
              f'xuống {pct(g["models"]["pca_fld_hmm"]["accuracy"])}; '
              'ảnh hưởng đến toàn chuỗi còn phụ thuộc phân đoạn và xếp hạng ứng viên.', '',
              'Kết quả chỉ mô tả Gaussian một thành phần/lớp với covariance dùng chung trong cấu hình này; '
              'không đại diện mọi mô hình phát xạ HMM. '
              'Không tự động thay bản phát hành đang dùng. Mọi tối ưu tiếp theo cần chọn trên validation và kiểm tra trên holdout mới.', '',
              'Dự đoán raw, decoded và bbox của cả hai nhánh đối chiếu trùng hoàn toàn với thí nghiệm trước; '
              'bbox/score detector trùng trên mọi ảnh giữa ba nhánh. '
              'Nhãn glyph còn là nhãn yếu và chưa có holdout theo camera.', '',
              '## 5. Tái lập', '', '```powershell',
              '.\\.venv\\Scripts\\python.exe scripts/run_no_svm_comparison.py --run reports/my-no-svm-run --reference-run reports/pca-fld-hmm-180-20261026-v2 --repeats 3',
              '.\\.venv\\Scripts\\python.exe scripts/build_no_svm_report.py --run reports/my-no-svm-run',
              f'.\\.venv\\Scripts\\python.exe scripts/run_notebook.py {NOTEBOOK}', '```', '',
              'Cần Registry/dữ liệu lịch sử và candidate.joblib của lần đối sánh trước. '
              'Dùng `--vnlp-root` và `--ocr-root` nếu di chuyển dữ liệu. '
              f'Tài nguyên cục bộ nằm tại `reports/{run.name}`: protocol khóa trước test, audit, '
              'manifest có hash, source snapshot, no_svm_candidate.joblib, dự đoán và thời gian từng lượt, summary.json. '
              f'Notebook phân tích: `notebooks/{NOTEBOOK}.executed.ipynb`.', '',
              f'Protocol SHA-256: `{summary["protocol_sha256"]}`.', '']
    report = '\n'.join(lines)
    REPORT.write_text(report, encoding='utf-8')
    RESULT.write_text(json.dumps(dict(protocol=protocol, training=training, split=split, summary=summary,
        summary_sha256=sha256((run / 'summary.json').read_bytes()).hexdigest()), ensure_ascii=False, indent=2), encoding='utf-8')
    notebook = nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(report), nbf.v4.new_code_cell(
        "from pathlib import Path\nimport json\nimport pandas as pd\nimport matplotlib.pyplot as plt\n"
        "ROOT = Path.cwd().resolve()\nif not (ROOT / 'pyproject.toml').exists(): ROOT = ROOT.parent\n"
        f"data = json.loads((ROOT / 'docs/results/{RESULT.name}').read_text(encoding='utf-8'))\n"
        "summary = data['summary']\n"
        "table = pd.DataFrame({k: {'exact_match': v['decoded']['exact_match'], 'CER': v['decoded']['character_error_rate'], 'mean_ms': v['latency']['mean_ms']} for k,v in summary['metrics'].items()}).T\n"
        "display(table)\nfig, axes = plt.subplots(1,2, figsize=(13,5))\n"
        "table[['exact_match','CER']].plot.bar(ax=axes[0], rot=15, title='Full pipeline: 180 scenes')\n"
        "axes[0].set_ylim(0, max(.6, table[['exact_match','CER']].max().max()*1.3))\n"
        "table[['mean_ms']].plot.bar(ax=axes[1], rot=15, title='Full pipeline latency (ms)')\n"
        "plt.tight_layout()\n"
        f"fig.savefig(ROOT / 'reports/{run.name}/no_svm_comparison.png', dpi=150, bbox_inches='tight')\nplt.show()\n")])
    notebook.metadata.kernelspec = dict(display_name='Python 3 (PlateDetect)', language='python', name='python3')
    path = ROOT / 'notebooks' / f'{NOTEBOOK}.ipynb'
    nbf.write(notebook, path)
    print(REPORT)
    print(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    build(parser.parse_args().run)
