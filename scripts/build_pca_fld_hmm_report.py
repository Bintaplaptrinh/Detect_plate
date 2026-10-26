"""Build the Vietnamese report and notebook from a completed paired experiment."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
LABELS = dict(current_svm='SVM hiện tại + decoder hiện tại', svm_hmm='SVM hiện tại + HMM',
              pca_fld_svm='PCA + FLD + SVM + decoder hiện tại', pca_fld_svm_hmm='PCA + FLD + SVM + HMM')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def pct(x):
    return f'{100*x:.2f}%'


def build(run):
    run = run.resolve()
    summary, protocol = read(run / 'summary.json'), read(run / 'protocol_before_test.json')
    training, split = read(run / 'training_audit.json'), read(run / 'split_audit.json')
    if sha256((run / 'protocol_before_test.json').read_bytes()).hexdigest() != summary['protocol_sha256']:
        raise ValueError('Protocol checksum mismatch')
    rows = {name: read(run / f'{name}_predictions.json') for name in LABELS}
    for name in LABELS:
        if len(rows[name]) != protocol['samples'] or any(len(r['latency_ms']) != protocol['repeats'] for r in rows[name]):
            raise ValueError(f'Incomplete measurement records: {name}')
        for current, other in zip(rows['current_svm'], rows[name]):
            if current['path'] != other['path'] or current['detections'] != other['detections']:
                raise ValueError(f'Detector or scene pairing changed: {name}')
    # Independently check that profiling/decoder injection did not alter release predictions.
    historical = read(ROOT / 'Registration/Registry/research_glyph_filter/v1/predictions.json')
    candidates = [value for key, value in historical.items() if key != 'baseline_v3']
    history = next((value for value in candidates if len(value) == len(rows['current_svm'])), None)
    if history is None:
        raise ValueError('Cannot find historical release predictions')
    old = {r['path']: r for r in history}
    mismatches = [r['path'] for r in rows['current_svm'] if r['path'] not in old or
                  any(r[k] != old[r['path']][k] for k in ('raw', 'text', 'box'))]
    if mismatches:
        raise ValueError(f'Current release differs from historical predictions: {mismatches[:3]}')
    summary['historical_baseline_check'] = dict(samples=len(history), mismatches=0, fields=['raw', 'text', 'box'])
    paired = summary['paired_comparisons']['pca_fld_svm_hmm_minus_current_svm']
    a, b = summary['metrics']['current_svm'], summary['metrics']['pca_fld_svm_hmm']
    lines = [
        '# Đối sánh SVM và PCA + FLD + SVM + HMM trên toàn pipeline', '',
        '## 1. Phạm vi và thiết kế', '',
        f'Thí nghiệm ngày 26/10/2026 đánh giá đủ {protocol["samples"]} ảnh test C lịch sử, '
        '60 ảnh cho mỗi bố cục: một hàng, hai hàng ô tô và hai hàng xe máy. '
        'Đây là đối sánh ghép cặp trên tập đã từng được đánh giá, không phải tập giữ lại độc lập mới.', '',
        'Notebook tham khảo `04_pca_fld_hmm_recognition.ipynb` huấn luyện PCA + FLD + **SVM** '
        'và minh họa HMM bằng xác suất mô phỏng. Bảng được gọi là end-to-end trong notebook '
        'chỉ tổng hợp phân loại ký tự; chưa chạy detector, phân đoạn và OCR toàn ảnh xe. '
        'Chia ngẫu nhiên từng ký tự có thể đưa ký tự cùng biển vào cả train và test. '
        'Thí nghiệm này giữ nguyên chia tập theo danh tính biển số và tái dựng dữ liệu train của bản phát hành hiện tại.', '',
        f'Bốn nhánh dùng cùng {training["total"]:,} ký tự train '
        f'({training["legacy_glyphs"]:,} legacy + {training["vnlp_glyphs"]:,} VNLP), '
        f'{len(training["classes"])} lớp. Kiểm tra lại hash pixel legacy, hash từng glyph VNLP, '
        'thống kê scaler và toàn bộ support vectors của SVM hiện tại đều đạt. '
        f'Train có {split["counts"]["train"]} scene, validation có {split["counts"]["validation"]} scene; '
        'không có giao danh tính hay pixel scene giữa ba tập.', '',
        f'Giảm chiều: StandardScaler → PCA {protocol["diagnostics"]["pca_features"]} → '
        f'FLD {protocol["diagnostics"]["fld_features"]} → SVC RBF C=3, class_weight=balanced, probability=True. '
        'Scaler, PCA, FLD và SVC chỉ fit train; không tìm siêu tham số trên test. '
        f'PCA giữ {pct(protocol["diagnostics"]["pca_explained_variance_ratio_sum"])} phương sai train. '
        'FLD dùng solver SVD; PCA không tự bảo đảm ma trận scatter nội lớp khả nghịch.', '',
        f'HMM học bigram ký tự trên {protocol["hmm"]["training_sequences"]} chuỗi biển train duy nhất, '
        'Laplace smoothing=1, ràng buộc theo các template nghiên cứu sẵn có, giải mã Viterbi. '
        'Đầu vào là posterior SVC dùng như pseudo-emission, không phải likelihood ảnh của một HMM sinh. '
        'Khi không khớp bố cục hoặc mức giảm log xác suất thị giác trung bình vượt 0.35, giữ chuỗi raw. '
        'Các template không đại diện đầy đủ quy định đăng ký biển số.', '',
        'Detector, bộ lọc glyph, các giả thuyết phân đoạn, HOG 540 chiều, top-k và trọng số xếp hạng giữ nguyên. '
        'OCR và decoder vẫn có thể thay đổi thứ hạng giả thuyết và ứng viên theo cơ chế hiện tại. '
        'Nhãn chuẩn không được đưa vào inference hoặc dùng để chọn ứng viên.', '',
        f'Mỗi ảnh chạy {protocol["repeats"]} lần cho mỗi nhánh trên CPU, giới hạn một luồng, '
        'warm-up ba ảnh validation. Thứ tự nhánh xoay theo ảnh và lượt. '
        'Độ trễ chính là trung bình median mỗi ảnh; tính toàn bộ detector → phân đoạn → HOG → '
        'PCA/FLD nếu có → predict_proba → decoder → xếp hạng; không gồm nạp model, đọc/giải mã ảnh và mạng. '
        'Thời gian từng công đoạn đo với cùng instrumentation cho cả bốn nhánh.', '',
        '## 2. Kết quả toàn pipeline', '',
        '| Nhánh | Đúng toàn biển | Exact match | CER ↓ | Chính xác ký tự theo alignment | Localized exact IoU≥0.5 | Mean ms ↓ | P95 ms ↓ |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, label in LABELS.items():
        m = summary['metrics'][name]
        lines.append(f'| {label} | {m["exact_count"]}/{protocol["samples"]} | {pct(m["decoded"]["exact_match"])} | '
                     f'{pct(m["decoded"]["character_error_rate"])} | {pct(m["aligned_text"]["character_accuracy"])} | '
                     f'{pct(m["localized_plate"]["localized_exact_match"])} | {m["latency"]["mean_ms"]:.2f} | {m["latency"]["p95_ms"]:.2f} |')
    accuracy, latency = paired['accuracy'], paired['latency']
    lo, hi = accuracy['exact_match_delta_ci95']
    llo, lhi = latency['mean_delta_ms_ci95']
    clo, chi = accuracy['cer_delta_ci95']
    lines += ['', f'PCA + FLD + SVM + HMM so với hệ hiện tại: '
              f'exact-match chênh {accuracy["exact_match_delta"]*100:+.2f} điểm phần trăm '
              f'(CI95% [{lo*100:+.2f}; {hi*100:+.2f}]); '
              f'CER chênh {accuracy["cer_delta"]*100:+.2f} điểm phần trăm '
              f'(CI95% [{clo*100:+.2f}; {chi*100:+.2f}]). '
              f'Có {accuracy["candidate_only_correct"]} ảnh chỉ nhánh mới đúng, '
              f'{accuracy["current_only_correct"]} ảnh chỉ hệ hiện tại đúng; '
              f'McNemar exact p={accuracy["mcnemar_exact_p"]:.4g}.', '',
              f'Tốc độ toàn pipeline đạt {latency["speedup"]:.3f}× '
              f'(CI95% [{latency["speedup_ci95"][0]:.3f}; {latency["speedup_ci95"][1]:.3f}]); '
              f'chênh độ trễ {latency["mean_delta_ms"]:+.2f} ms/ảnh '
              f'(CI95% [{llo:+.2f}; {lhi:+.2f}]). '
              'Khoảng tin cậy dùng bootstrap ghép cặp 10.000 lần theo ảnh, '
              'không biểu diễn biến thiên giữa nhiều máy hoặc nhiều lần khởi động hệ thống.', '',
              '### Tách tác động giảm chiều và HMM', '',
              '| Đối sánh (nhánh sau trừ nhánh trước) | Chênh exact (điểm %) | CI95% chênh exact | Chênh CER (điểm %) | Tốc độ × |',
              '|---|---:|---:|---:|---:|']
    for key, comparison in summary['paired_comparisons'].items():
        candidate, reference = key.split('_minus_')
        m = comparison['accuracy']
        low, high = m['exact_match_delta_ci95']
        lines.append(f'| {LABELS[candidate]} − {LABELS[reference]} | {m["exact_match_delta"]*100:+.2f} | '
                     f'[{low*100:+.2f}; {high*100:+.2f}] | {m["cer_delta"]*100:+.2f} | {comparison["latency"]["speedup"]:.3f} |')
    lines += ['',
              '### Thời gian từng công đoạn', '',
              '| Nhánh | Detector | Phân đoạn | HOG | Bộ phân loại gồm PCA/FLD | Decoder |',
              '|---|---:|---:|---:|---:|---:|']
    for name, label in LABELS.items():
        stages = summary['metrics'][name]['stages_mean_ms']
        lines.append('| ' + label + ' | ' + ' | '.join(f'{stages[k]:.2f}' for k in ('detector', 'segmentation', 'hog', 'classifier', 'decoder')) + ' |')
    lines += ['', 'Đơn vị ms/ảnh; cộng thời gian trên mọi ứng viên và giả thuyết được xử lý. '
              'Mỗi công đoạn lấy median riêng nên tổng các cột không nhất thiết bằng mean độ trễ toàn pipeline; '
              'phần xếp hạng và overhead nằm ngoài các cột công đoạn.', '',
              '### Theo bố cục', '', '| Bố cục | SVM exact | PCA/FLD/HMM exact | SVM CER | PCA/FLD/HMM CER |',
              '|---|---:|---:|---:|---:|']
    for layout, m in a['by_layout'].items():
        n = b['by_layout'][layout]
        lines.append(f'| {layout} | {pct(m["exact_match"])} | {pct(n["exact_match"])} | {pct(m["character_error_rate"])} | {pct(n["character_error_rate"])} |')
    lines += ['', '## 3. Chẩn đoán crop GT và ký tự', '',
              'Crop GT bỏ detector nhưng vẫn chạy phân đoạn, bộ lọc, HOG và chọn giả thuyết OCR. '
              'Đây là chẩn đoán, không phải kết quả triển khai toàn ảnh.', '',
              '| Nhánh | Crop GT exact | Crop GT CER | Mean ms/crop |', '|---|---:|---:|---:|']
    for name, label in LABELS.items():
        m = summary['oracle_crop'][name]
        lines.append(f'| {label} | {pct(m["decoded"]["exact_match"])} | {pct(m["decoded"]["character_error_rate"])} | {m["latency"]["mean_ms"]:.2f} |')
    g = summary['glyph_diagnostic']
    lines += ['', f'Phân loại ký tự riêng dùng {g["acceptance"]["glyphs"]} glyph có nhãn yếu, '
              f'được nhận từ {g["acceptance"]["accepted_plates"]}/{g["acceptance"]["plates"]} crop GT '
              'bởi segmentation gốc để tương thích tập train. Chỉ nhận biển khớp count và rows; '
              'đây là tập chọn lọc dễ hơn, không thay thế đánh giá chuỗi trên mọi ảnh. '
              f'Có {g["exact_train_glyph_duplicates"]} glyph trùng hash chuẩn hóa với train; '
              'báo thêm accuracy sau khi loại các glyph này.', '',
              '| Bộ phân loại | Accuracy | Macro F1 | Accuracy không trùng train | Median ms/batch predict_proba |',
              '|---|---:|---:|---:|---:|']
    for name, m in g['models'].items():
        nonduplicate = pct(m['nonduplicate_accuracy']) if m['nonduplicate_accuracy'] is not None else 'N/A'
        lines.append(f'| {LABELS[name]} | {pct(m["accuracy"])} | {pct(m["macro_f1"])} | {nonduplicate} | {m["batch_median_ms"]:.2f} |')
    lines += ['', 'Timing batch dùng HOG đã tính trước, gồm phép chiếu và predict_proba; không gồm HOG và HMM. '
              'Kết quả không tương đương timing predict() một lần trong notebook tham khảo.', '',
              '## 4. Diễn giải và giới hạn', '']
    lines += [f'Ở lần chạy này, hệ hiện tại đúng {a["exact_count"]} biển; nhánh đầy đủ đúng {b["exact_count"]} biển. '
              f'Tốc độ toàn pipeline thay đổi {latency["speedup"]:.3f}×. '
              'Đánh giá việc thay thế cần căn cứ cả sai số chuỗi lẫn độ trễ, không suy từ số chiều đặc trưng. '
              'Chưa thực hiện kiểm định không thua kém vì chưa định trước biên chấp nhận sai số.', '',
              f'Bộ phân loại trong pipeline giảm từ {a["stages_mean_ms"]["classifier"]:.2f} xuống '
              f'{b["stages_mean_ms"]["classifier"]:.2f} ms/ảnh, trong khi detector và phân đoạn '
              f'đã chiếm khoảng {a["stages_mean_ms"]["detector"] + a["stages_mean_ms"]["segmentation"]:.2f} ms/ảnh. '
              'Đây là nguyên nhân mức tăng tốc toàn hệ thống nhỏ hơn nhiều so với đo phân loại ký tự riêng. '
              f'Trên batch {g["acceptance"]["glyphs"]} glyph, tốc độ predict_proba tăng '
              f'{g["models"]["current_svm"]["batch_median_ms"]/g["models"]["pca_fld_svm"]["batch_median_ms"]:.2f}×; '
              f'accuracy thay đổi từ {pct(g["models"]["current_svm"]["accuracy"])} xuống '
              f'{pct(g["models"]["pca_fld_svm"]["accuracy"])}.', '',
              f'Thêm HMM vào nhánh giảm chiều làm số biển đúng thay đổi từ '
              f'{summary["metrics"]["pca_fld_svm"]["exact_count"]} xuống {b["exact_count"]}. '
              'Cấu hình HMM này chưa cho thấy lợi ích trên test C. '
              + ('CER của nhánh đầy đủ tăng với CI95% chênh lệch nằm trên 0; mức giảm accuracy ký tự nhỏ không bảo đảm giữ chất lượng chuỗi. '
                 if clo > 0 else 'Khoảng tin cậy CER cần được xem cùng exact match trước khi cân nhắc thay thế. ')
              + 'Giữ bản phát hành SVM hiện tại; chỉ xem nhánh giảm chiều là ứng viên nghiên cứu. '
              'Mọi tối ưu tiếp theo nên chọn trên validation và xác minh bằng holdout mới, không chọn lại theo test C.', '',
              'Đối chiếu hệ SVM chạy lại với dự đoán bản phát hành lịch sử: '
              f'{len(history)}/{len(history)} chuỗi raw, decoded và bbox trùng khớp. '
              'Danh sách bbox/score detector cũng trùng nhau trên mọi ảnh giữa bốn nhánh. '
              'Mọi ảnh không phát hiện được biển hoặc tách sai vẫn nằm trong mẫu số toàn pipeline. '
              'Không tự động thay thế bản phát hành đang dùng.', '',
              'Các giới hạn: test C đã từng được quan sát; nhãn glyph là nhãn yếu; '
              'nguồn gốc augmentation legacy và quan hệ nguồn ảnh legacy–VNLP chưa xác minh; '
              'chưa có holdout theo camera. Thời gian phản ánh CPU/môi trường cục bộ và inference model đã nạp; '
              'không đánh giá độ trễ URL hoặc thời gian khởi động.', '',
              '### Ví dụ thay đổi kết quả', '',
              '| Trường hợp | Nhãn chuẩn | SVM hiện tại | PCA/FLD/SVM/HMM |', '|---|---|---|---|']
    for kind, condition in [('Chỉ hệ hiện tại đúng', lambda a, b: a['text'] == a['truth'] and b['text'] != b['truth']),
                            ('Chỉ nhánh mới đúng', lambda a, b: a['text'] != a['truth'] and b['text'] == b['truth'])]:
        selected = [(x, y) for x, y in zip(rows['current_svm'], rows['pca_fld_svm_hmm']) if condition(x, y)][:4]
        for x, y in selected:
            lines.append(f'| {kind} | {x["truth"]} | {x["text"] or "∅"} | {y["text"] or "∅"} |')
    lines += ['',
              '## 5. Tái lập và tài nguyên', '', '```powershell',
              '.\\.venv\\Scripts\\python.exe scripts/run_pca_fld_hmm_comparison.py --run reports/my-pca-fld-hmm-run --repeats 3',
              '.\\.venv\\Scripts\\python.exe scripts/build_pca_fld_hmm_report.py --run reports/my-pca-fld-hmm-run', '```', '',
              'Notebook phân tích chỉ đọc số liệu đã lưu; có thể thực thi bằng '
              '`python scripts/run_notebook.py 05_pca_fld_hmm_pipeline_comparison` '
              'sau khi cài kernel `plate-ml-314` như mục tái lập môi trường.', '',
              'Cần phục hồi Registry và các manifest lịch sử theo [quy trình tái lập](reproduction.md). '
              'Có thể chỉ định `--vnlp-root`, `--ocr-root`, `--pca-components` và `--reference-notebook`. '
              'Thay cấu hình tạo thí nghiệm mới; không chọn cấu hình tốt nhất bằng test C.', '',
              f'Run cục bộ: `reports/{run.name}`. Gồm protocol khóa trước test, audit train/split, '
              'manifest có hash, snapshot nguồn, candidate.joblib, dự đoán từng ảnh cho bốn nhánh và crop GT, '
              'thời gian từng lượt, chẩn đoán glyph và summary.json. '
              'Notebook phân tích được sinh tại `notebooks/05_pca_fld_hmm_pipeline_comparison.ipynb`.', '',
              f'Protocol SHA-256: `{summary["protocol_sha256"]}`.', '']
    report = '\n'.join(lines)
    destination = ROOT / 'docs/pca-fld-hmm-comparison.md'
    destination.write_text(report, encoding='utf-8')
    compact = dict(protocol=protocol, training=training, split=split, summary=summary,
                   summary_sha256=sha256((run / 'summary.json').read_bytes()).hexdigest())
    (ROOT / 'docs/results/pca-fld-hmm-180-20261026.json').write_text(json.dumps(compact, ensure_ascii=False, indent=2), encoding='utf-8')
    notebook = nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(report), nbf.v4.new_code_cell(
        "from pathlib import Path\nimport json\nimport pandas as pd\nimport matplotlib.pyplot as plt\n"
        "ROOT = Path.cwd().resolve()\nif not (ROOT / 'pyproject.toml').exists(): ROOT = ROOT.parent\n"
        f"data = json.loads((ROOT / 'docs/results/pca-fld-hmm-180-20261026.json').read_text(encoding='utf-8'))\n"
        "summary = data['summary']\n"
        "table = pd.DataFrame({k: {'exact_match': v['decoded']['exact_match'], 'CER': v['decoded']['character_error_rate'], 'mean_ms': v['latency']['mean_ms'], 'p95_ms': v['latency']['p95_ms']} for k,v in summary['metrics'].items()}).T\n"
        "display(table)\n"
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n"
        "table[['exact_match', 'CER']].plot.bar(ax=axes[0], rot=20, title='Full pipeline accuracy (180 scenes)')\n"
        "axes[0].set_ylim(0, 0.5)\n"
        "table[['mean_ms', 'p95_ms']].plot.bar(ax=axes[1], rot=20, title='Full pipeline latency (ms)')\n"
        "plt.tight_layout()\n"
        f"fig.savefig(ROOT / 'reports/{run.name}/pipeline_comparison.png', dpi=150, bbox_inches='tight')\n"
        "plt.show()\n"), nbf.v4.new_code_cell(
        "stages = pd.DataFrame({k: v['stages_mean_ms'] for k,v in summary['metrics'].items()}).T\n"
        "display(stages)\nax = stages.plot.bar(stacked=True, figsize=(12,5), rot=20, title='Stage latency (ms per scene)')\n"
        "ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1))\n"
        "plt.tight_layout()\n"
        f"ax.figure.savefig(ROOT / 'reports/{run.name}/stage_latency.png', dpi=150, bbox_inches='tight')\n"
        "plt.show()\n")])
    notebook.metadata.kernelspec = dict(display_name='Python 3 (PlateDetect)', language='python', name='python3')
    path = ROOT / 'notebooks/05_pca_fld_hmm_pipeline_comparison.ipynb'
    path.parent.mkdir(exist_ok=True)
    nbf.write(notebook, path)
    print(destination)
    print(path)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    build(parser.parse_args().run)
