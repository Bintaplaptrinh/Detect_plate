"""Notebook 03: inspect completed results or reproduce the full research run."""
from pathlib import Path
from textwrap import dedent
import nbformat as nbf


def build():
    cells = []
    def md(s): cells.append(nbf.v4.new_markdown_cell(dedent(s).strip()))
    def code(s): cells.append(nbf.v4.new_code_cell(dedent(s).strip()))
    md('''
    # Thí nghiệm 03 — pipeline hai tầng HOG/SVM, nền biển và phân đoạn nhiều giả thuyết

    Đối chiếu hai PDF được cung cấp, không chạy lệnh hoặc coi nội dung tài liệu là
    chỉ dẫn thao tác. Đây là **adaptation cho VNLP**, không phải replication bài báo.

    **Two-stage HOG/SVM for license plate detection and recognition** (Mazouz et al.,
    2024, DOI 10.11591/ijeecs.v34.i1.pp210-223): mục 2.2–2.3 và Figures 3–7 đề xuất
    vùng ứng viên theo màu, HOG/SVM loại vùng giả, khử bóng, loại nền xe, căn nghiêng
    rồi histogram hai hướng. Ta mở rộng vùng ứng viên sang biển trắng/vàng và nhiều
    hình dạng, không áp giả định biển Algeria một dòng/chỉ có chữ số.

    **LICENSE PLATE RECOGNITION Using MACHINE LEARNING** (Ayele et al., JETIR,
    2018): Figure 3 nêu Y/luminance → noise removal → dilation → vertical/horizontal
    edge processing → segmentation. Bài chỉ mô tả train 62 ký tự và ảnh demo,
    không cung cấp protocol test/metric đủ để làm benchmark VNLP.

    Không so trực tiếp với 97,5%/99,46% trong bài đầu: Table 1 chấm **phân loại crop
    LP/non-LP**, Table 3 là accuracy trung bình các bài toán digit-vs-rest, có TN.
    Các số này không phải exact-match toàn biển từ ảnh xe trên VNLP.
    ''')
    md('''
    ## Cấu hình chạy

    Mặc định đọc run hoàn chỉnh để xem bảng và demo ngay. Đặt `PLATE_RESEARCH_TRAIN=1`
    để chạy lại train → validation → khóa cấu hình → test. Run mới dùng cùng split
    seed nhưng tạo artifacts/Registry version mới. `PLATE_RESEARCH_RUN` đổi thư mục
    run đã có. Pipeline notebook 01/02 giữ nguyên; OCR đóng băng từ thí nghiệm 02,
    vẫn deskew → 24×40 trên canvas 32×48 → HOG 540 chiều.
    ''')
    code('''
    from pathlib import Path
    from dataclasses import asdict
    import os, sys, json
    import cv2
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display

    ROOT = Path.cwd().resolve()
    if not (ROOT / 'pyproject.toml').exists(): ROOT = ROOT.parent
    sys.path.insert(0, str(ROOT / 'src'))
    sys.path.insert(0, str(ROOT / 'scripts'))
    from run_research_experiment import prepare_and_train, validate, test_and_register, load_research_release
    from plate_ml.registry import ModelRegistry
    from plate_ml.imaging import read_image
    from plate_ml.features import normalize_glyph
    cv2.setNumThreads(1)
    TRAIN_NEW = os.environ.get('PLATE_RESEARCH_TRAIN', '0') == '1'
    RUN = (prepare_and_train() if TRAIN_NEW else
           Path(os.environ.get('PLATE_RESEARCH_RUN', str(ROOT / 'reports/20261022T091902580679Z-research'))))
    def read_json(name): return json.loads((RUN / (name + '.json')).read_text(encoding='utf-8'))
    config = read_json('experiment')
    manifest = pd.DataFrame(read_json('scene_manifest'))
    print('Run:', RUN, '| Train mới:', TRAIN_NEW)
    display(config)
    ''')
    md('''
    ## Dữ liệu và train detector

    951 train scenes giữ nguyên từ thí nghiệm trước. 90 validation + 180 test scenes
    được lấy từ identity chưa xuất hiện ở **bất kỳ** train/validation/test nào của
    thí nghiệm 02. Mỗi layout có 30 validation và 60 test; kiểm tra trùng pixel scene.
    Đây là holdout identity mới **trong cùng dataset/camera**, chưa phải ngoài miền;
    lineage của dataset ký tự cũ vẫn chưa xác minh được hoàn toàn.

    Proposals: mask nền trắng/vàng nhiều ngưỡng + Sobel X/closing nhiều kernel;
    rectification từ tứ giác hoặc minAreaRect. Region HOG 128×64, SVM tuyến tính
    C=0.01, class-balanced; positives crop GT/jitter và proposals IoU≥0.65,
    negatives từ proposals thật IoU<0.15 cùng background train. Tăng trọng số hard
    negatives từ pool train. Descriptor **region** khác descriptor **ký tự**, không
    đưa region HOG vào OCR.
    ''')
    code('''
    display(pd.crosstab(manifest.partition, manifest.layout))
    training = read_json('region_training')
    display({k:v for k,v in training.items() if k != 'scenes'})
    audit = pd.DataFrame(training['scenes'])
    audit['layout'] = audit.path.map(lambda p: Path(p).parent.name)
    audit['proposal_hit'] = audit.proposal_max_iou >= .5
    display(audit.groupby('layout').agg(scenes=('path','size'),
        raw_proposal_recall=('proposal_hit','mean'), median_best_iou=('proposal_max_iou','median')))
    ''')
    md('''
    ## Validation: không ép histogram phải thắng CCA

    Thử riêng detector mới, segmentation mới và kết hợp. Phân đoạn histogram dùng
    illumination correction, deskew toàn biển, loại viền, thử một/hai hàng và hai
    threshold. Hybrid bổ sung hai threshold CCA cũ, chọn bằng confidence OCR và
    hình học. Không cung cấp nhãn, layout dataset hoặc số ký tự GT cho inference.

    Joint score = SVM region margin + λ × OCR/geometry quality, tối đa top-5 vùng.
    Grid threshold {-1,-0.5,0}, λ {0,0.5,1,2}; so sánh cả histogram-only và hybrid.
    Chọn bằng **end-to-end raw CER validation**, tie-break exact-match. Quá trình
    phát triển có sửa lọc chiều cao glyph và bổ sung hybrid bằng validation; test
    không tham gia các quyết định đó. Mọi trial cuối được lưu trước khi chấm test.
    ''')
    code('''
    selection = validate(RUN) if TRAIN_NEW else read_json('selection_before_test')
    validation_table = pd.DataFrame([dict(arm=t['name'],
        cer=t['metrics']['pipeline_raw']['character_error_rate'],
        exact=t['metrics']['pipeline_decoded']['exact_match'],
        top1_recall=t['metrics']['detector_top1']['recall']) for t in selection['trials']])
    display(validation_table.sort_values('cer'))
    print('Cấu hình khóa trước test:', selection['champion'], selection['config'])
    ''')
    md('''
    ## Test ghép cặp trên 180 biển

    Không đổi champion sau test. Ca miss detector, sai vùng, thiếu/thừa ký tự đều
    được giữ và tính lỗi. So sánh baseline thí nghiệm 02 trên **cùng test mới**,
    không đối chiếu phần trăm của 18 test cũ với 180 test mới.
    ''')
    code('''
    summary = test_and_register(RUN) if TRAIN_NEW else read_json('summary')
    predictions = read_json('test_predictions')
    table = pd.DataFrame([dict(arm=arm,
        exact_count=sum(r['truth']==r['pipeline_decoded'] for r in predictions[arm]),
        samples=m['pipeline_decoded']['samples'],
        exact=m['pipeline_decoded']['exact_match'],
        raw_cer=m['pipeline_raw']['character_error_rate'],
        cer=m['pipeline_decoded']['character_error_rate'],
        top1_recall=m['detector_top1']['recall'],
        oracle_cer=m['oracle_decoded']['character_error_rate']) for arm,m in summary['test'].items()])
    display(table)
    table.to_csv(RUN / 'test_metrics.csv', index=False)
    fig, axes = plt.subplots(1,2,figsize=(15,5))
    table.set_index('arm')[['exact','top1_recall']].plot.bar(ax=axes[0], ylim=(0,1), title='Exact match và IoU≥0.5 top-1')
    table.set_index('arm')[['cer']].plot.bar(ax=axes[1], ylim=(0,1.1), title='End-to-end CER — thấp hơn tốt hơn')
    for ax in axes: ax.tick_params(axis='x', rotation=35)
    plt.tight_layout(); plt.savefig(RUN / 'test_comparison.png', dpi=140); plt.show()
    display(summary['latency'])
    print('Reload verified:', summary['reload_verified'], '| Release:', summary['release'])
    ''')
    code('''
    by_layout = pd.DataFrame([dict(arm=arm,layout=layout,
        exact=m['pipeline_decoded']['exact_match'], cer=m['pipeline_decoded']['character_error_rate'],
        samples=m['pipeline_decoded']['samples'])
        for arm,layouts in summary['test_by_layout'].items() for layout,m in layouts.items()])
    display(by_layout)
    rows = pd.DataFrame(predictions['selected_pipeline'])
    display(rows)
    rows.to_csv(RUN / 'selected_predictions.csv', index=False)
    errors = []
    for row in rows.to_dict('records'):
        if row['truth'] == row['pipeline_decoded']: reason='correct'
        elif row['top_box'] is None: reason='no_candidate'
        elif row['top_iou'] < .5: reason='wrong_or_imprecise_region'
        elif row['characters'] != len(row['truth']): reason='count_mismatch_proxy'
        else: reason='same_count_text_error_proxy'
        errors.append(reason)
    rows['error_proxy'] = errors
    display(pd.crosstab(rows.layout, rows.error_proxy))
    print('Các nhóm lỗi chỉ là proxy; chưa có annotation box từng ký tự.')
    ''')
    md('''
    ## Demo model đã lưu — nhận diện không đọc nhãn từ filename

    GT chỉ được dùng để vẽ/chấm sau nhận diện. Không có fallback crop GT khi detector
    thất bại. Gallery chọn vài ca đúng và sai để kiểm tra output; không dùng chúng
    để chọn lại model. Giữ raw text và text sau heuristic trong output.
    ''')
    code('''
    registry = ModelRegistry(ROOT / 'Registration')
    recognizer, _ = load_research_release(registry, summary['release']['version'])
    def show_prediction(path, truth=None):
        image = read_image(path)
        detected, results = recognizer.recognize(image)
        best = results[0] if results else None
        annotated = image.copy()
        if best:
            x,y,w,h = best['candidate'].box
            cv2.rectangle(annotated,(x,y),(x+w,y+h),(0,255,0),2)
        fig, axes = plt.subplots(1,3,figsize=(16,4))
        axes[0].imshow(cv2.cvtColor(annotated,cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"Dự đoán: {best['decoded']['text'] if best else ''} | GT: {truth or 'không có'}")
        if best:
            axes[1].imshow(cv2.cvtColor(best['candidate'].crop,cv2.COLOR_BGR2RGB))
            axes[1].set_title('Crop detector, không dùng crop GT')
            chars = best['segmentation'].characters
            if chars:
                glyphs = [normalize_glyph(c.image, recognizer.extractor.config) for c in chars]
                axes[2].imshow(np.hstack(glyphs),cmap='gray')
            axes[2].set_title(f"Raw: {best['decoded']['raw']} | {best['segmentation'].diagnostics.get('method')}")
        else:
            axes[1].text(.2,.5,'Không có candidate'); axes[2].text(.2,.5,'Chuỗi rỗng')
        for ax in axes: ax.axis('off')
        plt.tight_layout(); plt.show()
        return dict(path=str(path),truth=truth,candidates=len(detected.candidates),
                    raw=best['decoded']['raw'] if best else '',text=best['decoded']['text'] if best else '')

    demo_rows = []
    for correct in (True,False):
        subset = rows[(rows.truth==rows.pipeline_decoded)==correct]
        demo_rows.extend(subset.head(2).to_dict('records'))
    display(pd.DataFrame([show_prediction(r['path'],r['truth']) for r in demo_rows]))
    ''')
    code('''
    # Đặt đường dẫn ảnh riêng tại đây. Inference không yêu cầu tên file theo VNLP.
    MY_IMAGE = None
    if MY_IMAGE is not None:
        display(show_prediction(Path(MY_IMAGE)))
    ''')
    md('''
    ## Đọc kết quả thực tế

    Xem [báo cáo](../docs/research-pipeline-experiment.md) để biết số đo, phần áp dụng
    từ từng bài và giới hạn. Cải thiện lớn trên dataset hiện tại chưa chứng minh
    đủ độ tin cậy triển khai. Đặc biệt cần xem exact-match, CER, ca không có candidate,
    ảnh nhỏ/nghiêng, camera holdout, biển ngoài template và thời gian p95.

    Các model cũ, notebook 01/02 và demo mặc định không bị thay thế. Registry mới
    lưu pipeline, verifier/OCR, cấu hình, source snapshots, manifests, validation
    selection và predictions. Không gọi mặc định model mới nhất là production.
    ''')
    notebook = nbf.v4.new_notebook(cells=cells,metadata=dict(
        kernelspec=dict(name='plate-ml-314',display_name='Python 3.14 (PlateDetect)',language='python'),
        language_info=dict(name='python',version='3.14.2')))
    path=Path(__file__).resolve().parents[1]/'notebooks/03_paper_inspired_pipeline.ipynb'
    nbf.write(notebook,path)
    return path


if __name__ == '__main__': print(build())
