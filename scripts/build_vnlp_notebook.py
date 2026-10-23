"""Build the independent OCR training-data ablation; leave notebook 01 untouched."""
from pathlib import Path
from textwrap import dedent

import nbformat as nbf


def build():
    cells = []

    def md(text):
        cells.append(nbf.v4.new_markdown_cell(dedent(text).strip()))

    def code(text):
        cells.append(nbf.v4.new_code_cell(dedent(text).strip()))

    md('''
    # Thí nghiệm 02 — OCR học từ ký tự cắt trên VNLP

    Chỉ tiếp thu **ý tưởng nguồn dữ liệu train** từ `biensoxe-modelsvm.ipynb`.
    Tái sử dụng nguyên `CharacterSegmenter`, `normalize_glyph`, `HOGExtractor` và
    `make_svm` của notebook 01: nhị phân → crop nét → padded deskew → **24×40 trên
    canvas 32×48 → OpenCV HOG 540 chiều**. Không lấy 28×28, zoning, threshold,
    detector hay hậu xử lý của notebook tham khảo.

    Ba nhánh: `baseline` (pipeline/v3), `vnlp_only`, `legacy_plus_vnlp`. Nhánh cuối
    trộn **nguồn ảnh train**, tất cả vẫn đi qua cùng một preprocessing của notebook 01.
    Detector, segmentation, HOG và heuristic đóng băng từ pipeline/v3.

    Giữ nguyên validation/test VNLP của detector/v3; không chia lại từng ký tự.
    Lấy thêm tối đa 300 scene/layout cho **train**, loại mọi identity thuộc heldout.
    Đặt `PLATE_VNLP_EXTRA_PER_LAYOUT=0` để chỉ dùng đúng 54 train scenes cũ;
    tăng giá trị để mở rộng. Vì số lượng/nguồn dữ liệu thay đổi, đây là ablation
    nguồn dữ liệu, không phải so sánh thuật toán trên cùng số mẫu.

    **Exploratory:** holdout cũ từng được quan sát, chỉ có 18 test scenes; nhãn ký tự
    từ filename + alignment là nhãn yếu. Count-match và row-match không chứng minh
    mọi crop đúng; cần kiểm tra gallery và annotation ký tự trước benchmark độc lập.
    ''')
    code('''
    from pathlib import Path
    from dataclasses import asdict
    from datetime import datetime, timezone
    from hashlib import sha256
    import json, os, sys, time
    import cv2
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display

    ROOT = Path.cwd().resolve()
    if not (ROOT / 'pyproject.toml').exists():
        ROOT = ROOT.parent
    assert (ROOT / 'src' / 'plate_ml').is_dir(), 'Mở từ Detect_plate hoặc notebooks'
    sys.path.insert(0, str(ROOT / 'src'))
    from plate_ml.domain_ocr import (assert_group_disjoint, remove_scene_duplicates,
        extract_vnlp_glyphs, glyph_metrics, cache_scene_features, evaluate_cached,
        check_baseline_pipeline)
    from plate_ml.features import HOGExtractor, normalize_glyph
    from plate_ml.imaging import read_image, gray_u8
    from plate_ml.models import make_svm
    from plate_ml.pipeline import PlateRecognizer
    from plate_ml.registry import ModelRegistry, code_info, source_snapshot, environment_info
    from plate_ml.vnlp import Scene, scan_vnlp, crop_scene
    from plate_ml.evaluation import sequence_metrics

    cv2.setNumThreads(1)
    SEED = 42
    BASELINE_VERSION = 'v3'
    EXTRA_PER_LAYOUT = int(os.environ.get('PLATE_VNLP_EXTRA_PER_LAYOUT', '300'))
    assert EXTRA_PER_LAYOUT >= 0
    VNLP_ROOT = Path(os.environ.get('PLATE_VNLP_ROOT', str(ROOT.parent / 'data/VNLP_detection/detection')))
    OCR_ROOT = Path(os.environ.get('PLATE_OCR_ROOT', str(ROOT / 'CNN/CNN_letter_Dataset')))
    RUN_ID = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-vnlp'
    OUT = ROOT / 'reports' / RUN_ID
    OUT.mkdir(parents=True, exist_ok=False)
    REGISTRY = ModelRegistry(ROOT / 'Registration')
    START = time.perf_counter()

    def save_json(name, data):
        (OUT / (name + '.json')).write_text(
            json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')

    bundle, baseline_receipt = REGISTRY.load('pipeline', BASELINE_VERSION, trusted=True)
    baseline = PlateRecognizer.from_bundle(bundle, REGISTRY.root / 'RuntimeAssets')
    check_baseline_pipeline(baseline)
    # Byte-for-byte source check for all frozen preprocessing / decoding components.
    snapshot = json.loads((Path(REGISTRY.root) / 'Registry/pipeline' / BASELINE_VERSION /
                           'source_snapshot.json').read_text(encoding='utf-8'))
    for name in ('features', 'segmentation', 'imaging', 'detection', 'heuristics', 'pipeline', 'config', 'models'):
        relative = f'src/plate_ml/{name}.py'
        assert (ROOT / relative).read_text(encoding='utf-8') == snapshot[relative], f'Baseline source changed: {name}'
    HOG, SEGMENTATION = baseline.extractor.config, baseline.segmenter.config
    assert HOG.size == (32, 48) and HOG.deskew and not HOG.preserve_aspect
    display(dict(run_id=RUN_ID, baseline=BASELINE_VERSION, extra_per_layout=EXTRA_PER_LAYOUT,
                 hog=asdict(HOG), segmentation=asdict(SEGMENTATION), environment=environment_info()))
    ''')
    md('''
    ## Chia scene trước khi cắt ký tự

    Đọc manifests đã khóa trong Registry, bảo toàn identity split. Bổ sung VNLP chỉ
    vào train, không lấy bất kỳ biến thể cùng biển nào ở validation/test. Loại scene
    trùng pixel, ưu tiên giữ heldout; lỗi trùng validation/test phải được giải quyết
    bằng split mới riêng, không âm thầm sửa test. Lưu danh sách toàn bộ scene đã dùng.
    ''')
    code('''
    _, detector_receipt = REGISTRY.load('detector', baseline_receipt['metadata']['detector_version'], trusted=True)
    detector_dir = Path(REGISTRY.root) / 'Registry/detector' / detector_receipt['version']
    manifest = json.loads((detector_dir / 'split_manifest.json').read_text(encoding='utf-8'))
    partitions = {name: [] for name in ('train', 'validation', 'test')}
    for row in manifest:
        path = VNLP_ROOT / row['layout'] / Path(row['path']).name
        partitions[row['partition']].append(Scene(str(path.resolve()), row['text'], tuple(row['box']), row['layout'], row['group']))
    original_sizes = {k: len(v) for k, v in partitions.items()}
    reserved = {s.group for name in ('validation', 'test') for s in partitions[name]}
    seen_paths = {s.path for group in partitions.values() for s in group}
    extra_audit = dict(selected=0)
    if EXTRA_PER_LAYOUT:
        extras, extra_audit = scan_vnlp(VNLP_ROOT, max_per_layout=EXTRA_PER_LAYOUT, seed=SEED)
        partitions['train'].extend(s for s in extras if s.group not in reserved and s.path not in seen_paths)
    assert_group_disjoint(partitions)
    partitions, duplicate_scenes = remove_scene_duplicates(partitions)
    assert_group_disjoint(partitions)
    assert len(partitions['validation']) == original_sizes['validation']
    assert len(partitions['test']) == original_sizes['test']
    scene_manifest = [dict(**asdict(s), partition=name) for name, scenes in partitions.items() for s in scenes]
    save_json('scene_manifest', scene_manifest)
    save_json('scene_duplicates', duplicate_scenes)
    save_json('extra_selection', extra_audit)
    display(pd.DataFrame([dict(partition=name, scenes=len(scenes), identities=len({s.group for s in scenes}))
                          for name, scenes in partitions.items()]))
    ''')
    md('''
    ## Thu ký tự và kiểm tra nhãn yếu

    Chạy đúng segmentation đã chọn ở notebook 01. Chỉ nhận **toàn bộ biển** có count
    khớp và thứ tự đọc trái→phải, trên→dưới. Biển một dòng phải có một hàng; biển hai
    dòng phải khớp row lengths của template tương thích với nhãn. Không cắt bớt,
    thêm crop hay dùng dự đoán baseline để gán nhãn. Biển lỗi vẫn được giữ khi tính
    exact-match/CER toàn biển, chỉ loại khỏi tập ký tự có nhãn yếu.
    ''')
    code('''
    glyphs = {name: extract_vnlp_glyphs(scenes, SEGMENTATION, HOG, baseline.templates)
              for name, scenes in partitions.items() if name != 'test'}
    assert len(glyphs['train'].y) > 0, 'Không có glyph train hợp lệ; xem segmentation audit'
    assert len(set(glyphs['train'].y)) >= 2, 'Cần ít nhất hai lớp cho SVM'
    for name, data in glyphs.items():
        save_json(name + '_glyph_manifest', data.records)
        save_json(name + '_segmentation_audit', data.plates)
    display(pd.DataFrame({name: data.summary() for name, data in glyphs.items()}).T)
    distribution = pd.DataFrame({name: pd.Series(data.summary()['classes']) for name, data in glyphs.items()}).fillna(0)
    display(distribution.astype(int))
    distribution.plot.bar(figsize=(14, 4), title='VNLP glyphs sau bộ lọc nhãn yếu')
    plt.tight_layout(); plt.show()
    ''')
    code('''
    # Gallery chọn từ train; kiểm tra cả ký tự gốc, row/position và normalized glyph.
    gallery = np.random.default_rng(SEED).choice(len(glyphs['train'].records),
              min(48, len(glyphs['train'].records)), replace=False)
    scene_lookup = {s.path: s for s in partitions['train']}
    fig, axes = plt.subplots(8, 6, figsize=(14, 17))
    for ax in axes.ravel(): ax.axis('off')
    for ax, index in zip(axes.ravel(), gallery):
        record = glyphs['train'].records[index]
        result = baseline.segmenter.segment(crop_scene(scene_lookup[record['path']]))
        char = result.characters[record['position']]
        ax.imshow(normalize_glyph(char.image, HOG), cmap='gray')
        ax.set_title(f"{record['label']} | {record['text']}\\npos={record['position']} row={record['row']}", fontsize=8)
    fig.suptitle('TRAIN — nhãn từ filename, cần kiểm tra thủ công')
    plt.tight_layout(); plt.savefig(OUT / 'train_glyph_gallery.png', dpi=140); plt.show()
    ''')
    md('''
    ## Train SVM — chỉ thay dữ liệu, giữ cấu hình HOG

    Nhánh `legacy_plus_vnlp` chỉ lấy **train** từ manifest OCR cũ; kiểm tra pixel hash
    trước khi dùng lại. Scaler và calibration nội bộ SVC chỉ fit trên train.
    Grid C={0.3,1,3}, kernel={linear,rbf} giống notebook 01. Chọn bằng **CER raw
    trên tất cả crop GT validation**, tie-break theo exact match, không chọn bằng test.
    Học lại SVM từ đầu, không gọi đây là fine-tuning/incremental learning.
    ''')
    code('''
    _, ocr_receipt = REGISTRY.load('ocr', baseline_receipt['metadata']['ocr_version'], trusted=True)
    ocr_dir = Path(REGISTRY.root) / 'Registry/ocr' / ocr_receipt['version']
    legacy_manifest = json.loads((ocr_dir / 'split_manifest.json').read_text(encoding='utf-8'))
    old_train = [r for r in legacy_manifest if r['partition'] == 'train']
    legacy_images = []
    for row in old_train:
        path = OCR_ROOT / row['label'] / Path(row['path']).name
        image = read_image(path)
        gray = gray_u8(image)
        digest = sha256(str(gray.shape).encode() + gray.tobytes()).hexdigest()
        assert digest == row['digest'], f'Legacy training pixels changed: {path}'
        legacy_images.append(image)
    legacy_X = HOGExtractor(HOG).transform(legacy_images)
    legacy_y = np.asarray([r['label'] for r in old_train])
    del legacy_images
    train_sources = {
        'vnlp_only': (glyphs['train'].X, glyphs['train'].y),
        'legacy_plus_vnlp': (np.vstack([legacy_X, glyphs['train'].X]), np.r_[legacy_y, glyphs['train'].y])}
    save_json('legacy_train_manifest', old_train)
    display(pd.DataFrame([dict(arm=name, samples=len(y), classes=len(set(y)),
                missing_baseline_classes=''.join(sorted(set(baseline.model.classes_) - set(y))))
                for name, (X, y) in train_sources.items()]))

    val_cache = cache_scene_features(partitions['validation'], baseline)
    models = {'baseline': baseline.model}
    trials, best_params = [], {}
    def selection_key(metrics):
        raw = metrics['oracle_raw']
        return raw['character_error_rate'], -raw['exact_match']
    for arm, (X_train, y_train) in train_sources.items():
        best_key = None
        for kernel in ('linear', 'rbf'):
            for c in (0.3, 1.0, 3.0):
                model = make_svm(c=c, kernel=kernel, seed=SEED).fit(X_train, y_train)
                metrics, _ = evaluate_cached(model, val_cache, baseline.templates)
                key = selection_key(metrics)
                trials.append(dict(arm=arm, C=c, kernel=kernel,
                                   oracle_raw_cer=key[0], oracle_raw_exact=-key[1],
                                   glyph_validation=glyph_metrics(model, glyphs['validation'])))
                print(arm, kernel, c, 'validation CER:', round(key[0], 4), flush=True)
                if best_key is None or key < best_key:
                    best_key, models[arm] = key, model
                    best_params[arm] = dict(C=c, kernel=kernel)
    validation = {name: evaluate_cached(model, val_cache, baseline.templates)[0] for name, model in models.items()}
    # Insertion order means baseline wins exact ties; no automatic promotion.
    champion = min(validation, key=lambda name: selection_key(validation[name]))
    selection = dict(champion=champion, criterion='validation oracle_raw CER then exact_match; baseline wins ties',
                     best_params=best_params, validation=validation)
    save_json('trials', trials)
    save_json('selection_before_test', selection)
    print('Champion khóa trước test:', champion)
    display(pd.DataFrame(trials))
    display(pd.DataFrame({name: m['oracle_raw'] for name, m in validation.items()}).T)
    ''')
    md('''
    ## Test ghép cặp — mở sau khi khóa lựa chọn

    Tất cả nhánh dùng đúng cùng top-1 detector và cùng crop/segmentation, kể cả
    thất bại. Không dùng nhãn filename hay count-match khi nhận diện. Chỉ dùng
    nhãn khi chấm điểm; không loại biển tách sai khỏi exact-match/CER.
    Ký tự test có nhãn yếu được báo riêng kèm tỷ lệ scene được nhận và lớp chưa học.
    Đây là so sánh ba nhánh đã định trước, không chọn lại champion theo test.
    ''')
    code('''
    assert json.loads((OUT / 'selection_before_test.json').read_text(encoding='utf-8'))['champion'] == champion
    glyphs['test'] = extract_vnlp_glyphs(partitions['test'], SEGMENTATION, HOG, baseline.templates)
    save_json('test_glyph_manifest', glyphs['test'].records)
    save_json('test_segmentation_audit', glyphs['test'].plates)
    test_cache = cache_scene_features(partitions['test'], baseline)
    test_metrics, predictions = {}, {}
    for arm, model in models.items():
        test_metrics[arm], predictions[arm] = evaluate_cached(model, test_cache, baseline.templates)
    # Ensure the controlled baseline reproduces the saved notebook 01 metrics.
    expected = baseline_receipt['metadata']['test']
    for key in expected:
        assert test_metrics['baseline'][key] == expected[key], f'Baseline mismatch: {key}'
    char_test = {arm: glyph_metrics(model, glyphs['test']) for arm, model in models.items()}
    display(glyphs['test'].summary())
    display(pd.DataFrame(char_test).T)
    table = pd.DataFrame([dict(arm=arm, stage=stage, **values)
                         for arm, metrics in test_metrics.items() for stage, values in metrics.items()])
    display(table)
    wide = pd.DataFrame([dict(arm=arm, **row) for arm, rows in predictions.items() for row in rows])
    display(wide)
    table.to_csv(OUT / 'test_metrics.csv', index=False)
    wide.to_csv(OUT / 'test_predictions.csv', index=False)
    save_json('test_predictions', predictions)
    layout_metrics = {arm: {layout: {stage: sequence_metrics(
        [r['truth'] for r in rows if r['layout'] == layout],
        [r[stage] for r in rows if r['layout'] == layout]) for stage in metrics}
        for layout in sorted({r['layout'] for r in rows})}
        for (arm, rows), metrics in zip(predictions.items(), test_metrics.values())}
    save_json('test_by_layout', layout_metrics)
    table.pivot(index='stage', columns='arm', values='character_error_rate').plot.bar(
        figsize=(12, 4), title='Test CER — cùng detector / segmentation; thấp hơn tốt hơn')
    plt.tight_layout(); plt.savefig(OUT / 'test_cer.png', dpi=140); plt.show()
    ''')
    md('''
    ## Lưu experiment riêng và kiểm tra reload

    Ghi model ứng viên vào namespace riêng, không đổi pipeline/v3 hay demo notebook
    01. Bundle champion giữ nguyên detector/config/cascade, chỉ thay OCR. Lưu
    manifests, audits, tuning, metrics, source snapshot và mã nguồn notebook/generator
    để tái lập. Reload kiểm tra probabilities và decoded text trên validation crop.
    ''')
    code('''
    data_summary = {name: data.summary() for name, data in glyphs.items()}
    releases = {}
    experiment_files = dict(source_snapshot=source_snapshot(ROOT), scene_manifest=scene_manifest,
        glyph_manifests={k: v.records for k, v in glyphs.items()},
        segmentation_audits={k: v.plates for k, v in glyphs.items()},
        legacy_train_manifest=old_train, selection=selection, tuning=trials,
        predictions=predictions, test_by_layout=layout_metrics,
        notebook_source=json.loads((ROOT / 'notebooks/02_vnlp_ocr_ablation.ipynb').read_text(encoding='utf-8')),
        generator_source={'source': (ROOT / 'scripts/build_vnlp_notebook.py').read_text(encoding='utf-8')})
    context = dict(run_id=RUN_ID, baseline_version=BASELINE_VERSION,
                   baseline_artifact_sha256=baseline_receipt['artifact_sha256'],
                   hog=asdict(HOG), segmentation=asdict(SEGMENTATION),
                   extra_per_layout=EXTRA_PER_LAYOUT, data=data_summary, code=code_info(ROOT),
                   evaluation_status='exploratory_reused_holdout_and_weak_glyph_labels',
                   selection=selection, test=test_metrics, glyph_test=char_test)
    for arm in train_sources:
        releases[arm] = REGISTRY.register('ocr_' + arm, dict(ocr_model=models[arm], hog=HOG),
            dict(**context, arm=arm, params=best_params[arm]), files=experiment_files)
    candidate_bundle = dict(bundle, ocr_model=models[champion])
    release = REGISTRY.register('pipeline_vnlp_experiment', candidate_bundle,
        dict(**context, champion=champion), files=experiment_files)
    restored_bundle, _ = REGISTRY.load('pipeline_vnlp_experiment', release['version'], trusted=True)
    restored = PlateRecognizer.from_bundle(restored_bundle, REGISTRY.root / 'RuntimeAssets')
    candidate = PlateRecognizer.from_bundle(candidate_bundle, REGISTRY.root / 'RuntimeAssets')
    probe = crop_scene(partitions['validation'][0])
    before, after = candidate.recognize_crop(probe), restored.recognize_crop(probe)
    np.testing.assert_allclose(before.probabilities, after.probabilities)
    assert before.decoded == after.decoded
    SUMMARY = dict(**context, champion=champion, scene_counts={k: len(v) for k, v in partitions.items()},
                   output=str(OUT), elapsed_seconds=round(time.perf_counter() - START, 2),
                   releases={k: dict(version=v['version'], path=v['path']) for k, v in releases.items()},
                   pipeline_release=dict(version=release['version'], path=release['path']), reload_verified=True)
    save_json('summary', SUMMARY)
    display(SUMMARY)
    print('Đã lưu:', OUT)
    ''')
    md('''
    ## Diễn giải

    - Đọc `oracle_raw` để thấy thay đổi OCR; `oracle_decoded` để thấy ảnh hưởng heuristic.
    - CER giảm nhưng exact-match không tăng có thể vì segmentation vẫn thiếu/thừa ký tự.
    - End-to-end còn chịu giới hạn recall detector; đổi OCR không cứu ảnh không có candidate.
    - VNLP-only có thể thiếu lớp hoặc quá ít chữ; xem support và `unseen_truth_classes`.
    - Glyph accuracy chỉ trên subset count/row-match có nhãn yếu; không coi là accuracy
      mọi ký tự VNLP và không so trực tiếp với 97,57% của dataset ký tự cũ.
    - Kết luận tăng/giảm phải dựa trên kết quả thật; không dùng test để chọn lại nhánh.
      Cần thêm annotation glyph, camera/session holdout mới cho kết luận tổng quát.
    ''')

    notebook = nbf.v4.new_notebook(cells=cells, metadata=dict(
        kernelspec=dict(display_name='Python 3.14 (PlateDetect)', language='python', name='plate-ml-314'),
        language_info=dict(name='python', version='3.14.2')))
    path = Path(__file__).resolve().parents[1] / 'notebooks/02_vnlp_ocr_ablation.ipynb'
    nbf.write(notebook, path)
    return path


if __name__ == '__main__':
    print(build())
