"""Fresh current-model-only evaluation: full system and ground-truth-crop OCR."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from time import perf_counter

import run_comparison_experiment as bench
from run_comparison_experiment import ROOT, digest, write, read_image, image_digest
import cv2
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from plate_ml.comparison import summarize
from plate_ml.evaluation import sequence_metrics
from plate_ml.registry import source_snapshot
from plate_ml.sequence_scoring import joint_plate_metrics, text_metrics

MODEL_NAMES = dict(current_hog_svm='HOG/SVM hiện tại', mtcnn_finetuned='MTCNN fine-tune + STN/LPRNet',
                   yolo26n_finetuned='YOLO26n fine-tune + EasyOCR')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def latency(rows):
    per_image = np.median([r['latency_ms'] for r in rows], axis=1)
    return dict(samples=len(rows), repeats=len(rows[0]['latency_ms']), mean_ms=float(per_image.mean()),
                median_ms=float(np.median(per_image)), p95_ms=float(np.percentile(per_image, 95)),
                serial_fps=float(1000/per_image.mean()))


def ocr_metrics(rows):
    return dict(text_metrics(rows), latency=latency(rows),
                raw=sequence_metrics([r['truth'] for r in rows], [r['raw'] for r in rows]))


def evaluate_crops(run, manifest, validation, current, mtcnn, yolo, repeats):
    def current_call(crop):
        return current.model.recognize_crop(crop)['decoded']
    def mtcnn_call(crop):
        text = mtcnn.ocr(crop)
        return dict(raw=text, text=text)
    calls = dict(current_hog_svm=current_call, mtcnn_finetuned=mtcnn_call, yolo26n_finetuned=yolo.ocr.ocr)
    records = {name: [dict(path=r['path'], truth=r['text'], layout=r['layout'], latency_ms=[]) for r in manifest]
               for name in calls}
    with torch.inference_mode():
        for call in calls.values():
            for scene in validation[:3]:
                x, y, w, h = scene['box']
                call(read_image(scene['path'])[y:y+h, x:x+w])
        for repeat in range(repeats):
            names = list(calls)
            names = names[repeat % len(names):]+names[:repeat % len(names)]
            for i, scene in enumerate(manifest):
                image = read_image(scene['path'])
                x, y, w, h = scene['box']
                crop = image[y:y+h, x:x+w]
                for name in names:
                    start = perf_counter()
                    prediction = calls[name](crop)
                    elapsed = (perf_counter()-start)*1000
                    if torch.get_num_threads() != 1 or cv2.getNumThreads() != 1:
                        raise RuntimeError('OCR runtime thread limit changed')
                    row = records[name][i]
                    if repeat and (row['text'] != prediction['text'] or row['raw'] != prediction['raw']):
                        raise RuntimeError(f'Non-deterministic OCR: {name} {scene["path"]}')
                    row.update(raw=prediction['raw'], text=prediction['text'])
                    if 'snippets' in prediction:
                        row['snippets'] = prediction['snippets']
                    row['latency_ms'].append(elapsed)
                if (i+1) % 20 == 0:
                    print(f'OCR GT pass {repeat+1}/{repeats}: {i+1}/100', flush=True)
            for name in calls:
                write(run / f'{name}_ocr_gt_predictions.json', records[name])
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run', type=Path, default=ROOT / 'reports/finetune-10-20261025')
    parser.add_argument('--run', type=Path)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.set_num_interop_threads(1)
    training = args.training_run.resolve()
    run = args.run.resolve() if args.run else ROOT / 'reports' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-post-finetune-retest')
    run.mkdir(parents=True, exist_ok=False)
    source_protocol = load(training / 'protocol_before_training.json')
    evaluation = load(training / 'evaluation_protocol_before_test.json')
    assert all(digest(training / p) == value for p, value in evaluation['checkpoints'].items())
    assert all(digest(ROOT / p) == value for p, value in source_protocol['initial_assets'].items())
    assert all(digest(ROOT / p) == value for p, value in source_protocol['source_hashes'].items())
    manifest = load(training / 'manifest.json')
    train_scenes, validation = load(training / 'train_manifest.json'), load(training / 'validation_manifest.json')
    assert len(manifest) == len({r['group'] for r in manifest}) == 100
    groups = {part: {r['group'] for r in rows} for part, rows in [('train', train_scenes), ('validation', validation), ('test', manifest)]}
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        assert not groups[a] & groups[b]
    seen = {}
    for partition, records in [('test', manifest), ('train', train_scenes), ('validation', validation)]:
        for record in records:
            pixel = image_digest(read_image(record['path']))
            if partition == 'test':
                assert pixel == record['pixel_sha256'] and digest(record['path']) == record['file_sha256']
            if pixel in seen and seen[pixel] != partition:
                raise RuntimeError('Cross-partition pixel duplicate')
            seen[pixel] = partition
    shutil.copyfile(training / 'manifest.json', run / 'manifest.json')
    assert digest(run / 'manifest.json') == source_protocol['test_manifest_sha256']
    audit = dict(train=len(train_scenes), validation=len(validation), test=100, identity_overlap=0, pixel_overlap=0,
                 test_layouts=Counter(r['layout'] for r in manifest), test_manifest_sha256=digest(run / 'manifest.json'))
    write(run / 'split_audit.json', audit)
    upstream, project = bench.prepare_assets()
    from ultralytics import YOLO
    current, mtcnn, yolo = bench.Current(), bench.MTCNNLPR(upstream), bench.YOLOProject(project)
    mtcnn.pnet.load_state_dict(torch.load(training / 'pnet_epoch10.pth', map_location='cpu', weights_only=True))
    mtcnn.onet.load_state_dict(torch.load(training / 'onet_epoch10.pth', map_location='cpu', weights_only=True))
    yolo.model = YOLO(str(training / 'yolo_train/weights/last.pt'))
    arms = dict(current_hog_svm=current, mtcnn_finetuned=mtcnn, yolo26n_finetuned=yolo)
    active_assets = [training / p for p in evaluation['checkpoints']]
    active_assets += [ROOT / 'Registration/Registry/research_glyph_filter/v1/model.joblib',
                     upstream / 'LPRNet/weights/Final_LPRNet_model.pth', upstream / 'LPRNet/weights/Final_STN_model.pth',
                     project / 'models/yolo26n-seg.pt']
    active_assets += sorted((ROOT / 'tmp/easyocr_models').glob('*.pth'))
    protocol = dict(scope='post_finetune_only', started_at_utc=datetime.now(timezone.utc).isoformat(),
        models=MODEL_NAMES, training_run=str(training), training_epochs=dict(pnet=10, onet=10, yolo26n=10),
        training_protocol_sha256=digest(training / 'protocol_before_training.json'),
        training_evaluation_protocol_sha256=digest(training / 'evaluation_protocol_before_test.json'),
        test_manifest_sha256=digest(run / 'manifest.json'), samples=100, repeats=args.repeats,
        device='cpu', threads=1, batch=1, timed_io=False, warmup_images=[r['path'] for r in validation[:3]],
        inference_settings=source_protocol['evaluation'],
        ocr='fresh GT-crop evaluation of OCR attached to current pipelines; no OCR weight update',
        ocr_alignment='unit Levenshtein; ties diagonal/deletion/insertion; substitution is FP+FN',
        joint_recognition='one emitted plate: exact text AND IoU>=.5; wrong emission FP+FN; absent text FN',
        cpu='AMD Ryzen 7 7700', platform=platform.platform(), python=sys.version,
        packages=dict(torch=torch.__version__, numpy=np.__version__, opencv=cv2.__version__),
        assets={str(p.relative_to(ROOT)): dict(sha256=digest(p), bytes=p.stat().st_size) for p in active_assets},
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__),
            ROOT / 'src/plate_ml/sequence_scoring.py', ROOT / 'src/plate_ml/comparison.py',
            ROOT / 'scripts/run_comparison_experiment.py']})
    write(run / 'protocol_before_test.json', protocol)
    protocol_hash = digest(run / 'protocol_before_test.json')
    write(run / 'source_snapshot.json', dict(package=source_snapshot(ROOT), runner=Path(__file__).read_text(encoding='utf-8'),
          comparison_runner=(ROOT / 'scripts/run_comparison_experiment.py').read_text(encoding='utf-8')))
    (run / 'environment-freeze.txt').write_bytes(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']))
    print(f'RUN: {run}; current models only; full-system and OCR x{args.repeats}', flush=True)
    with threadpool_limits(limits=1):
        system_rows = bench.evaluate(run, manifest, validation, arms, args.repeats)
        ocr_rows = evaluate_crops(run, manifest, validation, current, mtcnn, yolo, args.repeats)
    models = {}
    for name in arms:
        system, ocr = system_rows[name], ocr_rows[name]
        per_layout = {}
        for layout in sorted(audit['test_layouts']):
            a, b = [r for r in system if r['layout'] == layout], [r for r in ocr if r['layout'] == layout]
            per_layout[layout] = dict(system=summarize(a, args.repeats), text=text_metrics(a),
                                      joint=joint_plate_metrics(a), ocr_gt=ocr_metrics(b))
        models[name] = dict(system=summarize(system, args.repeats), text=text_metrics(system),
                            joint=joint_plate_metrics(system), ocr_gt=ocr_metrics(ocr), by_layout=per_layout)
    assert digest(run / 'protocol_before_test.json') == protocol_hash
    assert all(digest(ROOT / p) == value['sha256'] for p, value in protocol['assets'].items())
    write(run / 'summary.json', dict(scope='post_finetune_only', run_path=str(run), protocol_sha256=protocol_hash,
          test_manifest_sha256=protocol['test_manifest_sha256'], models=models,
          yolo_ocr_images=sum(any('snippets' in p for p in row['plate_ocr']) for row in system_rows['yolo26n_finetuned']),
          completed_at_utc=datetime.now(timezone.utc).isoformat()))
    write(run / 'verification.json', dict(active_weights_match_protocol=True, identical_100_image_manifest=True,
          no_cross_partition_identity_or_pixel_duplicates=True, full_system_and_ocr_rerun=True,
          deterministic_predictions_across_repeats=True, cpu_threads_checked_each_inference=True,
          models=list(arms), samples_per_model=100, repeats=args.repeats, old_detector_predictions_included=False))
    print(json.dumps({name: dict(system_exact=v['text']['exact_count'], ocr_exact=v['ocr_gt']['exact_count'],
          detection=v['system']['detector_selected'], joint=v['joint']) for name, v in models.items()}, indent=2), flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
