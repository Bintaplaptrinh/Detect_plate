"""Fine-tune both detector stages and YOLO26n for exactly ten epochs, then retest.

The previous 100-image manifest is immutable. OCR and inference settings stay
identical to run_comparison_experiment.py; final-epoch checkpoints are evaluated.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
from time import perf_counter

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import run_comparison_experiment as bench
from run_comparison_experiment import ROOT, write, digest, read_image, image_digest
import cv2
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from plate_ml.comparison import paired_difference, summarize
from plate_ml.mtcnn_training import encode_patch, mtcnn_loss, patch_label, sample_patch_boxes
from plate_ml.registry import source_snapshot

PREVIOUS = ROOT / 'reports/comparison-project-100-20261025-v2'
SEED = 2026102510
EPOCHS = 10


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def prepare(run):
    run.mkdir(parents=True, exist_ok=False)
    upstream, project = bench.prepare_assets()
    manifest = load(PREVIOUS / 'manifest.json')
    training = [r for r in load(bench.TRAIN_SOURCE) if r['partition'] == 'train']
    validation = [r for r in load(bench.VALIDATION_SOURCE) if r['partition'] == 'validation']
    test_groups, test_pixels = {r['group'] for r in manifest}, {r['pixel_sha256'] for r in manifest}
    assert len(manifest) == len(test_groups) == 100
    assert not test_groups & {r['group'] for r in training + validation}
    assert not {r['group'] for r in training} & {r['group'] for r in validation}
    seen = {}
    for partition, rows in [('test', manifest), ('train', training), ('validation', validation)]:
        for row in rows:
            pixels = image_digest(read_image(row['path']))
            if partition == 'test':
                assert pixels == row['pixel_sha256'] and digest(row['path']) == row['file_sha256']
            if pixels in seen and seen[pixels] != partition:
                raise RuntimeError(f'Pixel leakage {seen[pixels]} -> {partition}: {row["path"]}')
            seen[pixels] = partition
    shutil.copyfile(PREVIOUS / 'manifest.json', run / 'manifest.json')
    write(run / 'train_manifest.json', training)
    write(run / 'validation_manifest.json', validation)
    write(run / 'split_audit.json', dict(train=len(training), validation=len(validation), test=100,
          identity_overlap=0, pixel_overlap=0, test_manifest_sha256=digest(run / 'manifest.json'),
          original_test_manifest_sha256=digest(PREVIOUS / 'manifest.json'),
          training_layouts=Counter(r['layout'] for r in training),
          scope='existing VNLP train split, not all remaining VNLP images; historical test reused'))
    assets = [upstream / 'MTCNN/weights/pnet_Weights', upstream / 'MTCNN/weights/onet_Weights',
              upstream / 'LPRNet/weights/Final_LPRNet_model.pth', upstream / 'LPRNet/weights/Final_STN_model.pth',
              project / 'models/best.pt', project / 'models/yolo26n-seg.pt']
    assets += sorted((ROOT / 'tmp/easyocr_models').glob('*.pth'))
    protocol = dict(seed=SEED, epochs=EPOCHS, device='cuda:0', gpu=torch.cuda.get_device_name(0),
        train_images=len(training), validation_images=len(validation), test_images=100,
        train_manifest_sha256=digest(run / 'train_manifest.json'), validation_manifest_sha256=digest(run / 'validation_manifest.json'),
        test_manifest_sha256=digest(run / 'manifest.json'), previous_run=str(PREVIOUS),
        checkpoint_selection='final epoch 10 for all networks; no early stopping or best-test choice',
        mtcnn=dict(stages=['PNet', 'ONet'], epochs_per_stage=10, batch=128, optimizer='Adam', lr=.0001,
                   loss='.02 cross entropy on labels>=0 + .6 MSE on labels!=0',
                   patch_quotas_per_scene=dict(positive=12, partial=8, negative=32),
                   patch_sizes=dict(pnet=[47, 12], onet=[94, 24]),
                   onet_hard_mining='PNet epoch10 proposals from train only, max16/image, IoU labels .65/.4/.3',
                   ocr='STN/LPRNet weights and decoder frozen'),
        yolo=dict(initial=str(project / 'models/best.pt'), epochs=10, imgsz=640, batch=16, workers=0,
                  optimizer='AdamW', lr0=.001, lrf=.01, warmup_epochs=1., close_mosaic=0,
                  patience=0, amp=True, augmentations='remaining Ultralytics defaults',
                  ocr='English EasyOCR and vehicle segmentation weights frozen'),
        evaluation=dict(device='cpu', threads=1, repeats=3, inference_settings='unchanged from previous comparison',
                        mtcnn_min_lp=[50, 15], pnet_threshold=.6, onet_threshold=.8,
                        yolo_infer_scale=.5, yolo_plate_imgsz=416, conf_plate=.3, ocr_min_conf=.75),
        initial_assets={str(p.relative_to(ROOT)): digest(p) for p in assets},
        source_hashes={str(p.relative_to(ROOT)): digest(p) for p in
                       [Path(__file__), ROOT / 'src/plate_ml/mtcnn_training.py',
                        ROOT / 'src/plate_ml/comparison.py', ROOT / 'scripts/run_comparison_experiment.py']})
    write(run / 'protocol_before_training.json', protocol)
    write(run / 'source_snapshot.json', dict(package=source_snapshot(ROOT), runner=Path(__file__).read_text(encoding='utf-8'),
          comparison_runner=(ROOT / 'scripts/run_comparison_experiment.py').read_text(encoding='utf-8')))
    (run / 'environment-freeze.txt').write_bytes(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']))
    print(f'RUN: {run}; train {len(training)}, validation {len(validation)}, test 100; leakage 0', flush=True)


def patch_data(scenes, seed):
    rng = np.random.default_rng(seed)
    patches, labels, targets = [], [], []
    for i, scene in enumerate(scenes):
        image = read_image(scene['path'])
        for box, label in sample_patch_boxes(image.shape, scene['box'], rng):
            patch, target = encode_patch(image, box, scene['box'], label)
            patches.append(patch); labels.append(label); targets.append(target)
        if (i+1) % 100 == 0:
            print(f'Patch generation {i+1}/{len(scenes)}', flush=True)
    return np.stack(patches), np.int64(labels), np.stack(targets)


def gpu_data(data, pnet=False):
    patches, labels, targets = data
    if pnet:
        patches = np.stack([cv2.resize(p, (47, 12), interpolation=cv2.INTER_LINEAR) for p in patches])
    return (torch.from_numpy(patches.transpose(0, 3, 1, 2).copy()).to('cuda'),
            torch.from_numpy(labels).to('cuda'), torch.from_numpy(targets).to('cuda'))


def train_stage(model, train, val, run, name):
    model.to('cuda'); model.is_train = True
    optimizer = torch.optim.Adam(model.parameters(), lr=.0001)
    history, start = [], perf_counter()
    generator = torch.Generator(device='cuda').manual_seed(SEED)
    for epoch in range(1, EPOCHS+1):
        record = dict(epoch=epoch)
        for phase, data in [('train', train), ('validation', val)]:
            model.train(phase == 'train')
            order = torch.randperm(len(data[0]), device='cuda', generator=generator) if phase == 'train' else torch.arange(len(data[0]), device='cuda')
            sums, count = np.zeros(3), 0
            for indices in order.split(128):
                image = (data[0][indices].float()-127.5)*.0078125
                optimizer.zero_grad(set_to_none=True)
                with torch.set_grad_enabled(phase == 'train'):
                    offsets, logits = model(image)
                    loss, ce, mse = mtcnn_loss(offsets, logits, data[1][indices], data[2][indices])
                    if not torch.isfinite(loss):
                        raise RuntimeError(f'Nonfinite {name} loss in epoch {epoch}')
                    if phase == 'train':
                        loss.backward(); optimizer.step()
                size = len(indices)
                sums += np.array([loss.item(), ce.item(), mse.item()])*size
                count += size
            record[phase] = dict(loss=float(sums[0]/count), ce=float(sums[1]/count), mse=float(sums[2]/count), samples=count)
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, run / f'{name}_epoch{epoch:02d}.pth')
        history.append(record)
        write(run / f'{name}_history.json', history)
        print(f'{name} epoch {epoch}/10: train {record["train"]["loss"]:.6f}, validation {record["validation"]["loss"]:.6f}', flush=True)
    model.eval(); model.is_train = False
    write(run / f'{name}_training.json', dict(epochs=10, elapsed_seconds=perf_counter()-start,
          final_checkpoint=str(run / f'{name}_epoch10.pth'), sha256=digest(run / f'{name}_epoch10.pth')))


def train_mtcnn(run):
    upstream, _ = bench.prepare_assets()
    adapter = bench.MTCNNLPR(upstream)
    train_scenes, val_scenes = load(run / 'train_manifest.json'), load(run / 'validation_manifest.json')
    training, validation = patch_data(train_scenes, SEED), patch_data(val_scenes, SEED+1)
    write(run / 'patch_audit.json', dict(train_patches=len(training[0]), validation_patches=len(validation[0]),
          train_labels=Counter(map(int, training[1])), validation_labels=Counter(map(int, validation[1]))))
    train_gpu, val_gpu = gpu_data(training, True), gpu_data(validation, True)
    train_stage(adapter.pnet, train_gpu, val_gpu, run, 'pnet')
    del train_gpu, val_gpu; torch.cuda.empty_cache()
    extra_images, extra_labels, extra_targets = [], [], []
    with torch.inference_mode():
        for i, scene in enumerate(train_scenes):
            image = read_image(scene['path'])
            boxes = adapter.mtcnn.detect_pnet(adapter.pnet, image, (50, 15), torch.device('cuda'))
            count = 0
            for box in sorted(boxes, key=lambda b: -float(b[4])):
                ih, iw = image.shape[:2]
                x0, y0, x1, y1 = map(int, box[:4])
                x0, y0, x1, y1 = max(0, x0), max(0, y0), min(iw, x1+1), min(ih, y1+1)
                if x1-x0 < 8 or y1-y0 < 6:
                    continue
                crop_box = (x0, y0, x1-x0, y1-y0)
                label = patch_label(crop_box, scene['box'])
                if label is None:
                    continue
                patch, target = encode_patch(image, crop_box, scene['box'], label)
                extra_images.append(patch); extra_labels.append(label); extra_targets.append(target)
                count += 1
                if count == 16:
                    break
            if (i+1) % 100 == 0:
                print(f'ONet hard mining {i+1}/{len(train_scenes)}; extra patches {len(extra_images)}', flush=True)
    if extra_images:
        training = (np.concatenate([training[0], np.stack(extra_images)]),
                    np.concatenate([training[1], np.int64(extra_labels)]),
                    np.concatenate([training[2], np.stack(extra_targets)]))
    write(run / 'hard_mining_audit.json', dict(source_partition='train', images=len(train_scenes),
          extra_patches=len(extra_images), extra_labels=Counter(extra_labels), onet_train_patches=len(training[0])))
    adapter.pnet.to('cpu')
    train_gpu, val_gpu = gpu_data(training), gpu_data(validation)
    train_stage(adapter.onet, train_gpu, val_gpu, run, 'onet')
    del adapter, train_gpu, val_gpu, training, validation
    gc.collect(); torch.cuda.empty_cache()


def train_yolo(run):
    from ultralytics import YOLO
    _, project = bench.prepare_assets()
    data = run / 'yolo_dataset'
    for partition, name in [('train', 'train_manifest.json'), ('val', 'validation_manifest.json')]:
        for folder in ['images', 'labels']:
            (data / folder / partition).mkdir(parents=True, exist_ok=True)
        for i, row in enumerate(load(run / name)):
            source = Path(row['path']); image = read_image(source); ih, iw = image.shape[:2]
            target = data / 'images' / partition / f'{i:06d}{source.suffix}'
            if not target.exists():
                try:
                    os.link(source, target)
                except OSError:
                    shutil.copyfile(source, target)
            x, y, w, h = row['box']
            (data / 'labels' / partition / f'{i:06d}.txt').write_text(
                f'0 {(x+w/2)/iw:.10f} {(y+h/2)/ih:.10f} {w/iw:.10f} {h/ih:.10f}\n', encoding='ascii')
    write(data / 'data.yaml', dict(path=str(data), train='images/train', val='images/val', names=['License Plate']))
    started = perf_counter()
    model = YOLO(str(project / 'models/best.pt'))
    model.train(data=str(data / 'data.yaml'), epochs=10, imgsz=640, batch=16, workers=0, device=0,
                optimizer='AdamW', lr0=.001, lrf=.01, warmup_epochs=1., close_mosaic=0,
                patience=0, amp=True, seed=SEED, deterministic=True, plots=False,
                save=True, save_period=1, project=str(run), name='yolo_train', exist_ok=False)
    import csv
    with (run / 'yolo_train/results.csv').open() as stream:
        history = list(csv.DictReader(stream))
    assert len(history) == 10, f'Expected 10 completed YOLO epochs, got {len(history)}'
    final = run / 'yolo_train/weights/last.pt'
    write(run / 'yolo_training.json', dict(epochs=10, elapsed_seconds=perf_counter()-started,
          final_checkpoint=str(final), sha256=digest(final), history=history))
    del model
    gc.collect(); torch.cuda.empty_cache()


def evaluate(run):
    from ultralytics import YOLO
    upstream, project = bench.prepare_assets()
    current, mtcnn_before, mtcnn_after = bench.Current(), bench.MTCNNLPR(upstream), bench.MTCNNLPR(upstream)
    mtcnn_after.pnet.load_state_dict(torch.load(run / 'pnet_epoch10.pth', map_location='cpu', weights_only=True))
    mtcnn_after.onet.load_state_dict(torch.load(run / 'onet_epoch10.pth', map_location='cpu', weights_only=True))
    yolo_before, yolo_after = bench.YOLOProject(project), bench.YOLOProject(project)
    yolo_after.model = YOLO(str(run / 'yolo_train/weights/last.pt'))
    arms = dict(current_hog_svm=current, mtcnn_before=mtcnn_before, mtcnn_after=mtcnn_after,
                yolo_before=yolo_before, yolo_after=yolo_after)
    # Training can change global CPU thread settings; restore and verify limits.
    torch.set_num_threads(1); cv2.setNumThreads(1)
    manifest, validation = load(run / 'manifest.json'), load(run / 'validation_manifest.json')
    protocol = load(run / 'protocol_before_training.json')
    assert all(digest(ROOT / p) == h for p, h in protocol['initial_assets'].items())
    evaluation_protocol = dict(training_protocol_sha256=digest(run / 'protocol_before_training.json'),
        manifest_sha256=digest(run / 'manifest.json'), repeats=3, device='cpu', threads=1,
        checkpoints={str(p.relative_to(run)): digest(p) for p in
                     [run / 'pnet_epoch10.pth', run / 'onet_epoch10.pth', run / 'yolo_train/weights/last.pt']},
        frozen_ocr_assets_verified=True, test_used_for_training_or_selection=False)
    write(run / 'evaluation_protocol_before_test.json', evaluation_protocol)
    with threadpool_limits(limits=1):
        rows = bench.evaluate(run, manifest, validation, arms, 3)
    metrics = {name: summarize(records, 3) for name, records in rows.items()}
    changes = {name: paired_difference(rows[f'{name}_before'], rows[f'{name}_after']) for name in ['mtcnn', 'yolo']}
    # Both baseline networks must reproduce the prior predictions on all images.
    for name, old in [('current_hog_svm', 'current_hog_svm'), ('mtcnn_before', 'mtcnn_stn_lprnet'),
                      ('yolo_before', 'yolo26_project_easyocr')]:
        previous = load(PREVIOUS / f'{old}_predictions.json')
        assert [(r['path'], r['text'], r['box']) for r in rows[name]] == [(r['path'], r['text'], r['box']) for r in previous]
    summary = dict(run_path=str(run), metrics=metrics, paired_after_minus_before=changes,
        yolo_ocr_images={name: sum(any('snippets' in p for p in r['plate_ocr']) for r in rows[name])
                         for name in ['yolo_before', 'yolo_after']},
        epochs=dict(pnet=len(load(run / 'pnet_history.json')), onet=len(load(run / 'onet_history.json')),
                    yolo=load(run / 'yolo_training.json')['epochs']),
        previous_predictions_reproduced=True, frozen_ocr_verified=True,
        evaluation_protocol_sha256=digest(run / 'evaluation_protocol_before_test.json'))
    write(run / 'summary.json', summary)
    print(json.dumps({k: dict(exact=v['exact_count'], cer=v['decoded']['character_error_rate'],
          recall=v['detector_selected']['recall'], mean_ms=v['latency']['mean_ms']) for k, v in metrics.items()}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['all', 'prepare', 'mtcnn', 'yolo', 'evaluate'], default='all')
    parser.add_argument('--run', type=Path)
    args = parser.parse_args()
    run = args.run.resolve() if args.run else ROOT / 'reports' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-finetune-10')
    if not torch.cuda.is_available() and args.phase != 'evaluate':
        raise RuntimeError('CUDA training requires an available GPU')
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.set_num_threads(1); torch.set_num_interop_threads(1); cv2.setNumThreads(1)
    if args.phase in ['prepare', 'all']:
        prepare(run)
    else:
        protocol = load(run / 'protocol_before_training.json')
        assert all(digest(ROOT / p) == h for p, h in protocol['source_hashes'].items())
    if args.phase in ['mtcnn', 'all']:
        train_mtcnn(run)
    if args.phase in ['yolo', 'all']:
        train_yolo(run)
    if args.phase in ['evaluate', 'all']:
        evaluate(run)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
