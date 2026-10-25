"""Frozen, no-training comparison on 100 samples of the historical VNLP test C.

External sources are pinned; generated assets stay under reports/ and tmp/.
Run from the repository root with the same trusted local registry as run.py.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import cv2
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from plate_ml.comparison import paired_difference, select_test_subset, summarize, normalize_ocr, join_ocr_lines
from plate_ml.imaging import read_image
from plate_ml.inference import load_recognizer
from plate_ml.registry import environment_info, source_snapshot

UPSTREAM = 'aa5778d6e6da0f382894048b4199cca7a1151e95'
YOLO_PROJECT = '25f9099a8b1af72c8d481e8bf1ec62e586b98887'
TEST_SOURCE = ROOT / 'reports/20261022T104205345901Z-glyph-filter/fresh_test_manifest.json'
TRAIN_SOURCE = ROOT / 'Registration/Registry/pipeline_vnlp_experiment/v1/scene_manifest.json'
VALIDATION_SOURCE = ROOT / 'reports/20261022T091902580679Z-research/scene_manifest.json'


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def prepare_assets():
    upstream = ROOT / 'tmp/License_Plate_Detection_Pytorch'
    if not upstream.exists():
        subprocess.run(['git', 'clone', 'https://github.com/xuexingyu24/License_Plate_Detection_Pytorch.git', str(upstream)], check=True)
    actual = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != UPSTREAM:
        raise RuntimeError(f'Upstream revision mismatch: {actual}; expected {UPSTREAM}')
    project = ROOT / 'tmp/Vehicle-License-Plate-Detection-with-YOLO26'
    if not project.exists():
        subprocess.run(['git', 'clone', 'https://github.com/Esteebaan23/Vehicle-License-Plate-Detection-with-YOLO26.git', str(project)], check=True)
    actual = subprocess.check_output(['git', '-C', str(project), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != YOLO_PROJECT:
        raise RuntimeError(f'YOLO project revision mismatch: {actual}; expected {YOLO_PROJECT}')
    return upstream, project


class Current:
    def __init__(self):
        self.model = load_recognizer(ROOT / 'Registration', 'v1')

    def predict(self, image):
        detected, results = self.model.recognize(image)
        best = results[0] if results else None
        return dict(text=best['decoded']['text'] if best else '', raw=best['decoded']['raw'] if best else '',
                    box=list(best['candidate'].box) if best else None,
                    detections=[dict(box=list(c.box), score=float(c.score)) for c in detected.candidates])


class MTCNNLPR:
    def __init__(self, root):
        # Preserve upstream detection and greedy decoding, including its first-
        # timestep suppression. Only bypass GUI and per-image checkpoint loads.
        sys.path.insert(0, str(root / 'MTCNN'))
        self.mtcnn = module(root / 'MTCNN/MTCNN.py', 'benchmark_mtcnn')
        lpr = module(root / 'LPRNet/model/LPRNET.py', 'benchmark_lprnet')
        stn = module(root / 'LPRNet/model/STN.py', 'benchmark_stn')
        self.chars = lpr.CHARS
        self.pnet, self.onet = self.mtcnn.PNet(), self.mtcnn.ONet()
        self.lpr, self.stn = lpr.LPRNet(len(lpr.CHARS), 0), stn.STNet()
        for model, path in [(self.pnet, root / 'MTCNN/weights/pnet_Weights'),
                            (self.onet, root / 'MTCNN/weights/onet_Weights'),
                            (self.lpr, root / 'LPRNet/weights/Final_LPRNet_model.pth'),
                            (self.stn, root / 'LPRNet/weights/Final_STN_model.pth')]:
            model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
            model.eval()

    def ocr(self, crop):
        im = cv2.resize(crop, (94, 24), interpolation=cv2.INTER_CUBIC)
        im = (np.transpose(np.float32(im), (2, 0, 1)) - 127.5) * .0078125
        logits = self.lpr(self.stn(torch.from_numpy(im).unsqueeze(0))).numpy()[0]
        indices = logits.argmax(axis=0)
        previous, labels = indices[0], []
        for char in indices:
            if previous == char or char == len(self.chars) - 1:
                if char == len(self.chars) - 1:
                    previous = char
                continue
            labels.append(self.chars[char])
            previous = char
        return ''.join(labels)

    def predict(self, image):
        boxes = self.mtcnn.detect_pnet(self.pnet, image, (50, 15), torch.device('cpu'))
        boxes = self.mtcnn.detect_onet(self.onet, image, boxes, torch.device('cpu'))
        boxes = sorted(boxes, key=lambda b: -float(b[4]))
        detections, options = [], []
        ih, iw = image.shape[:2]
        for b in boxes:
            x0, y0, x1, y1 = [int(v) for v in b[:4]]
            # The upstream endpoint coordinates and crop slices are inclusive.
            box = [x0, y0, x1 - x0 + 1, y1 - y0 + 1]
            detections.append(dict(box=box, score=float(b[4])))
            crop = image[max(0, y0):min(ih, y1 + 1), max(0, x0):min(iw, x1 + 1)]
            text = self.ocr(crop) if crop.size else ''
            options.append((text, box))
        text, box = options[0] if options else ('', None)
        return dict(text=text, raw=text, box=box, detections=detections)


class ProjectOCR:
    def __init__(self, root):
        import easyocr
        sys.path.insert(0, str(root))
        text = module(root / 'process/ocr_extraction/text_extraction.py', 'benchmark_project_text')
        process = module(root / 'process/ocr_extraction/ocr.py', 'benchmark_project_ocr')
        # Same upstream classes, bypass their GPU-default constructors only.
        self.engine = text.TextExtraction.__new__(text.TextExtraction)
        self.engine.ocr = process.OcrProcess.__new__(process.OcrProcess)
        self.engine.ocr.reader = easyocr.Reader(['en'], gpu=False, verbose=False,
                          model_storage_directory=str(ROOT / 'tmp/easyocr_models'))
        self.engine.min_vertical_distance = 12

    def ocr(self, crop):
        processed = self.engine.image_contrast(crop)
        detected = self.engine.ocr.text_detection(processed)
        try:
            original = self.engine.process_text_line(detected)
            original_error = None
        except TypeError as exc:
            # Upstream main catches this exception and returns an empty string.
            original, original_error = '', f'{type(exc).__name__}: {exc}'
        raw = join_ocr_lines(detected, self.engine.min_vertical_distance)
        return dict(raw=raw, text=normalize_ocr(raw), original_raw=original,
                    original_text=normalize_ocr(original), original_error=original_error,
                    snippets=[dict(box=box, text=text, confidence=conf) for box, text, conf in detected])


class YOLOProject:
    def __init__(self, root):
        from ultralytics import YOLO
        sys.path.insert(0, str(root))
        vehicle = module(root / 'process/computer_vision_models/vehicle_detection.py', 'benchmark_project_vehicle')
        self.geometry = module(root / 'process/utils/geometry.py', 'benchmark_project_geometry')
        self.car = vehicle.VehicleDetection(str(root / 'models/yolo26n-seg.pt'), device='cpu')
        self.model = YOLO(str(root / 'models/best.pt'))
        self.ocr = ProjectOCR(root)
        self.ocr_calls = 0

    def predict(self, image):
        # Exact video defaults, adapting only frame-skip to one image/one frame.
        ih, iw = image.shape[:2]
        sw, sh = max(16, int(iw * .5)), max(16, int(ih * .5))
        inf = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_LINEAR)
        cars, _ = self.car.predict_cars(inf, conf=.15, imgsz=640)
        result = self.model.predict(inf, device='cpu', imgsz=416, conf=.30, verbose=False)[0]
        detections, options = [], []
        for b in sorted(result.boxes, key=lambda b: -float(b.conf.item())):
            x0, y0, x1, y1 = [int(v) for v in b.xyxy[0].tolist()]
            # Upstream yolo_result_to_boxes truncates before scaling.
            x0, y0, x1, y1 = int(x0*iw/sw), int(y0*ih/sh), int(x1*iw/sw), int(y1*ih/sh)
            score, box = float(b.conf.item()), [x0, y0, x1-x0, y1-y0]
            detections.append(dict(box=box, score=score))
            prediction = dict(raw='', text='', original_raw='', original_text='', original_error=None)
            if score >= .75:
                cx0, cy0, cx1, cy1 = self.geometry.clamp_box(x0, y0, x1, y1, iw, ih)
                crop = image[cy0:cy1, cx0:cx1]
                if crop.size and crop.shape[0] >= 12 and crop.shape[1] >= 30:
                    prediction = self.ocr.ocr(crop)
                    self.ocr_calls += 1
            options.append(prediction)
        best = options[0] if options else dict(raw='', text='', original_raw='', original_text='', original_error=None)
        return dict(raw=best['raw'], text=best['text'], box=detections[0]['box'] if detections else None,
                    detections=detections, original_raw=best['original_raw'], original_text=best['original_text'],
                    original_error=best['original_error'], plate_ocr=options, vehicle_count=len(cars))


def image_digest(image):
    return sha256(str(image.shape).encode() + image.tobytes()).hexdigest()


def prepare_run(args):
    run = args.run.resolve() if args.run else ROOT / 'reports' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-comparison-100')
    run.mkdir(parents=True, exist_ok=False)
    manifest = select_test_subset(json.loads(TEST_SOURCE.read_text(encoding='utf-8')), 100, args.seed)
    training = [r for r in json.loads(TRAIN_SOURCE.read_text(encoding='utf-8')) if r['partition'] == 'train']
    validation = [r for r in json.loads(VALIDATION_SOURCE.read_text(encoding='utf-8')) if r['partition'] == 'validation']
    selected_groups = {r['group'] for r in manifest}
    if selected_groups & {r['group'] for r in training + validation}:
        raise RuntimeError('Plate identity leakage into current training/validation')
    for r in manifest:
        image = read_image(r['path'])
        r.update(file_sha256=digest(r['path']), pixel_sha256=image_digest(image))
    test_pixels = {r['pixel_sha256'] for r in manifest}
    duplicates = [r['path'] for r in training + validation if image_digest(read_image(r['path'])) in test_pixels]
    if duplicates:
        raise RuntimeError(f'Pixel leakage: {duplicates}')
    write(run / 'manifest.json', manifest)
    write(run / 'split_audit.json', dict(train_images=len(training), validation_images=len(validation),
          test_images=100, identity_overlap=0, pixel_overlap=0, unique_test_identities=len(selected_groups),
          layout_counts=Counter(r['layout'] for r in manifest), source_manifest=str(TEST_SOURCE),
          source_sha256=digest(TEST_SOURCE), train_manifest_sha256=digest(TRAIN_SOURCE),
          validation_manifest_sha256=digest(VALIDATION_SOURCE),
          status='subset_of_previously_evaluated_test_C_not_new_independent_holdout',
          current_legacy_character_training_lineage='unknown; no claim of source-independent OCR evaluation',
          external_checkpoint_training_overlap='unverified; source metadata retained in protocol'))
    return run, manifest, validation


def evaluate(run, manifest, warmups, arms, repeats):
    rows = {name: [dict(path=r['path'], truth=r['text'], layout=r['layout'], gt_box=r['box'],
                         latency_ms=[]) for r in manifest] for name in arms}
    with torch.inference_mode():
        for name, arm in arms.items():
            for r in warmups[:3]:
                arm.predict(read_image(r['path']))
                # Ultralytics select_device resets torch threads on first use.
                # Restore the experimental limit after backend initialization.
                torch.set_num_threads(1)
                cv2.setNumThreads(1)
            print(f'Warmed {name}', flush=True)
        write(run / 'runtime_threads_after_warmup.json', dict(torch=torch.get_num_threads(),
              torch_interop=torch.get_num_interop_threads(), opencv=cv2.getNumThreads()))
        if torch.get_num_threads() != 1 or cv2.getNumThreads() != 1:
            raise RuntimeError('Runtime thread limit changed during warmup')
        names = list(arms)
        # Rotate the arm order between passes, so the same arm is not always
        # timed first. All images are decoded outside the inference timer.
        for repeat in range(repeats):
            order = names[repeat % len(names):] + names[:repeat % len(names)]
            for i, scene in enumerate(manifest):
                image = read_image(scene['path'])
                for name in order:
                    start = perf_counter()
                    result = arms[name].predict(image)
                    elapsed = (perf_counter() - start) * 1000
                    if torch.get_num_threads() != 1 or cv2.getNumThreads() != 1:
                        raise RuntimeError('Runtime thread limit changed during measured inference')
                    row = rows[name][i]
                    if repeat:
                        if row['text'] != result['text'] or row['raw'] != result['raw'] or row['box'] != result['box']:
                            raise RuntimeError(f'Non-deterministic prediction: {name} {scene["path"]}')
                    row.update(result)
                    row['latency_ms'].append(elapsed)
                if (i+1) % 10 == 0:
                    print(f'Pass {repeat+1}/{repeats}: {i+1}/100', flush=True)
            for name in names:
                write(run / f'{name}_predictions.json', rows[name])
    return rows


def oracle_diagnostics(run, manifest, current, mtcnn, ocr):
    records = {name: [] for name in ['current_gt_crop', 'lprnet_gt_crop', 'project_easyocr_gt_crop']}
    with torch.inference_mode():
        for i, scene in enumerate(manifest):
            image = read_image(scene['path'])
            x, y, w, h = scene['box']
            crop = image[y:y+h, x:x+w]
            for name, call in [('current_gt_crop', current.model.recognize_crop),
                               ('lprnet_gt_crop', mtcnn.ocr), ('project_easyocr_gt_crop', ocr.ocr)]:
                start = perf_counter()
                prediction = call(crop)
                elapsed = (perf_counter() - start) * 1000
                if name == 'current_gt_crop':
                    prediction = prediction['decoded']
                elif name == 'lprnet_gt_crop':
                    prediction = dict(text=prediction, raw=prediction)
                records[name].append(dict(path=scene['path'], truth=scene['text'], layout=scene['layout'],
                    raw=prediction['raw'], text=prediction['text'], gt_box=scene['box'], box=scene['box'],
                    detections=[dict(box=scene['box'], score=1.)], latency_ms=[elapsed]))
            if (i+1) % 20 == 0:
                print(f'Oracle OCR diagnosis {i+1}/100', flush=True)
    write(run / 'oracle_predictions.json', records)
    return {name: summarize(value, 1) for name, value in records.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=20261025)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--run', type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    upstream, project = prepare_assets()
    run, manifest, validation = prepare_run(args)
    print(f'RUN: {run}', flush=True)
    with threadpool_limits(limits=1):
        current, mtcnn = Current(), MTCNNLPR(upstream)
        yolo = YOLOProject(project)
        arms = dict(current_hog_svm=current, mtcnn_stn_lprnet=mtcnn, yolo26_project_easyocr=yolo)
        assets = [upstream / p for p in ['MTCNN/weights/pnet_Weights', 'MTCNN/weights/onet_Weights',
                  'LPRNet/weights/Final_LPRNet_model.pth', 'LPRNet/weights/Final_STN_model.pth']]
        assets += [project / 'models/best.pt', project / 'models/yolo26n-seg.pt',
                   ROOT / 'Registration/Registry/research_glyph_filter/v1/model.joblib']
        assets += sorted((ROOT / 'tmp/easyocr_models').glob('*.pth'))
        protocol = dict(seed=args.seed, samples=100, repeats=args.repeats, device='cpu',
            precision='YOLO and LPRNet float32; EasyOCR CPU default quantize=True',
            threads=1, batch=1, warmup_images=[r['path'] for r in validation[:3]],
            timing='decode and checkpoint loads excluded; full in-memory pipeline; median of passes per image',
            selected_plate='current release OCR reranking; other detectors highest confidence; no GT selection',
            test_manifest_sha256=digest(run / 'manifest.json'), trained=False, tuned_on_test=False,
            upstream_revision=UPSTREAM, yolo_project_revision=YOLO_PROJECT,
            yolo=dict(classes=yolo.model.names, car_imgsz=640, plate_imgsz=416, conf_car=.15, conf_plate=.30,
                      infer_scale=.5, frame_skip=1, roi=False, plates_in_cars_only=False,
                      ocr_min_conf=.75, ocr_languages=['en'], ocr_gpu=False,
                      training_provenance=yolo.model.ckpt.get('train_args'),
                      changes='static-image frame skip 1; CPU OCR; bypass GUI/logging/disk debug; bbox center-y fix for same_line',
                      original_error_audit='save original_text and original_error for each crop; upstream main catches error and emits empty text',
                      text_normalization='uppercase and strip non-alphanumeric separators; no character substitution'),
            mtcnn=dict(scale=1, minimum_plate_size=[50, 15], pnet_threshold=.6, onet_threshold=.8,
                       resize_lpr=[94, 24], preprocessing='upstream RGB detector / BGR LPRNet',
                       decoder='upstream greedy; first timestep suppression preserved',
                       changes='preload/eval networks once; GUI removed; inclusive boxes converted to xywh'),
            assets={str(p.relative_to(ROOT)): dict(sha256=digest(p), bytes=p.stat().st_size) for p in assets},
            code_sha256={str(p.relative_to(ROOT)): digest(p) for p in
                         [Path(__file__), ROOT / 'src/plate_ml/comparison.py']},
            environment=dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                             processor=platform.processor(), torch=torch.__version__, cv2=cv2.__version__,
                             numpy=np.__version__, cuda_available=torch.cuda.is_available(),
                             cpu=platform.uname().processor, packages=environment_info()['packages']))
        write(run / 'source_snapshot.json', dict(package=source_snapshot(ROOT),
              runner=Path(__file__).read_text(encoding='utf-8')))
        (run / 'environment-freeze.txt').write_bytes(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']))
        # Lock configuration before exposing any test images to any recognizer.
        write(run / 'protocol_before_test.json', protocol)
        protocol_hash = digest(run / 'protocol_before_test.json')
        rows = evaluate(run, manifest, validation, arms, args.repeats)
        metrics = {name: summarize(value, args.repeats) for name, value in rows.items()}
        comparisons = {name: paired_difference(rows['current_hog_svm'], value)
                       for name, value in rows.items() if name != 'current_hog_svm'}
        original = [dict(r, raw=r['original_raw'], text=r['original_text']) for r in rows['yolo26_project_easyocr']]
        oracle = oracle_diagnostics(run, manifest, current, mtcnn, yolo.ocr)
        assert digest(run / 'protocol_before_test.json') == protocol_hash
        summary = dict(run_path=str(run), protocol_sha256=protocol_hash, metrics=metrics,
                       paired_candidate_minus_current=comparisons, oracle_ocr_diagnostic=oracle,
                       yolo_ocr_calls_including_warmup=yolo.ocr_calls,
                       yolo_original_code_metrics=summarize(original, args.repeats),
                       yolo_original_ocr_error_images=sum(bool(r['original_error']) for r in original),
                       yolo_status='project_checkpoint_and_easyocr_with_documented_bbox_fix')
        write(run / 'summary.json', summary)
        print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    return run


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
