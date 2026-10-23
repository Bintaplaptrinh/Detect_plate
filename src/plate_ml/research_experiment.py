"""Reproducible experiment orchestration; visualization belongs in notebook 03."""
from dataclasses import asdict, replace
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np

from .domain_ocr import assert_group_disjoint, remove_scene_duplicates
from .evaluation import detection_metrics
from .imaging import intersection_over_union, read_image
from .pipeline import PlateRecognizer
from .registry import code_info, environment_info, source_snapshot
from .research_pipeline import (RegionConfig, ResearchRecognizer, RowProjectionSegmenter, HybridSegmenter,
                                 TwoStageDetector, train_region_verifier, sequence_report)
from .vnlp import Scene, crop_scene


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def fresh_partitions(root, previous_manifest, seed=20261022, val_per_layout=30, test_per_layout=60):
    """New identity holdouts exclude *all* scenes exposed in experiment 02."""
    root = Path(root).resolve()
    known = {r['group'] for r in previous_manifest}
    train = [Scene(str(root/r['layout']/Path(r['path']).name), r['text'], tuple(r['box']), r['layout'], r['group'])
             for r in previous_manifest if r['partition'] == 'train']
    partitions = dict(train=train, validation=[], test=[])
    used = set(known)
    rng = np.random.default_rng(seed)
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        paths = sorted(folder.glob('*.jpg'))
        rng.shuffle(paths)
        needed = dict(validation=val_per_layout, test=test_per_layout)
        for path in paths:
            parts = path.stem.split('_')
            if len(parts) != 8:
                raise ValueError(f'Unexpected VNLP filename: {path.name}')
            text = parts[3].upper()
            if text in used:
                continue
            partition = next((name for name, remaining in needed.items() if remaining), None)
            if partition is None:
                break
            x0, y0, x1, y1 = map(int, parts[-4:])
            image = read_image(path)
            if not (0 <= x0 < x1 <= image.shape[1] and 0 <= y0 < y1 <= image.shape[0]):
                raise ValueError(f'Invalid bbox: {path.name}')
            partitions[partition].append(Scene(str(path), text, (x0,y0,x1-x0,y1-y0), folder.name, text))
            needed[partition] -= 1
            used.add(text)
        if any(needed.values()):
            raise ValueError(f'Insufficient unseen identities for {folder.name}: {needed}')
    assert_group_disjoint(partitions)
    clean, duplicates = remove_scene_duplicates(partitions)
    manifest = [dict(**asdict(s), partition=name) for name, scenes in clean.items() for s in scenes]
    return clean, manifest, dict(seed=seed, excluded_previous_identities=len(known),
                                duplicate_scenes=duplicates,
                                split_counts={k: len(v) for k,v in clean.items()},
                                status='new_plate_identity_holdout_same_dataset_not_camera_holdout')


def compact_recognition(result):
    return dict(raw=result['decoded']['raw'], text=result['decoded']['text'],
                quality=result['quality'], chars=len(result['segmentation'].characters),
                segmentation=result['segmentation'].diagnostics)


def build_cache(scenes, baseline, verifier, progress=None):
    """Cache model-independent boxes and OCR hypotheses; no truth-guided choice."""
    region = TwoStageDetector(verifier, RegionConfig(threshold=-1.0, max_candidates=12))
    old = ResearchRecognizer(baseline.model, baseline.detector, baseline.segmenter,
                            baseline.extractor.config, baseline.templates)
    new = ResearchRecognizer(baseline.model, region, RowProjectionSegmenter(),
                            baseline.extractor.config, baseline.templates)
    hybrid = ResearchRecognizer(baseline.model, region, HybridSegmenter(baseline.segmenter.config),
                               baseline.extractor.config, baseline.templates)
    cached = []
    for index, scene in enumerate(scenes):
        start = perf_counter()
        image = read_image(scene.path)
        old_detection = baseline.detector.detect(image)
        new_detection = region.detect(image)
        item = dict(path=scene.path, truth=scene.text, layout=scene.layout, gt_box=list(scene.box),
                    proposals=new_detection.diagnostics['raw_proposals'], old=[], new=[],
                    old_all_boxes=[list(c.box) for c in old_detection.candidates])
        for key, detected in [('old', old_detection), ('new', new_detection)]:
            # Only top-1 of old detector is needed for baseline/segmentation ablation.
            for candidate in detected.candidates[:1 if key == 'old' else 12]:
                entry = dict(box=list(candidate.box), margin=candidate.score, source=candidate.source)
                entry['old_seg'] = compact_recognition(old.recognize_crop(candidate.crop))
                entry['new_seg'] = compact_recognition(new.recognize_crop(candidate.crop))
                entry['hybrid_seg'] = compact_recognition(hybrid.recognize_crop(candidate.crop))
                item[key].append(entry)
        crop = crop_scene(scene)
        item['oracle_old'] = compact_recognition(old.recognize_crop(crop))
        item['oracle_new'] = compact_recognition(new.recognize_crop(crop))
        item['oracle_hybrid'] = compact_recognition(hybrid.recognize_crop(crop))
        item['cache_seconds'] = perf_counter()-start
        cached.append(item)
        if progress and ((index+1) % 10 == 0 or index+1 == len(scenes)):
            progress(f'evaluation cache {index+1}/{len(scenes)}')
    return cached


def score_cache(cached, detector='new', segmenter='new_seg', threshold=-0.5, rerank=0.5, top_k=5):
    rows, proposed, top_boxes, all_boxes = [], [], [], []
    for item in cached:
        candidates = item[detector]
        if detector == 'new':
            candidates = [c for c in candidates if c['margin'] >= threshold]
        eligible = candidates[:top_k]
        best = max(eligible, key=lambda c: c['margin']+rerank*c[segmenter]['quality'], default=None)
        predicted = best[segmenter] if best else dict(raw='', text='', chars=0)
        oracle = item[{'new_seg':'oracle_new', 'old_seg':'oracle_old', 'hybrid_seg':'oracle_hybrid'}[segmenter]]
        rows.append(dict(path=item['path'], layout=item['layout'], truth=item['truth'],
                         oracle_raw=oracle['raw'], oracle_decoded=oracle['text'],
                         pipeline_raw=predicted['raw'], pipeline_decoded=predicted['text'],
                         characters=predicted['chars'], candidates=len(candidates),
                         top_box=best['box'] if best else None,
                         top_iou=intersection_over_union(best['box'], item['gt_box']) if best else 0.0))
        proposed.append([item['gt_box']])
        top_boxes.append([best['box']] if best else [])
        all_boxes.append(item['old_all_boxes'] if detector == 'old' else [c['box'] for c in candidates])
    metrics = sequence_report(rows)
    metrics['detector_top1'] = detection_metrics(top_boxes, proposed)
    metrics['detector_all'] = detection_metrics(all_boxes, proposed)
    metrics['retained_candidate_recall_at_05'] = float(np.mean([
        any(intersection_over_union(box, item['gt_box']) >= .5 for box in boxes)
        for item, boxes in zip(cached, all_boxes)]))
    metrics['count_match'] = float(np.mean([r['characters'] == len(r['truth']) for r in rows]))
    return metrics, rows


def select_pipeline(validation_cache):
    variants = {
        'baseline_02': dict(detector='old', segmenter='old_seg', top_k=1, rerank=0),
        'new_detector_only': dict(detector='new', segmenter='old_seg', threshold=-.5, top_k=1, rerank=0),
        'new_segmentation_only': dict(detector='old', segmenter='new_seg', top_k=1, rerank=0),
        'two_stage_top1': dict(detector='new', segmenter='new_seg', threshold=-.5, top_k=1, rerank=0),
        'hybrid_top1': dict(detector='new', segmenter='hybrid_seg', threshold=-.5, top_k=1, rerank=0),
    }
    # This finite search is declared before opening the new test set.
    for segmenter in ('new_seg', 'hybrid_seg'):
        for threshold in (-1.0, -.5, 0.0):
            for rerank in (0.0, .5, 1.0, 2.0):
                variants[f'{segmenter}_t{threshold}_w{rerank}'] = dict(detector='new', segmenter=segmenter,
                                                                  threshold=threshold, rerank=rerank, top_k=5)
    trials = []
    for name, config in variants.items():
        metrics, _ = score_cache(validation_cache, **config)
        trials.append(dict(name=name, config=config, metrics=metrics))
    # Prefer baseline on exact ties; select before test, using end-to-end raw text.
    best = min(trials, key=lambda t: (t['metrics']['pipeline_raw']['character_error_rate'],
                                     -t['metrics']['pipeline_raw']['exact_match']))
    return dict(champion=best['name'], config=best['config'],
                criterion='validation end-to-end raw CER, then exact match; earlier arm wins ties', trials=trials)


def recognizer_for(selection_config, baseline, verifier):
    config = selection_config
    detector = (baseline.detector if config['detector'] == 'old' else
                TwoStageDetector(verifier, replace(RegionConfig(), threshold=config['threshold'])))
    segmenter = {'new_seg': RowProjectionSegmenter(), 'old_seg': baseline.segmenter,
                 'hybrid_seg': HybridSegmenter(baseline.segmenter.config)}[config['segmenter']]
    return ResearchRecognizer(baseline.model, detector, segmenter, baseline.extractor.config,
                              baseline.templates, config['rerank'], config['top_k'])


def benchmark_and_verify(recognizer, scenes, expected_rows, limit=30):
    timings = []
    for scene, expected in list(zip(scenes, expected_rows))[:limit]:
        image = read_image(scene.path)
        start = perf_counter()
        _, results = recognizer.recognize(image)
        timings.append(perf_counter()-start)
        text = results[0]['decoded']['text'] if results else ''
        assert text == expected['pipeline_decoded'], f'Cached/real inference mismatch: {scene.path}'
    return dict(samples=len(timings), mean_seconds=float(np.mean(timings)),
                median_seconds=float(np.median(timings)), p95_seconds=float(np.percentile(timings,95)),
                includes='full detector + segmentation + OCR, excludes image decode; CPU, one OpenCV thread')
