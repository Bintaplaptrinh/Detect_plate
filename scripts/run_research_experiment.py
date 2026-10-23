"""Run the paper-inspired pipeline in auditable train/validation/test phases."""
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import argparse
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import cv2
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from plate_ml.pipeline import PlateRecognizer
from plate_ml.registry import ModelRegistry, source_snapshot, code_info, environment_info
from plate_ml.research_pipeline import train_region_verifier, RowProjectionSegmenter, sequence_report
from plate_ml.imaging import read_image
from plate_ml.research_experiment import (fresh_partitions, build_cache, score_cache, select_pipeline,
    recognizer_for, benchmark_and_verify, write_json)
from plate_ml.vnlp import Scene
from plate_ml.vnlp import crop_scene


def load_context(run):
    registry = ModelRegistry(ROOT / 'Registration')
    bundle, receipt = registry.load('pipeline_vnlp_experiment', 'v1', trusted=True)
    baseline = PlateRecognizer.from_bundle(bundle, registry.root / 'RuntimeAssets')
    config = json.loads((run / 'experiment.json').read_text(encoding='utf-8'))
    manifest = json.loads((run / 'scene_manifest.json').read_text(encoding='utf-8'))
    partitions = {name: [Scene(r['path'], r['text'], tuple(r['box']), r['layout'], r['group'])
                         for r in manifest if r['partition'] == name]
                  for name in ('train', 'validation', 'test')}
    return registry, baseline, receipt, config, partitions


def load_research_release(registry, version):
    """Persist Python models/configs/bytes, never native OpenCV descriptors."""
    saved, receipt = registry.load('research_pipeline', version, trusted=True)
    baseline = PlateRecognizer.from_bundle(saved['baseline_bundle'], registry.root / 'RuntimeAssets')
    return recognizer_for(saved['selection_config'], baseline, saved['verifier']), receipt


def prepare_and_train(progress=print):
    cv2.setNumThreads(1)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-research'
    run = ROOT / 'reports' / run_id
    run.mkdir(parents=True, exist_ok=False)
    registry = ModelRegistry(ROOT / 'Registration')
    _, receipt = registry.load('pipeline_vnlp_experiment', 'v1', trusted=True)
    previous = json.loads((Path(registry.root) / 'Registry/pipeline_vnlp_experiment/v1/scene_manifest.json').read_text(encoding='utf-8'))
    vnlp_root = Path(os.environ.get('PLATE_VNLP_ROOT', str(ROOT.parent / 'data/VNLP_detection/detection')))
    partitions, manifest, split_audit = fresh_partitions(vnlp_root, previous)
    refs = []
    for filename in ['Two-stage_HOGSVM_for_license_plate_detection_and_r.pdf', 'LICENSE_PLATE_RECOGNITION_Using_MACHINE.pdf']:
        path = Path('C:/Users/Admin/Downloads') / filename
        refs.append(dict(filename=filename, sha256=sha256(path.read_bytes()).hexdigest() if path.exists() else None))
    config = dict(run_id=run_id, baseline='pipeline_vnlp_experiment/v1', split=split_audit,
                  papers=refs, seed=42, source=code_info(ROOT), environment=environment_info(),
                  ocr='frozen experiment 02 legacy_plus_vnlp; notebook 01 glyph HOG unchanged',
                  baseline_artifacts=receipt['artifact_sha256'],
                  selection_budget=dict(thresholds=[-1.0,-.5,0.0], rerank_weights=[0,.5,1,2], top_k=5),
                  test_status='sealed_before_validation_selection')
    write_json(run / 'experiment.json', config)
    write_json(run / 'scene_manifest.json', manifest)
    write_json(run / 'source_snapshot.json', source_snapshot(ROOT))
    progress('RUN: ' + str(run))
    progress('Split: ' + str(split_audit['split_counts']))
    start = perf_counter()
    verifier, training = train_region_verifier(partitions['train'], progress=progress)
    joblib.dump(verifier, run / 'region_verifier.joblib', compress=3)
    training['elapsed_seconds'] = perf_counter()-start
    write_json(run / 'region_training.json', training)
    progress('Region verifier trained: ' + str({k:v for k,v in training.items() if k != 'scenes'}))
    return run


def validate(run, progress=print, reuse_cache=False):
    cv2.setNumThreads(1)
    _, baseline, _, config, partitions = load_context(run)
    verifier = joblib.load(run / 'region_verifier.joblib')
    cache = (joblib.load(run / 'validation_cache.joblib') if reuse_cache else
             build_cache(partitions['validation'], baseline, verifier, progress))
    # Upgrade older validation caches without recomputing OCR hypotheses.
    for item in cache:
        if 'old_all_boxes' not in item:
            item['old_all_boxes'] = [list(c.box) for c in baseline.detector.detect(read_image(item['path'])).candidates]
    joblib.dump(cache, run / 'validation_cache.joblib', compress=3)
    selection = select_pipeline(cache)
    selection['source'] = code_info(ROOT)
    write_json(run / 'validation_source_snapshot.json', source_snapshot(ROOT))
    write_json(run / 'selection_before_test.json', selection)
    for trial in sorted(selection['trials'], key=lambda t: t['metrics']['pipeline_raw']['character_error_rate']):
        m = trial['metrics']
        progress(str(dict(arm=trial['name'], raw_cer=m['pipeline_raw']['character_error_rate'],
                         exact=m['pipeline_decoded']['exact_match'],
                         top1_recall=m['detector_top1']['recall'],
                         retained_candidate_recall=m['retained_candidate_recall_at_05'],
                         oracle_cer=m['oracle_decoded']['character_error_rate'])))
    progress('LOCKED CHAMPION: ' + selection['champion'])
    return selection


def test_and_register(run, progress=print, reuse_cache=False):
    cv2.setNumThreads(1)
    registry, baseline, baseline_receipt, config, partitions = load_context(run)
    selection_path = run / 'selection_before_test.json'
    selection_hash = sha256(selection_path.read_bytes()).hexdigest()
    selection = json.loads(selection_path.read_text(encoding='utf-8'))
    # Refuse source drift between validation selection and test. Training and
    # validation snapshots remain separate when development fixes precede test.
    assert selection['source']['source_sha256'] == code_info(ROOT)['source_sha256'], 'Source changed since validation; rerun validation before test'
    verifier = joblib.load(run / 'region_verifier.joblib')
    cache = (joblib.load(run / 'test_cache.joblib') if reuse_cache else
             build_cache(partitions['test'], baseline, verifier, progress))
    joblib.dump(cache, run / 'test_cache.joblib', compress=3)
    comparisons = {t['name']: t['config'] for t in selection['trials'][:5]}
    comparisons['selected_pipeline'] = selection['config']
    metrics, predictions = {}, {}
    for name, arm_config in comparisons.items():
        metrics[name], predictions[name] = score_cache(cache, **arm_config)
    assert sha256(selection_path.read_bytes()).hexdigest() == selection_hash
    candidate = recognizer_for(selection['config'], baseline, verifier)
    # The manifest is ordered by layout: taking its first 30 would time only cars
    # with one row. Measure ten scenes per layout without choosing by outcome.
    timing_indices = [i for layout in sorted({s.layout for s in partitions['test']})
                      for i in [j for j,s in enumerate(partitions['test']) if s.layout == layout][:10]]
    latency = benchmark_and_verify(candidate, [partitions['test'][i] for i in timing_indices],
                                   [predictions['selected_pipeline'][i] for i in timing_indices])
    latency['sampling'] = 'first 10 scenes of each layout, 30 total; no selection by correctness'
    by_layout = {arm: {layout: sequence_report([r for r in rows if r['layout'] == layout])
        for layout in sorted({r['layout'] for r in rows})} for arm, rows in predictions.items()}
    metadata = dict(config, source=code_info(ROOT), training_source=config['source'],
                    selection=selection, selection_sha256=selection_hash,
                    test=metrics, test_by_layout=by_layout, latency=latency,
                    test_status='evaluated_after_frozen_validation_selection',
                    metric_status='exploratory_new_identity_same_camera_dataset_unknown_legacy_OCR_lineage')
    candidate_bundle = dict(schema='research_pipeline_v1', baseline_bundle=baseline.to_bundle(),
                            verifier=verifier, selection_config=selection['config'])
    release = registry.register('research_pipeline', candidate_bundle, metadata,
        files=dict(source_snapshot=source_snapshot(ROOT),
                   training_source_snapshot=json.loads((run/'source_snapshot.json').read_text(encoding='utf-8')),
                   scene_manifest=json.loads((run/'scene_manifest.json').read_text(encoding='utf-8')),
                   region_training=json.loads((run/'region_training.json').read_text(encoding='utf-8')),
                   selection=selection, predictions=predictions,
                   runner_source={'source':Path(__file__).read_text(encoding='utf-8')}))
    reloaded, _ = load_research_release(registry, release['version'])
    probe = crop_scene(partitions['validation'][0])
    before, after = candidate.recognize_crop(probe), reloaded.recognize_crop(probe)
    np.testing.assert_allclose(before['probabilities'], after['probabilities'])
    assert before['decoded'] == after['decoded']
    benchmark_and_verify(reloaded, partitions['validation'][:1],
        score_cache(joblib.load(run/'validation_cache.joblib')[:1], **selection['config'])[1], limit=1)
    summary = dict(**metadata, release=dict(version=release['version'], path=release['path']),
                   reload_verified=True, run_path=str(run))
    write_json(run/'summary.json', summary)
    write_json(run/'test_predictions.json', predictions)
    write_json(run/'test_by_layout.json', by_layout)
    progress(json.dumps(dict(champion=selection['champion'], test=metrics, latency=latency,
                             release=summary['release']), indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['validate','test','all'], default='all')
    parser.add_argument('--run', type=Path)
    parser.add_argument('--reuse-validation-cache', action='store_true')
    parser.add_argument('--reuse-test-cache', action='store_true')
    args = parser.parse_args()
    def progress(text): print(text, flush=True)
    run = args.run
    if args.phase in {'validate','all'}:
        run = run or prepare_and_train(progress)
        validate(run, progress, reuse_cache=args.reuse_validation_cache)
    if args.phase in {'test','all'}:
        if run is None: parser.error('--run is required for test')
        test_and_register(run, progress, reuse_cache=args.reuse_test_cache)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
