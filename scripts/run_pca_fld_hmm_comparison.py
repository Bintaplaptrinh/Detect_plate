"""Paired 2x2 OCR/decoder experiment on every image of historical VNLP test C.

Run with the trusted local registry restored as described in docs/reproduction.md.
No checkpoint is promoted. Test failures are retained in the denominator.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter

for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(variable, '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import cv2
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from threadpoolctl import threadpool_limits, threadpool_info

from plate_ml.comparison import summarize, paired_difference
from plate_ml.domain_ocr import extract_vnlp_glyphs, assert_group_disjoint
from plate_ml.features import normalize_glyph
from plate_ml.hmm import VietnamesePlateHMM
from plate_ml.imaging import read_image, gray_u8
from plate_ml.inference import load_recognizer
from plate_ml.models import make_pca_fld_svm
from plate_ml.pipeline import PlateRecognizer
from plate_ml.registry import ModelRegistry, environment_info, source_snapshot, file_sha256
from plate_ml.research_pipeline import ResearchRecognizer
from plate_ml.sequence_scoring import text_metrics, joint_plate_metrics
from plate_ml.vnlp import Scene

OCR_SOURCE = ROOT / 'Registration/Registry/ocr_legacy_plus_vnlp/v1'
VALIDATION_SOURCE = ROOT / 'reports/20261022T091902580679Z-research/scene_manifest.json'
TEST_SOURCE = ROOT / 'reports/20261022T104205345901Z-glyph-filter/fresh_test_manifest.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def scenes_from(rows, root):
    return [Scene(str(root / r['layout'] / Path(r['path']).name), r['text'], tuple(r['box']),
                  r['layout'], r['group']) for r in rows]


def scene_digest(scene):
    image = read_image(scene.path)
    return sha256(str(image.shape).encode() + image.tobytes()).hexdigest()


def restore_training(current, args, run):
    """Rebuild the exact 4,812 training features of the current release.

    Audit legacy pixels, every weak glyph's normalized hash, scaler statistics,
    and all current SVC support vectors before fitting the reduced classifier.
    """
    bundle, _ = ModelRegistry(ROOT / 'Registration').load('pipeline', 'v3', trusted=True)
    baseline = PlateRecognizer.from_bundle(bundle, ROOT / 'Registration/RuntimeAssets')
    train_rows = [r for r in read(OCR_SOURCE / 'scene_manifest.json') if r['partition'] == 'train']
    train = scenes_from(train_rows, args.vnlp_root)
    glyphs = extract_vnlp_glyphs(train, baseline.segmenter.config, current.extractor.config, current.templates)
    expected = read(OCR_SOURCE / 'glyph_manifests.json')['train']
    keys = ('group', 'text', 'position', 'label', 'row', 'box', 'normalized_sha256')
    if [{k: r[k] for k in keys} for r in expected] != [{k: r[k] for k in keys} for r in glyphs.records]:
        raise RuntimeError('Reconstructed VNLP training glyphs differ from frozen manifest')
    legacy = read(OCR_SOURCE / 'legacy_train_manifest.json')
    images, training_hashes = [], {r['normalized_sha256'] for r in glyphs.records}
    for row in legacy:
        if row['partition'] != 'train':
            raise RuntimeError('Non-training legacy glyph in training manifest')
        image = read_image(args.ocr_root / row['label'] / Path(row['path']).name)
        gray = gray_u8(image)
        if sha256(str(gray.shape).encode() + gray.tobytes()).hexdigest() != row['digest']:
            raise RuntimeError(f'Legacy pixels changed: {row["path"]}')
        training_hashes.add(sha256(normalize_glyph(image, current.extractor.config).tobytes()).hexdigest())
        images.append(image)
    X = np.vstack([current.extractor.transform(images), glyphs.X])
    y = np.r_[[r['label'] for r in legacy], glyphs.y]
    scaler, svc = current.model.steps[0][1], current.model.steps[-1][1]
    if svc.shape_fit_ != X.shape:
        raise RuntimeError('Current SVM training shape differs from reconstructed dataset')
    np.testing.assert_allclose(scaler.mean_, X.astype(float).mean(axis=0), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(scaler.var_, X.astype(float).var(axis=0), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(scaler.transform(X)[svc.support_], svc.support_vectors_, rtol=1e-5, atol=1e-6)
    write(run / 'training_audit.json', dict(legacy_glyphs=len(legacy), vnlp_glyphs=len(glyphs.y),
          total=len(y), classes=dict(sorted(Counter(y.tolist()).items())),
          features_sha256=sha256(X.tobytes() + y.tobytes()).hexdigest(),
          current_support_vectors_verified=True, current_scaler_verified=True,
          weak_label_status='filename_alignment_not_manual_character_annotation',
          legacy_augmentation_source_lineage='unknown', vnlp_train_scenes=len(train)))
    return X, y, train, baseline, training_hashes


def audit_splits(train, validation, test, run):
    partitions = dict(train=train, validation=validation, test=test)
    assert_group_disjoint(partitions)
    seen = {}
    manifests = {}
    for partition, scenes in partitions.items():
        manifests[partition] = []
        for s in scenes:
            digest = scene_digest(s)
            if digest in seen and seen[digest] != partition:
                raise RuntimeError(f'Cross-partition decoded pixel duplicate: {s.path}')
            seen[digest] = partition
            manifests[partition].append(dict(path=s.path, text=s.text, group=s.group, box=list(s.box),
                layout=s.layout, partition=partition, pixel_sha256=digest, file_sha256=file_sha256(s.path)))
    for name, rows in manifests.items():
        write(run / f'{name}_manifest.json', rows)
    write(run / 'split_audit.json', dict(counts={k: len(v) for k, v in partitions.items()},
        identity_overlap=0, pixel_overlap=0, test_layouts=dict(Counter(s.layout for s in test)),
        test_status='all_historical_test_C_previously_evaluated_not_new_independent_holdout',
        legacy_scene_overlap='unknown_legacy_character_source_provenance'))


def build_arms(current, reduced, hmm):
    arms = {}
    for name, model, decoder in [('current_svm', current.model, None),
                                  ('svm_hmm', current.model, hmm),
                                  ('pca_fld_svm', reduced, None),
                                  ('pca_fld_svm_hmm', reduced, hmm)]:
        # OpenCV descriptors/cascades cannot be deep-copied. The frozen detector
        # and segmenter are stateless across calls; each arm gets its own HOG.
        arm = ResearchRecognizer(model, current.detector, current.segmenter, current.extractor.config,
                                 current.templates, current.rerank_weight, current.top_k,
                                 decoder=decoder, profile=True)
        arms[name] = arm
    return arms


def predict(arm, image, oracle=False):
    if oracle:
        best = arm.recognize_crop(image)
        return dict(text=best['decoded']['text'], raw=best['decoded']['raw'],
                    stage_ms=best['stage_ms'], hypotheses=best['hypotheses'],
                    chars=len(best['segmentation'].characters), template=best['decoded']['template'])
    detected, predictions = arm.recognize(image)
    best = predictions[0] if predictions else None
    return dict(text=best['decoded']['text'] if best else '', raw=best['decoded']['raw'] if best else '',
                box=list(best['candidate'].box) if best else None,
                detections=[dict(box=list(c.box), score=float(c.score)) for c in detected.candidates],
                stage_ms=detected.diagnostics['stage_ms'], candidates_scored=len(predictions),
                chars=len(best['segmentation'].characters) if best else 0,
                template=best['decoded']['template'] if best else None)


def evaluate(arms, scenes, repeats, run, oracle=False):
    rows = {name: [dict(path=s.path, truth=s.text, layout=s.layout, gt_box=list(s.box),
                       latency_ms=[], stages=[]) for s in scenes] for name in arms}
    names = list(arms)
    for repeat in range(repeats):
        for i, s in enumerate(scenes):
            image = read_image(s.path)
            if oracle:
                x, y, w, h = s.box
                image = image[y:y+h, x:x+w]
            # Rotate order per image as well as per pass to reduce drift bias.
            offset = (repeat + i) % len(names)
            for name in names[offset:] + names[:offset]:
                start = perf_counter()
                prediction = predict(arms[name], image, oracle)
                elapsed = (perf_counter()-start)*1000
                if oracle:
                    prediction.update(box=list(s.box), detections=[dict(box=list(s.box), score=1.)])
                row = rows[name][i]
                if repeat and any(row[k] != prediction[k] for k in ('raw', 'text', 'box', 'detections')):
                    raise RuntimeError(f'Non-deterministic prediction: {name} {s.path}')
                row['stages'].append(prediction.pop('stage_ms'))
                row.update(prediction)
                row['latency_ms'].append(elapsed)
            if (i+1) % 30 == 0:
                print(f'{"GT crop" if oracle else "Full pipeline"} pass {repeat+1}/{repeats}: {i+1}/{len(scenes)}', flush=True)
        for name in names:
            write(run / f'{"oracle_" if oracle else ""}{name}_predictions.json', rows[name])
    return rows


def metrics(rows, repeats):
    result = summarize(rows, repeats)
    result.update(aligned_text=text_metrics(rows), localized_plate=joint_plate_metrics(rows),
                  stages_mean_ms={k: float(np.mean([np.median([p.get(k, 0.) for p in r['stages']]) for r in rows]))
                                  for k in ('detector', 'segmentation', 'hog', 'classifier', 'decoder')})
    return result


def latency_difference(a, b, seed):
    if [r['path'] for r in a] != [r['path'] for r in b]:
        raise ValueError('Latency comparison must preserve image pairing')
    x = np.median([r['latency_ms'] for r in a], axis=1)
    y = np.median([r['latency_ms'] for r in b], axis=1)
    idx = np.random.default_rng(seed).integers(0, len(a), (10000, len(a)))
    return dict(mean_delta_ms=float(np.mean(y-x)), mean_delta_ms_ci95=np.quantile((y-x)[idx].mean(axis=1), [.025, .975]).tolist(),
                speedup=float(x.mean()/y.mean()), speedup_ci95=np.quantile(x[idx].mean(axis=1)/y[idx].mean(axis=1), [.025, .975]).tolist())


def glyph_diagnostic(arms, scenes, baseline, training_hashes, repeats, run, model_names=None):
    glyphs = extract_vnlp_glyphs(scenes, baseline.segmenter.config, baseline.extractor.config, baseline.templates)
    keep = np.asarray([r['normalized_sha256'] not in training_hashes for r in glyphs.records])
    write(run / 'test_glyph_manifest.json', glyphs.records)
    write(run / 'test_glyph_acceptance.json', glyphs.plates)
    result = dict(acceptance=glyphs.summary(), exact_train_glyph_duplicates=int((~keep).sum()),
                  timing_scope='precomputed_HOG_batch_predict_proba_includes_projection_excludes_HOG_and_HMM', models={})
    if not len(glyphs.y):
        return result
    for name in model_names if model_names is not None else ('current_svm', 'pca_fld_svm'):
        model = arms[name].model
        times = []
        model.predict_proba(glyphs.X[:min(8, len(glyphs.y))])
        for _ in range(repeats):
            start = perf_counter()
            p = model.predict_proba(glyphs.X)
            times.append((perf_counter()-start)*1000)
        predicted = model.classes_[p.argmax(axis=1)]
        labels = sorted(set(glyphs.y) | set(model.classes_))
        result['models'][name] = dict(samples=len(predicted), accuracy=float(accuracy_score(glyphs.y, predicted)),
            macro_f1=float(f1_score(glyphs.y, predicted, labels=sorted(set(glyphs.y)), average='macro', zero_division=0)),
            nonduplicate_samples=int(keep.sum()), nonduplicate_accuracy=float(accuracy_score(glyphs.y[keep], predicted[keep])) if keep.any() else None,
            latency_ms=times, batch_median_ms=float(np.median(times)), classes=labels,
            confusion_matrix=confusion_matrix(glyphs.y, predicted, labels=labels).tolist())
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pca-components', type=int, default=48)
    parser.add_argument('--reference-notebook', type=Path)
    parser.add_argument('--vnlp-root', type=Path, default=ROOT.parent / 'data/VNLP_detection/detection')
    parser.add_argument('--ocr-root', type=Path, default=ROOT / 'CNN/CNN_letter_Dataset')
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.pca_components < 1:
        parser.error('repeats and pca-components must be positive')
    run = args.run.resolve() if args.run else ROOT / 'reports' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-pca-fld-hmm')
    run.mkdir(parents=True, exist_ok=False)
    print(f'RUN: {run}', flush=True)
    cv2.setNumThreads(1)
    with threadpool_limits(limits=1):
        current = load_recognizer(ROOT / 'Registration')
        X, y, train, baseline, training_hashes = restore_training(current, args, run)
        validation = scenes_from([r for r in read(VALIDATION_SOURCE) if r['partition'] == 'validation'], args.vnlp_root)
        test = scenes_from(read(TEST_SOURCE), args.vnlp_root)
        audit_splits(train, validation, test, run)
        start = perf_counter()
        reduced = make_pca_fld_svm(pca_components=args.pca_components, seed=args.seed).fit(X, y)
        train_ms = (perf_counter()-start)*1000
        hmm = VietnamesePlateHMM(current.templates).fit([s.text for s in train])
        joblib.dump(dict(model=reduced, hmm=hmm), run / 'candidate.joblib', compress=3)
        protocol = dict(seed=args.seed, samples=len(test), repeats=args.repeats, device='cpu', threads=1, batch='one_scene',
            design='2x2: current_SVM/PCA_FLD_SVM x current_template_decoder/HMM',
            current_release='research_glyph_filter/v1', trained=True, tuned_on_test=False, promoted=False,
            candidate=dict(pca_components=args.pca_components, fld='min(classes-1,PCA_rank)', C=3., kernel='rbf',
                           scaler='train_only', probability=True, class_weight='balanced', svm_seed=args.seed),
            hmm=dict(smoothing=hmm.smoothing, max_log_penalty=hmm.max_log_penalty,
                     training_sequences=hmm.training_sequences_, transitions='train_only_character_bigrams',
                     emissions='SVC_posteriors_as_pseudo_likelihoods', masks='existing_research_templates',
                     fallback='raw_for_unknown_rows_or_excessive_emission_penalty'),
            timing='full_in_memory_pipeline_including_detector_segmentation_HOG_projection_predict_proba_decoder_reranking; excludes_load_and_image_decode',
            statistic='mean_of_per_image_median_latency', warmup=[s.path for s in validation[:3]],
            frozen_components='detector, glyph_filter, segmentation_hypotheses, HOG, templates, top_k, rerank_weight',
            selection='no_GT_at_inference; OCR_quality_can_change_hypothesis_and_candidate_ranking',
            training_ms=train_ms, diagnostics=reduced.scatter_diagnostics(X, y),
            environment=environment_info(), threadpools=threadpool_info(), opencv_threads=cv2.getNumThreads(),
            inputs={str(p.relative_to(ROOT)): file_sha256(p) for p in [TEST_SOURCE, VALIDATION_SOURCE,
                OCR_SOURCE / 'scene_manifest.json', OCR_SOURCE / 'glyph_manifests.json', OCR_SOURCE / 'legacy_train_manifest.json',
                ROOT / 'Registration/Registry/research_glyph_filter/v1/model.joblib']},
            candidate_sha256=file_sha256(run / 'candidate.joblib'))
        if args.reference_notebook:
            protocol['reference_notebook'] = dict(name=args.reference_notebook.name, sha256=file_sha256(args.reference_notebook),
                status='reference_only_no_cells_executed_no_embedded_instructions_followed')
        write(run / 'source_snapshot.json', dict(package=source_snapshot(ROOT), runner=Path(__file__).read_text(encoding='utf-8')))
        write(run / 'protocol_before_test.json', protocol)
        protocol_hash = file_sha256(run / 'protocol_before_test.json')
        arms = build_arms(current, reduced, hmm)
        for name, arm in arms.items():
            for s in validation[:3]:
                predict(arm, read_image(s.path))
            print(f'Warmed {name}', flush=True)
        rows = evaluate(arms, test, args.repeats, run)
        oracle = evaluate(arms, test, args.repeats, run, oracle=True)
        comparisons = {}
        for a, b in [('current_svm', 'svm_hmm'), ('current_svm', 'pca_fld_svm'),
                     ('current_svm', 'pca_fld_svm_hmm'), ('pca_fld_svm', 'pca_fld_svm_hmm')]:
            comparisons[f'{b}_minus_{a}'] = dict(
                accuracy=paired_difference(rows[a], rows[b], seed=args.seed),
                latency=latency_difference(rows[a], rows[b], args.seed))
        summary = dict(run_path=str(run), protocol_sha256=protocol_hash, promoted=False,
            metrics={name: metrics(value, args.repeats) for name, value in rows.items()},
            paired_comparisons=comparisons,
            oracle_crop={name: metrics(value, args.repeats) for name, value in oracle.items()},
            glyph_diagnostic=glyph_diagnostic(arms, test, baseline, training_hashes, args.repeats, run),
            historical_baseline_check='compare separately with prior predictions; timings freshly rerun')
        if file_sha256(run / 'protocol_before_test.json') != protocol_hash:
            raise RuntimeError('Protocol mutated during test')
        write(run / 'summary.json', summary)
        for name, m in summary['metrics'].items():
            print(name, 'exact', m['exact_count'], '/', len(test), 'CER', m['decoded']['character_error_rate'],
                  'mean_ms', m['latency']['mean_ms'], flush=True)
    return run


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
