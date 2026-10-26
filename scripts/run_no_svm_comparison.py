"""Separate paired OCR benchmark: PCA + FLD + Gaussian-emission HMM, no OCR SVM."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from time import perf_counter

# Import the shared runner first: it sets CPU thread limits before NumPy loads.
from run_pca_fld_hmm_comparison import (
    ROOT, OCR_SOURCE, VALIDATION_SOURCE, TEST_SOURCE, read, write, restore_training,
    scenes_from, audit_splits, predict, evaluate, metrics, latency_difference, glyph_diagnostic,
)
import cv2
import joblib
import numpy as np
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits, threadpool_info

from plate_ml.comparison import paired_difference
from plate_ml.hmm import VietnamesePlateHMM
from plate_ml.imaging import read_image
from plate_ml.inference import load_recognizer
from plate_ml.models import PCA_FLD_Gaussian
from plate_ml.registry import file_sha256, source_snapshot, environment_info
from plate_ml.research_pipeline import ResearchRecognizer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--reference-run', type=Path, default=ROOT / 'reports/pca-fld-hmm-180-20261026-v2')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--vnlp-root', type=Path, default=ROOT.parent / 'data/VNLP_detection/detection')
    parser.add_argument('--ocr-root', type=Path, default=ROOT / 'CNN/CNN_letter_Dataset')
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error('repeats must be positive')
    run = args.run.resolve() if args.run else ROOT / 'reports' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '-no-svm')
    run.mkdir(parents=True, exist_ok=False)
    print(f'RUN: {run}', flush=True)
    cv2.setNumThreads(1)
    with threadpool_limits(limits=1):
        reference = args.reference_run.resolve()
        original = read(reference / 'protocol_before_test.json')
        if file_sha256(reference / 'candidate.joblib') != original['candidate_sha256']:
            raise ValueError('Reference candidate checksum mismatch')
        saved = joblib.load(reference / 'candidate.joblib')  # Trusted local benchmark artifact.
        current = load_recognizer(ROOT / 'Registration')
        X, y, train, baseline, training_hashes = restore_training(current, args, run)
        if read(run / 'training_audit.json')['features_sha256'] != read(reference / 'training_audit.json')['features_sha256']:
            raise ValueError('Training features differ from original paired experiment')
        validation = scenes_from([r for r in read(VALIDATION_SOURCE) if r['partition'] == 'validation'], args.vnlp_root)
        test = scenes_from(read(TEST_SOURCE), args.vnlp_root)
        audit_splits(train, validation, test, run)
        start = perf_counter()
        gaussian = PCA_FLD_Gaussian(pca_components=original['candidate']['pca_components']).fit(X, y)
        train_ms = (perf_counter()-start)*1000
        # Equal projections isolate the emission model rather than changing FLD.
        np.testing.assert_allclose(gaussian.transform(X), saved['model'].transform(X), rtol=1e-6, atol=1e-6)
        if any(isinstance(v, SVC) for v in vars(gaussian).values()) or hasattr(gaussian, 'svm_'):
            raise RuntimeError('Unexpected SVM in the no-SVM observation model')
        hmm = VietnamesePlateHMM(current.templates).fit([s.text for s in train])
        np.testing.assert_array_equal(hmm.transitions_, saved['hmm'].transitions_)
        np.testing.assert_array_equal(hmm.initial_, saved['hmm'].initial_)
        joblib.dump(dict(model=gaussian, hmm=hmm), run / 'no_svm_candidate.joblib', compress=3)
        specifications = [('current_svm', current.model, None),
                          ('pca_fld_svm_hmm', saved['model'], hmm),
                          ('pca_fld_hmm', gaussian, hmm)]
        arms = {name: ResearchRecognizer(model, current.detector, current.segmenter, current.extractor.config,
                    current.templates, current.rerank_weight, current.top_k, decoder=decoder, profile=True)
                for name, model, decoder in specifications}
        protocol = dict(samples=len(test), repeats=args.repeats, seed=args.seed, threads=1, device='cpu',
            design='separate_paired_current_SVM_vs_PCA_FLD_SVM_HMM_vs_PCA_FLD_Gaussian_HMM',
            current_release='research_glyph_filter/v1', promoted=False, tuned_on_test=False,
            no_svm_scope='OCR_observation_model_only; common_region_verifier_detector_still_uses_SVM',
            observation_model=dict(type='one_Gaussian_per_character_tied_full_covariance',
                likelihood='multivariate_normal_in_Fisher_space', normalization='row_softmax_log_likelihood_equal_class_priors',
                covariance_regularization=gaussian.covariance_regularization,
                covariance_estimator='pooled_train_within_class_unbiased_N_minus_C',
                feature_pipeline='unchanged_HOG_540_to_StandardScaler_to_PCA_48_to_FLD_34',
                no_ocr_svm=True, identical_projection_to_reference_verified=True),
            hmm=dict(transitions='identical_train_only_bigrams_to_original_experiment',
                training_sequences=hmm.training_sequences_, smoothing=hmm.smoothing,
                max_log_penalty=hmm.max_log_penalty, templates='unchanged_research_templates',
                gaussian_emissions='normalized_genuine_Gaussian_likelihoods',
                svm_comparator_emissions='SVC_posteriors_as_pseudo_likelihoods'),
            timing=original['timing'], statistic=original['statistic'],
            warmup=[s.path for s in validation[:3]], training_ms=train_ms,
            diagnostics=gaussian.scatter_diagnostics(X, y),
            reference_run=str(reference), reference_candidate_sha256=file_sha256(reference / 'candidate.joblib'),
            candidate_sha256=file_sha256(run / 'no_svm_candidate.joblib'),
            frozen_components=original['frozen_components'],
            environment=environment_info(), threadpools=threadpool_info(), opencv_threads=cv2.getNumThreads(),
            inputs={str(p.relative_to(ROOT)): file_sha256(p) for p in [TEST_SOURCE, VALIDATION_SOURCE,
                OCR_SOURCE / 'scene_manifest.json', OCR_SOURCE / 'glyph_manifests.json', OCR_SOURCE / 'legacy_train_manifest.json',
                ROOT / 'Registration/Registry/research_glyph_filter/v1/model.joblib']})
        write(run / 'protocol_before_test.json', protocol)
        protocol_hash = file_sha256(run / 'protocol_before_test.json')
        shared_runner = Path(__file__).with_name('run_pca_fld_hmm_comparison.py')
        write(run / 'source_snapshot.json', dict(package=source_snapshot(ROOT),
            runner=Path(__file__).read_text(encoding='utf-8'), shared_runner=shared_runner.read_text(encoding='utf-8')))
        for name, arm in arms.items():
            for s in validation[:3]:
                predict(arm, read_image(s.path))
            print(f'Warmed {name}', flush=True)
        rows = evaluate(arms, test, args.repeats, run)
        oracle = evaluate(arms, test, args.repeats, run, oracle=True)
        comparisons = {}
        for name in ('current_svm', 'pca_fld_svm_hmm'):
            comparisons[f'pca_fld_hmm_minus_{name}'] = dict(
                accuracy=paired_difference(rows[name], rows['pca_fld_hmm'], seed=args.seed),
                latency=latency_difference(rows[name], rows['pca_fld_hmm'], args.seed))
        summary = dict(run_path=str(run), protocol_sha256=protocol_hash, promoted=False,
            metrics={name: metrics(value, args.repeats) for name, value in rows.items()},
            paired_comparisons=comparisons,
            oracle_crop={name: metrics(value, args.repeats) for name, value in oracle.items()},
            glyph_diagnostic=glyph_diagnostic(arms, test, baseline, training_hashes, args.repeats, run,
                                             model_names=list(arms)))
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
