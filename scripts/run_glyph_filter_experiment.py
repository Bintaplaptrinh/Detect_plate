"""Image-quality rejection ablation, regression audit, and fresh identity holdout."""
from dataclasses import asdict, replace
from datetime import datetime, timezone
from hashlib import sha256
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from run_research_experiment import load_research_release
from plate_ml.evaluation import sequence_metrics
from plate_ml.glyph_quality import GlyphQualityConfig, QualityFilteredSegmenter
from plate_ml.imaging import read_image
from plate_ml.pipeline import PlateRecognizer
from plate_ml.registry import ModelRegistry, code_info, source_snapshot
from plate_ml.research_pipeline import ResearchRecognizer
from plate_ml.research_experiment import fresh_partitions, recognizer_for, write_json
from plate_ml.vnlp import Scene

PREVIOUS = ROOT/'reports/20261022T091902580679Z-research'
BASE_VERSION = 'v3'
CASES = ('36A40669','29X2534')


def make_filtered(base, quality=GlyphQualityConfig()):
    return ResearchRecognizer(base.model,base.detector,QualityFilteredSegmenter(base.segmenter,quality),
        base.extractor.config,base.templates,base.rerank_weight,base.top_k)


def load_filter_release(registry,version):
    saved, receipt = registry.load('research_glyph_filter',version,trusted=True)
    base_bundle = saved['base_bundle']
    baseline = PlateRecognizer.from_bundle(base_bundle['baseline_bundle'],registry.root/'RuntimeAssets')
    base = recognizer_for(base_bundle['selection_config'],baseline,base_bundle['verifier'])
    return make_filtered(base,GlyphQualityConfig(**saved['quality_config'])),receipt


def scene_objects(manifest,partition):
    return [Scene(r['path'],r['text'],tuple(r['box']),r['layout'],r['group'])
            for r in manifest if r['partition']==partition]


def evaluate_paired(scenes,arms,progress=print):
    """Share only detector proposals; each arm classifies its own kept glyph list."""
    predictions = {name:[] for name in arms}
    for i,scene in enumerate(scenes):
        image = read_image(scene.path)
        detected = next(iter(arms.values())).detector.detect(image)
        for name,recognizer in arms.items():
            start = perf_counter()
            options = []
            for candidate in detected.candidates[:recognizer.top_k]:
                result = recognizer.recognize_crop(candidate.crop)
                result['candidate'] = candidate
                result['score'] = candidate.score+recognizer.rerank_weight*result['quality']
                options.append(result)
            best = max(options,key=lambda r:r['score'],default=None)
            diagnostics = best['segmentation'].diagnostics if best else {}
            predictions[name].append(dict(path=scene.path,truth=scene.text,layout=scene.layout,
                raw=best['decoded']['raw'] if best else '',text=best['decoded']['text'] if best else '',
                box=list(best['candidate'].box) if best else None,
                method=diagnostics.get('method'),threshold=diagnostics.get('threshold'),
                glyph_quality=diagnostics.get('glyph_quality'),
                post_detector_seconds=perf_counter()-start))
        if progress and ((i+1)%15==0 or i+1==len(scenes)):
            progress(f'paired inference {i+1}/{len(scenes)}')
    metrics = {arm:dict(raw=sequence_metrics([r['truth'] for r in rows],[r['raw'] for r in rows]),
                       decoded=sequence_metrics([r['truth'] for r in rows],[r['text'] for r in rows]),
                       exact_count=sum(r['truth']==r['text'] for r in rows))
               for arm,rows in predictions.items()}
    return metrics,predictions


def changes(predictions,candidate='full_filter'):
    before,after=predictions['baseline_v3'],predictions[candidate]
    assert [r['path'] for r in before]==[r['path'] for r in after]
    repaired,regressed,changed=[],[],[]
    for a,b in zip(before,after):
        item=dict(path=a['path'],truth=a['truth'],before=a['text'],after=b['text'])
        if a['text']!=b['text']:changed.append(item)
        if a['text']!=a['truth'] and b['text']==b['truth']:repaired.append(item)
        if a['text']==a['truth'] and b['text']!=b['truth']:regressed.append(item)
    return dict(repaired=repaired,regressed=regressed,changed=changed)


def validate(progress=print):
    cv2.setNumThreads(1)
    registry=ModelRegistry(ROOT/'Registration')
    base,_=load_research_release(registry,BASE_VERSION)
    manifest=json.loads((PREVIOUS/'scene_manifest.json').read_text(encoding='utf-8'))
    run=ROOT/'reports'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-glyph-filter')
    run.mkdir(parents=True,exist_ok=False)
    full=GlyphQualityConfig()
    arms=dict(baseline_v3=base,
        height_only=make_filtered(base,replace(full,filter_fragmented_edges=False,filter_solid_edges=False)),
        full_filter=make_filtered(base,full))
    cases=[Scene(r['path'],r['text'],tuple(r['box']),r['layout'],r['group']) for r in manifest if r['text'] in CASES]
    case_metrics,case_predictions=evaluate_paired(cases,arms,progress)
    assert len(cases)==len(CASES)
    assert all(r['raw']==r['truth'] for r in case_predictions['full_filter']), 'Reported examples must be corrected without GT-guided inference'
    write_json(run/'reported_cases.json',dict(metrics=case_metrics,predictions=case_predictions))
    metrics,predictions=evaluate_paired(scene_objects(manifest,'validation'),arms,progress)
    # Two user-reported cases are development regressions, never heldout evidence.
    # The height-only arm is an ablation; it does not satisfy the missing-3 case.
    better = (metrics['full_filter']['raw']['character_error_rate'] <=
              metrics['baseline_v3']['raw']['character_error_rate'])
    selected='full_filter' if better else 'baseline_v3'
    selection=dict(selected=selected,quality_config=asdict(full),source=code_info(ROOT),
        criterion='full filter must repair both reported examples and not increase validation raw CER',
        metrics=metrics,changes=changes(predictions),
        fresh_test_status='not_evaluated',baseline_version=BASE_VERSION)
    write_json(run/'selection_before_fresh_test.json',selection)
    write_json(run/'validation_predictions.json',predictions)
    progress('RUN: '+str(run))
    progress(json.dumps(selection,indent=2))
    return run


def finalize(run,progress=print):
    cv2.setNumThreads(1)
    registry=ModelRegistry(ROOT/'Registration')
    base,receipt=load_research_release(registry,BASE_VERSION)
    selection_path=run/'selection_before_fresh_test.json'
    selection=json.loads(selection_path.read_text(encoding='utf-8'))
    assert selection['source']['source_sha256']==code_info(ROOT)['source_sha256']
    selection_hash=sha256(selection_path.read_bytes()).hexdigest()
    quality=GlyphQualityConfig(**selection['quality_config'])
    arms=dict(baseline_v3=base,full_filter=make_filtered(base,quality))
    old_manifest=json.loads((PREVIOUS/'scene_manifest.json').read_text(encoding='utf-8'))
    # Old 180 scenes include the reported examples: report as regression audit,
    # not an independent benchmark after developing this fix from those examples.
    old_metrics,old_predictions=evaluate_paired(scene_objects(old_manifest,'test'),arms,progress)
    write_json(run/'old_test_regression.json',dict(metrics=old_metrics,predictions=old_predictions,changes=changes(old_predictions)))
    prior_02=json.loads((registry.root/'Registry/pipeline_vnlp_experiment/v1/scene_manifest.json').read_text(encoding='utf-8'))
    prior_by_path={r['path']:r for r in prior_02+old_manifest}
    partitions,fresh_manifest,audit=fresh_partitions(
        Path(old_manifest[0]['path']).parents[1],list(prior_by_path.values()),
        seed=2026102204,val_per_layout=0,test_per_layout=60)
    fresh=partitions['test']
    write_json(run/'fresh_test_manifest.json',[r for r in fresh_manifest if r['partition']=='test'])
    metrics,predictions=evaluate_paired(fresh,arms,progress)
    assert sha256(selection_path.read_bytes()).hexdigest()==selection_hash
    by_layout={arm:{layout:sequence_metrics([r['truth'] for r in rows if r['layout']==layout],
        [r['text'] for r in rows if r['layout']==layout]) for layout in sorted({r['layout'] for r in rows})}
        for arm,rows in predictions.items()}
    metadata=dict(baseline_version=BASE_VERSION,baseline_artifacts=receipt['artifact_sha256'],
        selection=selection,selection_sha256=selection_hash,source=code_info(ROOT),
        fresh_test=metrics,fresh_by_layout=by_layout,fresh_changes=changes(predictions),
        fresh_split=audit,old_regression_metrics=old_metrics,
        old_regression_changes=changes(old_predictions),
        evaluation_status='fresh_identity_same_VNLP_camera_distribution_unknown_legacy_OCR_lineage')
    saved,_=registry.load('research_pipeline',BASE_VERSION,trusted=True)
    release=registry.register('research_glyph_filter',dict(base_bundle=saved,quality_config=asdict(quality)),metadata,
        files=dict(source_snapshot=source_snapshot(ROOT),fresh_manifest=[r for r in fresh_manifest if r['partition']=='test'],
            predictions=predictions,old_regression=old_predictions,
            reported_cases=json.loads((run/'reported_cases.json').read_text(encoding='utf-8')),
            runner_source={'source':Path(__file__).read_text(encoding='utf-8')}))
    restored,_=load_filter_release(registry,release['version'])
    probe=read_image(fresh[0].path)
    _,a=arms['full_filter'].recognize(probe);_,b=restored.recognize(probe)
    assert bool(a)==bool(b)
    if a:
        assert a[0]['decoded']==b[0]['decoded']
        np.testing.assert_allclose(a[0]['probabilities'],b[0]['probabilities'])
    summary=dict(metadata,run_path=str(run),release=dict(version=release['version'],path=release['path']),reload_verified=True)
    write_json(run/'summary.json',summary)
    write_json(run/'fresh_predictions.json',predictions)
    progress(json.dumps(dict(fresh_test=metrics,old_test=old_metrics,release=summary['release']),indent=2))
    return summary


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['validation','finalize','all'],default='all')
    parser.add_argument('--run',type=Path)
    args=parser.parse_args()
    def progress(s):print(s,flush=True)
    run=validate(progress) if args.phase in ('validation','all') else args.run
    if args.phase in ('finalize','all'):
        if run is None:parser.error('--run required for finalize')
        finalize(run,progress)
