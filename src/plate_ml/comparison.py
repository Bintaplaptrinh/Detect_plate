"""Paired benchmark scoring. Boxes use the project's half-open xywh convention."""
from collections import Counter

import numpy as np

from .evaluation import detection_metrics, edit_distance, sequence_metrics
from .imaging import intersection_over_union


def normalize_ocr(text):
    """Uppercase and remove separators, preserving foreign alphanumeric errors."""
    return ''.join(char for char in text.upper() if char.isalnum())


def join_ocr_lines(detections, vertical_distance=12):
    """Preserve upstream OCR order while reading scalar y from quadrilateral boxes."""
    chunks, previous_y = [], None
    for bbox, text, _ in detections:
        center_y = float(np.mean([point[1] for point in bbox]))
        if previous_y is not None:
            chunks.append(' ' if abs(center_y - previous_y) < vertical_distance else '\n')
        chunks.append(text)
        previous_y = center_y
    return ''.join(chunks).strip()


def select_test_subset(manifest, count=100, seed=20261025):
    """Balanced stratified sample of an existing test set, without replacement."""
    rows = [r for r in manifest if r['partition'] == 'test']
    layouts = sorted({r['layout'] for r in rows})
    if not layouts or count < len(layouts) or count > len(rows):
        raise ValueError('Invalid stratified sample count')
    rng = np.random.default_rng(seed)
    selected = []
    for i, layout in enumerate(layouts):
        pool = sorted([r for r in rows if r['layout'] == layout], key=lambda r: r['path'])
        size = count // len(layouts) + (i < count % len(layouts))
        if size > len(pool):
            raise ValueError(f'Insufficient samples in {layout}')
        selected.extend(pool[j].copy() for j in sorted(rng.choice(len(pool), size, replace=False)))
    if len({r['group'] for r in selected}) != count:
        raise ValueError('Selected plate identities are not unique')
    rng.shuffle(selected)
    return selected


def average_precision(rows, threshold):
    """Single-class 101-point interpolated AP with global confidence ranking.

    Each image has one annotated plate. This is bbox AP, not the full COCO
    evaluator (no area/crowd categories or COCO maxDets truncation).
    """
    detections = sorted(
        [(float(d['score']), i, d['box']) for i, r in enumerate(rows) for d in r['detections']],
        key=lambda d: -d[0],
    )
    if not detections:
        return 0.0
    matched, hits = set(), []
    for _, i, box in detections:
        hit = i not in matched and intersection_over_union(box, rows[i]['gt_box']) >= threshold
        hits.append(int(hit))
        if hit:
            matched.add(i)
    tp = np.cumsum(hits)
    recall = tp / len(rows)
    precision = tp / np.arange(1, len(hits) + 1)
    return float(np.mean([max(precision[recall >= r], default=0.0) for r in np.linspace(0, 1, 101)]))


def summarize(rows, repeats):
    if not rows or repeats < 1 or any(len(r['latency_ms']) != repeats for r in rows):
        raise ValueError('Expected complete paired timing records')
    truth = [r['truth'] for r in rows]
    gt = [[r['gt_box']] for r in rows]
    ordered = [[d['box'] for d in sorted(r['detections'], key=lambda d: -d['score'])] for r in rows]
    selected = [[r['box']] if r['box'] is not None else [] for r in rows]
    # Median per image reduces temporal noise; the mean of those medians is
    # the primary latency statistic, so hard/slow images retain equal weight.
    per_image = np.median([r['latency_ms'] for r in rows], axis=1)
    aps = [average_precision(rows, round(float(t), 2)) for t in np.arange(.5, .951, .05)]
    return dict(
        decoded=sequence_metrics(truth, [r['text'] for r in rows]),
        raw=sequence_metrics(truth, [r['raw'] for r in rows]),
        exact_count=sum(r['truth'] == r['text'] for r in rows),
        no_detection=sum(not r['detections'] for r in rows),
        errors=Counter(r['error'] for r in rows if r.get('error')),
        detector_selected=detection_metrics(selected, gt),
        detector_all=detection_metrics(ordered, gt),
        ap50=aps[0], map50_95=float(np.mean(aps)),
        latency=dict(samples=len(rows), repeats=repeats, mean_ms=float(np.mean(per_image)),
                     median_ms=float(np.median(per_image)), p95_ms=float(np.percentile(per_image, 95)),
                     serial_fps=float(1000 / np.mean(per_image)),
                     mean_all_passes_ms=float(np.mean([r['latency_ms'] for r in rows]))),
        by_layout={layout: sequence_metrics([r['truth'] for r in rows if r['layout'] == layout],
                                            [r['text'] for r in rows if r['layout'] == layout])
                   for layout in sorted({r['layout'] for r in rows})},
    )


def paired_difference(a, b, seed=20261025, draws=10000):
    """Bootstrap candidate-minus-current differences, preserving image pairing."""
    from scipy.stats import binomtest

    if [r['path'] for r in a] != [r['path'] for r in b] or not a:
        raise ValueError('Expected identically ordered paired predictions')
    truth = [r['truth'] for r in a]
    correct_a = np.array([r['text'] == t for r, t in zip(a, truth)])
    correct_b = np.array([r['text'] == t for r, t in zip(b, truth)])
    delta = correct_b.astype(float) - correct_a
    edits = np.array([edit_distance(t, y['text']) - edit_distance(t, x['text'])
                      for t, x, y in zip(truth, a, b)])
    lengths = np.array(list(map(len, truth)))
    idx = np.random.default_rng(seed).integers(0, len(a), (draws, len(a)))
    wins, losses = int(np.sum(~correct_a & correct_b)), int(np.sum(correct_a & ~correct_b))
    return dict(exact_match_delta=float(delta.mean()),
                exact_match_delta_ci95=np.quantile(delta[idx].mean(axis=1), [.025, .975]).tolist(),
                cer_delta=float(edits.sum() / lengths.sum()),
                cer_delta_ci95=np.quantile(edits[idx].sum(axis=1) / lengths[idx].sum(axis=1), [.025, .975]).tolist(),
                candidate_only_correct=wins, current_only_correct=losses,
                mcnemar_exact_p=float(binomtest(wins, wins + losses, .5).pvalue) if wins + losses else 1.0,
                bootstrap_draws=draws, seed=seed)
