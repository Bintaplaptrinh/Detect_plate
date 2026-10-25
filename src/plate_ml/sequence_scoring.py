"""Explicit OCR character alignment and localized whole-plate recognition metrics."""
import numpy as np

from .evaluation import sequence_metrics
from .imaging import intersection_over_union


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    return dict(tp=int(tp), fp=int(fp), fn=int(fn), precision=precision, recall=recall,
                f1=2*precision*recall/(precision+recall) if precision+recall else 0.)


def character_alignment(truth, predicted):
    """Unit Levenshtein alignment, tie priority: diagonal, deletion, insertion.

    A substitution contributes one FP and one FN. Alignment is for scoring
    after inference only; it never changes the predicted sequence.
    """
    n, m = len(truth), len(predicted)
    distance = np.zeros((n+1, m+1), dtype=np.int32)
    distance[:, 0], distance[0, :] = np.arange(n+1), np.arange(m+1)
    for i in range(1, n+1):
        for j in range(1, m+1):
            distance[i, j] = min(distance[i-1, j-1]+(truth[i-1] != predicted[j-1]),
                                 distance[i-1, j]+1, distance[i, j-1]+1)
    i, j = n, m
    counts = dict(correct=0, substitutions=0, deletions=0, insertions=0)
    while i or j:
        if i and j and distance[i, j] == distance[i-1, j-1]+(truth[i-1] != predicted[j-1]):
            counts['correct' if truth[i-1] == predicted[j-1] else 'substitutions'] += 1
            i -= 1; j -= 1
        elif i and distance[i, j] == distance[i-1, j]+1:
            counts['deletions'] += 1; i -= 1
        else:
            counts['insertions'] += 1; j -= 1
    return dict(counts, edit_distance=int(distance[n, m]))


def text_metrics(rows):
    if not rows or any(not r['truth'] for r in rows):
        raise ValueError('Expected paired records with nonempty truth')
    result = sequence_metrics([r['truth'] for r in rows], [r['text'] for r in rows])
    aligned = [character_alignment(r['truth'], r['text']) for r in rows]
    counts = {key: sum(a[key] for a in aligned) for key in aligned[0]}
    gt_chars, predicted_chars = sum(len(r['truth']) for r in rows), sum(len(r['text']) for r in rows)
    c, s, d, i = [counts[k] for k in ['correct', 'substitutions', 'deletions', 'insertions']]
    assert c+s+d == gt_chars and c+s+i == predicted_chars
    result.update(exact_count=sum(r['truth'] == r['text'] for r in rows),
                  nonempty_predictions=sum(bool(r['text']) for r in rows),
                  gt_characters=gt_chars, predicted_characters=predicted_chars, alignment=counts,
                  character_accuracy=c/gt_chars, character_prf=prf(c, s+i, s+d))
    return result


def joint_plate_metrics(rows, threshold=.5):
    """Top-one emitted text+box: exact string AND IoU qualify as a TP.

    Wrong emitted plates yield FP+FN, absent text yields FN. Only one GT plate
    is annotated per image. This differs from text exact match without IoU.
    """
    if not rows:
        raise ValueError('Expected nonempty paired rows')
    tp = fp = fn = 0
    for row in rows:
        emitted = bool(row['text']) and row['box'] is not None
        correct = emitted and row['text'] == row['truth'] and intersection_over_union(row['box'], row['gt_box']) >= threshold
        tp += bool(correct)
        fp += bool(emitted and not correct)
        fn += bool(not correct)
    return dict(prf(tp, fp, fn), samples=len(rows), localized_exact_match=tp/len(rows), iou_threshold=threshold)
