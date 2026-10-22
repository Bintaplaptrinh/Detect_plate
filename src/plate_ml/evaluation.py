import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss
from .imaging import intersection_over_union


def evaluate_classifier(model, X, y) -> dict:
    probabilities = model.predict_proba(X)
    predictions = model.classes_[np.argmax(probabilities, axis=1)]
    labels = list(model.classes_)
    return dict(accuracy=float(accuracy_score(y, predictions)),
                macro_f1=float(f1_score(y, predictions, labels=labels, average="macro", zero_division=0)),
                log_loss=float(log_loss(y, probabilities, labels=labels)),
                classes=labels, confusion_matrix=confusion_matrix(y, predictions, labels=labels).tolist(),
                report=classification_report(y, predictions, labels=labels, output_dict=True, zero_division=0))


def confusion_pairs(metrics: dict, limit=20) -> list[dict]:
    matrix = np.asarray(metrics["confusion_matrix"])
    pairs = []
    for i, label in enumerate(metrics["classes"]):
        for j, predicted in enumerate(metrics["classes"]):
            if i != j and matrix[i, j]:
                pairs.append(dict(truth=label, predicted=predicted, count=int(matrix[i, j])))
    return sorted(pairs, key=lambda p: p["count"], reverse=True)[:limit]


def edit_distance(a: str, b: str) -> int:
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        next_row = [i]
        for j, cb in enumerate(b, 1):
            next_row.append(min(next_row[-1] + 1, row[j] + 1, row[j - 1] + (ca != cb)))
        row = next_row
    return row[-1]


def sequence_metrics(truth: list[str], predictions: list[str]) -> dict:
    if len(truth) != len(predictions) or not truth or not sum(map(len, truth)):
        raise ValueError("Expected nonempty paired sequences")
    return dict(exact_match=float(np.mean([a == b for a, b in zip(truth, predictions)])),
                character_error_rate=sum(edit_distance(a, b) for a, b in zip(truth, predictions)) / sum(map(len, truth)),
                samples=len(truth))


def detection_metrics(predictions: list[list[tuple]], truth: list[list[tuple]], iou_threshold=0.5) -> dict:
    """Greedy one-to-one IoU matching; input detections must be score ordered."""
    if len(predictions) != len(truth) or not truth:
        raise ValueError("Expected paired detections and ground truth")
    tp = fp = fn = 0
    for predicted, targets in zip(predictions, truth):
        unmatched = set(range(len(targets)))
        for box in predicted:
            best = max(unmatched, key=lambda i: intersection_over_union(box, targets[i]), default=None)
            if best is not None and intersection_over_union(box, targets[best]) >= iou_threshold:
                unmatched.remove(best)
                tp += 1
            else:
                fp += 1
        fn += len(unmatched)
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(precision=precision, recall=recall, f1=2 * precision * recall / max(precision + recall, 1e-12),
                tp=tp, fp=fp, fn=fn, iou_threshold=iou_threshold)
