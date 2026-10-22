import cv2
import numpy as np
from scipy.special import expit
from .config import SegmentationConfig
from .imaging import preprocess
from .types import Character, SegmentationResult


def _runs(values: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.r_[False, values, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def _split_wide(binary, box, config):
    x, y, w, h = box
    ih, iw = binary.shape
    if h > config.max_height_ratio * ih or w > 0.9 * iw:
        return [box]  # never turn the plate frame into artificial glyphs
    if w / h <= config.split_width_ratio:
        return [box]
    # A joined pair can have no empty column. Cut only at a deep interior valley.
    projection = np.count_nonzero(binary[y:y + h, x:x + w], axis=0)
    lo, hi = max(1, int(w * 0.25)), min(w - 1, int(w * 0.75))
    if hi <= lo:
        return [box]
    cut = lo + int(np.argmin(projection[lo:hi]))
    if projection[cut] > 0.12 * h:
        return [box]
    return [(x, y, cut, h), (x + cut, y, w - cut, h)]


def _projection_boxes(binary):
    height, width = binary.shape
    horizontal = np.count_nonzero(binary, axis=1)
    # Join interruptions within strokes while retaining the gap between two rows.
    active = (horizontal > max(1, width * 0.015)).astype(np.uint8)
    active = cv2.morphologyEx(active[:, None], cv2.MORPH_CLOSE, np.ones((3, 1), np.uint8)).ravel() > 0
    boxes = []
    for y0, y1 in _runs(active):
        if y1 - y0 < height * 0.15:
            continue
        projection = np.count_nonzero(binary[y0:y1], axis=0)
        for x0, x1 in _runs(projection > max(1, (y1 - y0) * 0.04)):
            points = cv2.findNonZero(binary[y0:y1, x0:x1])
            if points is not None:
                x, y, w, h = cv2.boundingRect(points)
                boxes.append((x0 + x, y0 + y, w, h))
    return boxes


def _row_order(items, gap_ratio):
    if not items:
        return []
    median_height = float(np.median([c.box[3] for c in items]))
    ordered = sorted(items, key=lambda c: c.box[1] + c.box[3] / 2)
    centers = np.array([c.box[1] + c.box[3] / 2 for c in ordered])
    gaps = np.diff(centers)
    cut = int(np.argmax(gaps)) + 1 if len(gaps) and max(gaps) > gap_ratio * median_height else len(ordered)
    rows = [ordered[:cut], ordered[cut:]]
    result = []
    for row, chars in enumerate(rows):
        for char in sorted(chars, key=lambda c: c.box[0]):
            char.row = row
            result.append(char)
    return result


class CharacterSegmenter:
    """CCA+ = connected components, soft geometric gates and projection splitting.

    Sigmoid scores are hand-designed plausibility scores, not learned probabilities.
    Returns rejected components too, so threshold choices can be audited.
    """

    def __init__(self, config: SegmentationConfig = SegmentationConfig()):
        if config.method not in {"cca", "projection", "cca_plus"}:
            raise ValueError("Unknown segmentation method")
        if config.threshold not in {"auto", "otsu", "adaptive"}:
            raise ValueError("Unknown threshold")
        self.config = config

    def segment(self, image: np.ndarray) -> SegmentationResult:
        stages = preprocess(image)
        keys = ("otsu", "adaptive") if self.config.threshold == "auto" else (self.config.threshold,)
        results = [self._segment_binary(stages[key], key) for key in keys]

        def quality(result):
            chars = result.characters
            if not chars:
                return -100.0
            # Count is a soft preference only; never force a plate length.
            heights = np.array([c.box[3] for c in chars])
            return float(np.mean([c.score for c in chars]) - np.std(heights) / np.mean(heights)
                         - 0.12 * max(0, 6 - len(chars)) - 0.08 * max(0, len(chars) - 10))
        best = max(results, key=quality)
        best.diagnostics["threshold_comparison"] = {key: quality(result) for key, result in zip(keys, results)}
        return best

    def _segment_binary(self, binary, threshold):
        config = self.config
        height, width = binary.shape
        count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        boxes = [tuple(int(v) for v in stat[:4]) for stat in stats[1:]]
        if config.method == "projection":
            boxes = _projection_boxes(binary)
        elif config.method == "cca_plus":
            boxes = [part for box in boxes for part in _split_wide(binary, box, config)]
        records, chars = [], []
        for box in boxes:
            x, y, w, h = box
            ink = float(np.count_nonzero(binary[y:y + h, x:x + w]) / (w * h))
            hr, aspect = h / height, w / h
            # Conservative hard limits remove dots, screws, frame and implausible blobs.
            reasons = []
            if not config.min_height_ratio <= hr <= config.max_height_ratio:
                reasons.append("height")
            if not config.min_aspect <= aspect <= config.max_aspect:
                reasons.append("aspect")
            if ink < config.min_ink:
                reasons.append("ink")
            border = (x <= 1 or y <= 1 or x + w >= width - 1 or y + h >= height - 1) and (w > width * 0.5 or h > height * 0.78)
            if border:
                reasons.append("frame")
            score = float(expit(config.sigmoid_gain * (hr - 0.10))
                          * expit(config.sigmoid_gain * (1.85 - aspect))
                          * expit(config.sigmoid_gain * (ink - 0.02)))
            if config.method == "cca_plus" and score < config.min_score:
                reasons.append("soft_score")
            records.append(dict(box=box, height_ratio=hr, aspect=aspect, ink=ink,
                                score=score, accepted=not reasons, reasons=reasons))
            if not reasons:
                chars.append(Character(box, binary[y:y + h, x:x + w].copy(), 0, score))
        chars = _row_order(chars, config.row_gap_ratio)
        projections = []
        for row in sorted({c.row for c in chars}):
            row_chars = [c for c in chars if c.row == row]
            y0 = min(c.box[1] for c in row_chars)
            y1 = max(c.box[1] + c.box[3] for c in row_chars)
            projections.append(np.count_nonzero(binary[y0:y1], axis=0))
        return SegmentationResult(chars, binary, np.count_nonzero(binary, axis=1), projections, records,
                                  {"method": config.method, "threshold": threshold, "raw_components": count - 1})
