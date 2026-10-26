"""Local app inference using the frozen research models, with bounded multi-plate search.

The multi-image adapter is deliberately separate from historical experiment code.
It never trains, reads plate labels from filenames, or modifies the saved release.
"""
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import re
from time import perf_counter

import cv2
import joblib
import numpy as np

from .imaging import intersection_over_union
from .inference import load_recognizer
from .research_pipeline import ResearchRecognizer, TwoStageDetector
from .types import PlateCandidate


ALGORITHMS = ('svm', 'pca_fld_hmm')
MAX_PIXELS = 16_000_000
PLATE_PATTERN = re.compile(r'^[0-9]{2}[A-Z]{1,2}[0-9]{4,6}$')


def tile_origins(length, tile=960, overlap=.35):
    """Cover the entire axis, including the far edge, without duplicate tiles."""
    if length <= tile:
        return [0]
    step = max(1, round(tile * (1-overlap)))
    return sorted(set([*range(0, length-tile+1, step), length-tile]))


def same_region(a, b, iou=.45):
    """Merge duplicate boxes by overlap, retaining separate nearby plates."""
    if intersection_over_union(a, b) >= iou:
        return True
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    intersection = max(0, min(ax+aw, bx+bw)-max(ax, bx)) * max(0, min(ay+ah, by+bh)-max(ay, by))
    small, large = sorted((aw*ah, bw*bh))
    return small > 0 and small / max(large, 1) >= .35 and intersection / small >= .85


def group_candidates(candidates, limit=32):
    groups = []
    for candidate in sorted(candidates, key=lambda c: c.score, reverse=True):
        group = next((g for g in groups if same_region(candidate.box, g[0].box)), None)
        if group is not None:
            group.append(candidate)
        elif len(groups) < limit:
            groups.append([candidate])
    return groups


def observation_confidence(result, classes):
    p = result['probabilities']
    text = result['decoded']['text']
    if not len(p) or len(text) != len(p):
        return 0.0
    lookup = {label: i for i, label in enumerate(classes)}
    values = [p[i, lookup[c]] if c in lookup else 0 for i, c in enumerate(text)]
    return float(np.exp(np.log(np.maximum(values, 1e-12)).mean()))


def acceptable_multi(result, confidence):
    text = result['decoded']['text']
    return bool(7 <= len(text) <= 10 and PLATE_PATTERN.fullmatch(text)
                and result['quality'] >= -.85 and confidence >= .45)


def select_distinct(results, limit=24):
    kept = []
    for result in sorted(results, key=lambda r: r['rank_score'], reverse=True):
        if all(not same_region(result['box'], previous['box'], iou=.30) for previous in kept):
            kept.append(result)
        if len(kept) == limit:
            break
    # Number by image position, not by recognition score.
    return sorted(kept, key=lambda r: (r['box'][1], r['box'][0]))


def annotate(image, detections):
    """Draw readable labels inside the canvas, even at image edges."""
    output = image.copy()
    height, width = output.shape[:2]
    scale = min(1.1, max(.52, max(height, width) / 2000 * .80))
    thickness = max(1, round(scale * 2))
    pad = max(4, round(scale * 6))
    for detection in detections:
        x, y, w, h = detection['box']
        color = (42, 115, 20)
        cv2.rectangle(output, (x, y), (min(width-1, x+w), min(height-1, y+h)), color, thickness+1)
        label = f"{detection['id']}. {detection['text']}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        # Long labels on tiny images are scaled down rather than clipped.
        label_scale = scale * min(1.0, (width-2*pad) / max(tw, 1))
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, label_scale, thickness)
        label_h = th + baseline + 2*pad
        left = max(0, min(x, width-tw-2*pad))
        top = y-label_h if y >= label_h else min(height-label_h, y+h)
        top = max(0, top)
        cv2.rectangle(output, (left, top), (min(width-1, left+tw+2*pad), min(height-1, top+label_h)), color, -1)
        cv2.putText(output, label, (left+pad, top+pad+th), cv2.FONT_HERSHEY_SIMPLEX,
                    label_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return output


class DemoEngine:
    def __init__(self, root):
        self.root = Path(root)
        svm = load_recognizer(self.root / 'Registration')
        candidate = self.root / 'reports/pca-fld-hmm-no-svm-180-20261026/no_svm_candidate.joblib'
        if not candidate.is_file():
            raise ValueError('Thiếu mô hình PCA + FLD + HMM đã thực nghiệm. Xem hướng dẫn trong app/README.md.')
        # Only the trusted local artifact matching the frozen protocol is loaded.
        protocol = json.loads(candidate.with_name('protocol_before_test.json').read_text(encoding='utf-8'))
        if sha256(candidate.read_bytes()).hexdigest() != protocol['candidate_sha256']:
            raise ValueError('Dấu kiểm tra mô hình PCA + FLD + HMM không khớp bản thực nghiệm.')
        saved = joblib.load(candidate)
        if not np.array_equal(svm.model.classes_, saved['model'].classes_):
            raise ValueError('Các mô hình không có cùng bảng ký tự.')
        gaussian = ResearchRecognizer(saved['model'], svm.detector, svm.segmenter,
            svm.extractor.config, svm.templates, svm.rerank_weight, svm.top_k, decoder=saved['hmm'])
        self.recognizers = dict(svm=svm, pca_fld_hmm=gaussian)
        self.multi_detector = TwoStageDetector(svm.detector.verifier,
            replace(svm.detector.config, max_side=1600, max_proposals=420,
                    max_candidates=32, threshold=1.0, nms_iou=.50))
        self.tile_detector = TwoStageDetector(svm.detector.verifier,
            replace(svm.detector.config, max_side=1000, max_proposals=300,
                    max_candidates=24, threshold=1.0, nms_iou=.50))

    def multi_candidates(self, image):
        h, w = image.shape[:2]
        candidates = []
        views = 1
        detected = self.multi_detector.detect(image)
        proposals = detected.diagnostics['raw_proposals']
        for c in detected.candidates:
            c.source = 'full/' + c.source
            candidates.append(c)
        if max(h, w) > 1050:
            for oy in tile_origins(h):
                for ox in tile_origins(w):
                    # The full image was already evaluated above.
                    tile = image[oy:min(h, oy+960), ox:min(w, ox+960)]
                    if tile.shape[:2] == image.shape[:2]:
                        continue
                    detected = self.tile_detector.detect(tile)
                    views += 1
                    proposals += detected.diagnostics['raw_proposals']
                    for c in detected.candidates:
                        x, y, bw, bh = c.box
                        quad = c.quad + np.float32([ox, oy]) if c.quad is not None else None
                        candidates.append(PlateCandidate((x+ox, y+oy, bw, bh), c.crop,
                            c.score, f'tile/{ox}/{oy}/{c.source}', quad))
        candidates = [c for c in candidates if .65 <= c.box[2]/max(c.box[3], 1) <= 6.5
                      and c.box[2]*c.box[3] <= .18*w*h]
        return group_candidates(candidates), dict(views=views, raw_proposals=proposals)

    def recognize_multi(self, image, recognizer):
        groups, diagnostics = self.multi_candidates(image)
        results, calls = [], 0
        for group in groups:
            if calls >= 96:
                break
            # Keep both full-image and tiled geometry. A larger verifier margin
            # alone must not discard a tighter crop with much better characters.
            full = [c for c in group if c.source.startswith('full/')]
            tiled = [c for c in group if c.source.startswith('tile/')]
            variants = []
            for pool in (full, tiled):
                if pool:
                    variants.append(pool[0])
            for c in group:
                if len(variants) == 3:
                    break
                if all(c is not previous for previous in variants):
                    variants.append(c)
            best = None
            for c in variants:
                if calls >= 96:
                    break
                result = recognizer.recognize_crop(c.crop)
                calls += 1
                confidence = observation_confidence(result, recognizer.model.classes_)
                entry = self._entry(c, result, confidence)
                if acceptable_multi(result, confidence) and (best is None or entry['rank_score'] > best['rank_score']):
                    best = entry
            if best is None and group[0].score >= 1.0 and calls < 94:
                # Axis-aligned alternatives prevent an unstable quadrilateral
                # from cutting a character. These generic crops do not use GT.
                c = group[0]
                x, y, w, h = c.box
                for inset in (0.0, .05):
                    dx, dy = round(w*inset), round(h*inset)
                    box = x+dx, y+dy, w-2*dx, h-2*dy
                    crop = image[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
                    if crop.size == 0:
                        continue
                    alternative = PlateCandidate(box, crop, c.score, c.source+'/axis')
                    result = recognizer.recognize_crop(crop)
                    calls += 1
                    confidence = observation_confidence(result, recognizer.model.classes_)
                    entry = self._entry(alternative, result, confidence)
                    if acceptable_multi(result, confidence) and (best is None or entry['rank_score'] > best['rank_score']):
                        best = entry
            if best:
                results.append(best)
        diagnostics.update(ocr_calls=calls, region_groups=len(groups))
        return select_distinct(results), diagnostics

    @staticmethod
    def _entry(candidate, result, confidence):
        return dict(box=[int(v) for v in candidate.box], text=result['decoded']['text'],
            raw=result['decoded']['raw'], observation_confidence=round(confidence, 5),
            region_score=round(float(candidate.score), 5),
            quality=round(float(result['quality']), 5),
            rank_score=float(candidate.score+2*result['quality']), source=candidate.source)

    def recognize(self, image, algorithm='svm', multi_detection=False):
        if algorithm not in ALGORITHMS:
            raise ValueError('Thuật toán không hợp lệ.')
        if type(multi_detection) is not bool:
            raise ValueError('Chế độ nhiều biển số phải là giá trị bật hoặc tắt.')
        if image is None or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError('Ảnh đầu vào không hợp lệ.')
        h, w = image.shape[:2]
        if min(h, w) < 16 or h*w > MAX_PIXELS:
            raise ValueError('Ảnh phải có mỗi chiều ít nhất 16 pixel và không vượt quá 16 triệu pixel.')
        start = perf_counter()
        recognizer = self.recognizers[algorithm]
        if multi_detection:
            detections, diagnostics = self.recognize_multi(image, recognizer)
        else:
            detected, predictions = recognizer.recognize(image)
            detections = []
            if predictions and predictions[0]['decoded']['text']:
                result = predictions[0]
                confidence = observation_confidence(result, recognizer.model.classes_)
                detections = [self._entry(result['candidate'], result, confidence)]
            diagnostics = dict(views=1, raw_proposals=detected.diagnostics.get('raw_proposals', 0),
                ocr_calls=len(predictions), region_groups=len(detected.candidates))
        for i, detection in enumerate(detections, 1):
            detection['id'] = i
            detection.pop('rank_score', None)
        annotated = annotate(image, detections)
        seconds = perf_counter()-start
        return dict(algorithm=algorithm, multi_detection=multi_detection,
                    processing_seconds=round(seconds, 6), image_width=w, image_height=h,
                    detections=detections, diagnostics=diagnostics), annotated
