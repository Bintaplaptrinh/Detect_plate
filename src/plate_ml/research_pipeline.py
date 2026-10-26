"""Paper-inspired two-stage region HOG/SVM and row-wise plate segmentation.

This is an adaptation for white/yellow, one/two-row VNLP plates, not a replication
of the Algerian yellow, digits-only experiment. OCR glyph HOG stays unchanged.
"""
from dataclasses import dataclass, replace
from time import perf_counter

import cv2
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .evaluation import sequence_metrics
from .features import HOGExtractor
from .heuristics import DEFAULT_TEMPLATES, decode_plate
from .imaging import gray_u8, intersection_over_union, nms, read_image, rectify
from .types import Character, DetectionResult, PlateCandidate, SegmentationResult
from .vnlp import crop_scene
from .segmentation import CharacterSegmenter


@dataclass(frozen=True)
class RegionConfig:
    max_side: int = 640
    max_proposals: int = 220
    nms_iou: float = 0.35
    threshold: float = -0.5
    max_candidates: int = 12


def region_hog(crop):
    """128x64 region descriptor; separate from the unchanged OCR glyph HOG."""
    image = cv2.resize(gray_u8(crop), (128, 64))
    image = cv2.createCLAHE(2.0, (8, 8)).apply(image)
    descriptor = cv2.HOGDescriptor((128, 64), (16, 16), (8, 8), (8, 8), 9)
    return descriptor.compute(image).ravel()


class PanelProposer:
    """Color panels + vertical-edge bands at multiple scales, without GT input."""

    def __init__(self, config=RegionConfig()):
        self.config = config

    def propose(self, image):
        original_h, original_w = image.shape[:2]
        scale = min(1.0, self.config.max_side / max(original_h, original_w))
        small = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1 else image
        h, w = small.shape[:2]
        gray = gray_u8(small)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        masks = []
        for value in (80, 130, 180, 220):
            white = ((hsv[:, :, 1] < 115) & (hsv[:, :, 2] > value)).astype(np.uint8) * 255
            white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((3, 5), np.uint8))
            masks.append((f'white_{value}', white))
        yellow = cv2.inRange(hsv, (12, 60, 70), (42, 255, 255))
        masks.append(('yellow', cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, np.ones((5, 9), np.uint8))))
        contrast = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
        gradient = cv2.convertScaleAbs(cv2.Sobel(contrast, cv2.CV_16S, 1, 0, ksize=3))
        gradient = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        for kw, kh in ((9, 3), (17, 3), (27, 5), (41, 7), (13, 13), (23, 21)):
            mask = cv2.morphologyEx(gradient, cv2.MORPH_CLOSE, np.ones((kh, kw), np.uint8))
            masks.append((f'edges_{kw}_{kh}', mask))
        proposals = []
        for source, mask in masks:
            contours = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
            for contour in contours:
                area = cv2.contourArea(contour)
                if not 0.0007 * w * h <= area <= 0.45 * w * h:
                    continue
                (cx, cy), (rw, rh), angle = cv2.minAreaRect(contour)
                if min(rw, rh) < 13 or max(rw, rh) / max(1, min(rw, rh)) > 11:
                    continue
                if source.startswith('edges'):
                    # Text bands need a little surrounding plate background.
                    rw, rh = rw * 1.12, rh * 1.20
                else:
                    rw, rh = rw * 1.025, rh * 1.025
                quad = cv2.boxPoints(((cx, cy), (rw, rh), angle))
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
                if not source.startswith('edges') and len(approx) == 4 and cv2.isContourConvex(approx):
                    quad = approx.reshape(4, 2).astype(np.float32)
                quad = np.clip(quad / scale, (0, 0), (original_w - 1, original_h - 1)).astype(np.float32)
                x, y, bw, bh = cv2.boundingRect(quad)
                bw, bh = min(bw, original_w - x), min(bh, original_h - y)
                if not 0.4 <= bw / max(1, bh) <= 12:
                    continue
                fill = min(1.0, area / max(rw * rh, 1))
                score = fill + (0.15 if source.startswith(('white', 'yellow')) else 0)
                proposals.append(PlateCandidate((x, y, bw, bh), None, float(score), source, quad))
        # Retain diverse scales before the learned verifier, rather than only a few
        # highest-scoring morphology regions as in the old proposal pipeline.
        kept = nms(proposals, 0.85)[:self.config.max_proposals]
        for p in kept:
            p.crop = rectify(image, p.quad)
        return kept


class TwoStageDetector:
    def __init__(self, verifier, config=RegionConfig()):
        self.verifier, self.config = verifier, config
        self.proposer = PanelProposer(config)

    def detect(self, image):
        proposals = self.proposer.propose(image)
        if proposals:
            scores = self.verifier.decision_function(np.asarray([region_hog(p.crop) for p in proposals]))
            for p, score in zip(proposals, scores):
                p.score = float(score)
        accepted = [p for p in proposals if p.score >= self.config.threshold]
        return DetectionResult(nms(accepted, self.config.nms_iou)[:self.config.max_candidates], {},
                               dict(raw_proposals=len(proposals), score_kind='linear_svm_margin'))


def train_region_verifier(scenes, seed=42, progress=None):
    """Actual proposal negatives and jittered positives, from train scenes only."""
    rng = np.random.default_rng(seed)
    proposer = PanelProposer()
    features, labels, audit = [], [], []
    for index, scene in enumerate(scenes):
        image = read_image(scene.path)
        ih, iw = image.shape[:2]
        x, y, w, h = scene.box
        positives = [image[y:y+h, x:x+w]]
        for _ in range(2):
            dx, dy = rng.uniform(-0.05, 0.05, 2)
            dw, dh = rng.uniform(0.95, 1.12, 2)
            x0, y0 = max(0, round(x+dx*w)), max(0, round(y+dy*h))
            x1, y1 = min(iw, round(x0+w*dw)), min(ih, round(y0+h*dh))
            positives.append(image[y0:y1, x0:x1])
        proposals = proposer.propose(image)
        overlaps = [intersection_over_union(p.box, scene.box) for p in proposals]
        positives.extend(p.crop for p, overlap in zip(proposals, overlaps) if overlap >= 0.65)
        negative_ids = [i for i, overlap in enumerate(overlaps) if overlap < 0.15]
        selected_ids = rng.choice(negative_ids, min(8, len(negative_ids)), replace=False).tolist()
        negatives = [proposals[i].crop for i in selected_ids]
        # Additional easy background examples make the verifier usable even when
        # a training image has very few proposals.
        for _ in range(30):
            nw, nh = max(20, min(iw, w)), max(20, min(ih, h))
            nx, ny = int(rng.integers(iw-nw+1)), int(rng.integers(ih-nh+1))
            if intersection_over_union((nx, ny, nw, nh), scene.box) < 0.05:
                negatives.append(image[ny:ny+nh, nx:nx+nw])
                break
        for label, crops in ((1, positives), (0, negatives)):
            features.extend(region_hog(crop) for crop in crops)
            labels.extend([label] * len(crops))
        audit.append(dict(path=scene.path, group=scene.group, positives=len(positives),
                          negatives=len(negatives), proposals=len(proposals),
                          proposal_max_iou=float(max(overlaps, default=0))))
        if progress and (index + 1) % 50 == 0:
            progress(f'region training crops {index+1}/{len(scenes)}')
    X, y = np.asarray(features, np.float32), np.asarray(labels)
    verifier = make_pipeline(StandardScaler(), LinearSVC(C=0.01, class_weight='balanced',
                                                        random_state=seed, max_iter=5000, dual='auto'))
    verifier.fit(X, y)
    # Hard-negative mining over the entire generated training feature pool.
    # These are actual non-plate proposals, not heldout backgrounds.
    margin = verifier.decision_function(X)
    hard = np.flatnonzero((y == 0) & (margin > -0.5))
    if len(hard):
        weights = np.ones(len(y))
        weights[hard] = 3.0
        verifier.fit(X, y, linearsvc__sample_weight=weights)
    return verifier, dict(scenes=audit, samples=len(y), positives=int(y.sum()),
                          negatives=int((y == 0).sum()), hard_negatives=len(hard),
                          descriptor_size=X.shape[1], C=0.01)


def _runs(active):
    edges = np.diff(np.r_[False, active, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _remove_frame(binary):
    """Remove only long lines near plate boundaries, preserving interior strokes."""
    h, w = binary.shape
    result = binary.copy()
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((1, max(15, round(w*.55))), np.uint8))
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((max(20, round(h*.70)), 1), np.uint8))
    mask = np.zeros_like(binary)
    band_y, band_x = max(2, round(h*.12)), max(2, round(w*.06))
    mask[:band_y] |= horizontal[:band_y]
    mask[-band_y:] |= horizontal[-band_y:]
    mask[:, :band_x] |= vertical[:, :band_x]
    mask[:, -band_x:] |= vertical[:, -band_x:]
    result[mask > 0] = 0
    n, connected, stats, _ = cv2.connectedComponentsWithStats(result)
    for i, (x, y, bw, bh, area) in enumerate(stats[1:], 1):
        touches = x == 0 or y == 0 or x+bw == w or y+bh == h
        if touches and (bw > .85*w or bh > .95*h) and area < .30*w*h:
            result[connected == i] = 0
    return result


def _rotation(image, angle, border):
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
    nw, nh = int(np.ceil(h*sine+w*cosine)), int(np.ceil(h*cosine+w*sine))
    matrix[0, 2] += nw/2-w/2
    matrix[1, 2] += nh/2-h/2
    return cv2.warpAffine(image, matrix, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=border)


def _deskew(gray):
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    binary = _remove_frame(binary)
    h, w = binary.shape
    def score(angle):
        rotated = _rotation(binary, angle, 0)
        projection = np.count_nonzero(rotated, axis=1).astype(float)
        return float(np.sum(projection**2) / max(1, projection.sum()))
    zero = score(0)
    angles = list(range(-24, 25, 4))
    angle = max(angles, key=lambda a: score(a) * (1 - .0005*abs(a)))
    if score(angle) < zero*1.035:
        angle = 0
    return (_rotation(gray, angle, 255) if angle else gray), angle


def _row_boxes(binary, y0, y1, row):
    """Projection intervals, with CCA cleanup for dots and disconnected strokes."""
    roi = binary[y0:y1]
    rh, rw = roi.shape
    if rh < 8:
        return []
    count, components, stats, _ = cv2.connectedComponentsWithStats(roi)
    clean = np.zeros_like(roi)
    heights = [h for x, y, w, h, a in stats[1:] if .15*rh <= h <= .98*rh and .08 <= w/h <= 1.3]
    reference = float(np.median(heights)) if heights else .75*rh
    for i, (x, y, w, h, area) in enumerate(stats[1:], 1):
        if h >= .45*reference and .07 <= w/max(1, h) <= 2.2 and area >= 5:
            if w > .75*rw or h > .99*rh:
                continue
            clean[components == i] = 255
    projection = np.count_nonzero(clean, axis=0)
    active = projection > max(1, .025*reference)
    # Close only one-pixel holes, avoiding arbitrary forced character counts.
    active = cv2.morphologyEx(active.astype(np.uint8)[None], cv2.MORPH_CLOSE,
                             np.ones((1, 2), np.uint8)).ravel() > 0
    intervals = _runs(active)
    boxes = []
    for left, right in intervals:
        width = right-left
        if width > 1.1*reference:
            # Split a joined blob only at a low-ink interior valley.
            lo, hi = left+int(.25*width), right-int(.25*width)
            cut = lo+int(np.argmin(projection[lo:hi])) if hi > lo else left
            parts = [(left, cut), (cut, right)] if cut > left and projection[cut] < .20*reference else [(left, right)]
        else:
            parts = [(left, right)]
        for left, right in parts:
            points = cv2.findNonZero(clean[:, left:right])
            if points is None:
                continue
            x, y, w, h = cv2.boundingRect(points)
            if h < .55*reference or not .07 <= w/h <= 1.05:
                continue
            box = int(left+x), int(y0+y), int(w), int(h)
            boxes.append(Character(box, clean[y:y+h, left+x:left+x+w].copy(), row, 1.0))
    return boxes


class RowProjectionSegmenter:
    """Illumination correction, whole-plate deskew, border removal, row projection."""

    def __init__(self, deskew=True):
        self.deskew = deskew

    def hypotheses(self, image):
        gray = gray_u8(image)
        scale = min(4.0, max(1.0, 128/gray.shape[0]))
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray, angle = _deskew(gray) if self.deskew else (gray, 0)
        h, w = gray.shape
        background = cv2.GaussianBlur(gray, (0, 0), max(3, min(h, w)/9))
        corrected = cv2.divide(gray, background, scale=220)
        corrected = cv2.GaussianBlur(corrected, (3, 3), 0)
        otsu = cv2.threshold(corrected, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        adaptive = cv2.adaptiveThreshold(corrected, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 31, 9)
        results = []
        for name, binary in [('illumination_otsu', otsu), ('illumination_adaptive', adaptive)]:
            binary = _remove_frame(binary)
            projection = np.count_nonzero(binary, axis=1).astype(float)
            # Evaluate one and two rows; no dataset layout or label is supplied.
            lo, hi = int(.30*h), max(int(.30*h)+1, int(.70*h))
            smooth = np.convolve(projection, np.ones(5)/5, mode='same')
            split = lo+int(np.argmin(smooth[lo:hi]))
            for bands in [[(0, h)], [(0, split), (split, h)]]:
                chars = []
                for row, (y0, y1) in enumerate(bands):
                    chars.extend(_row_boxes(binary, y0, y1, row))
                if not chars or any(not any(c.row == r for c in chars) for r in range(len(bands))):
                    continue
                heights = np.array([c.box[3] for c in chars])
                aspects = np.array([c.box[2]/c.box[3] for c in chars])
                geometry = float(1 - np.std(heights)/max(1, heights.mean())
                                 - .3*abs(np.median(aspects)-.55)
                                 - .15*min(abs(len(chars)-n) for n in (7, 8, 9)))
                results.append(SegmentationResult(chars, binary, projection, [], [],
                    dict(method='row_projection', threshold=name, angle=angle,
                         geometry=geometry, rows=len(bands))))
        if not results:
            results.append(SegmentationResult([], otsu, np.count_nonzero(otsu, axis=1), [], [],
                           dict(method='row_projection', angle=angle, geometry=-10, rows=0)))
        return results

    def segment(self, image):
        return max(self.hypotheses(image), key=lambda r: r.diagnostics['geometry'])


class HybridSegmenter:
    """Compare legacy CCA thresholds and paper-inspired row hypotheses by OCR.

    Ground-truth count/layout/text are never provided to this inference stage.
    """
    def __init__(self, legacy_config):
        self.legacy_config = legacy_config
        self.projection = RowProjectionSegmenter()

    def hypotheses(self, image):
        legacy = [CharacterSegmenter(replace(self.legacy_config, threshold=threshold)).segment(image)
                  for threshold in ('otsu', 'adaptive')]
        return legacy + self.projection.hypotheses(image)


class ResearchRecognizer:
    def __init__(self, ocr_model, detector, segmenter, hog, templates=DEFAULT_TEMPLATES,
                 rerank_weight=0.5, top_k=5, decoder=None, profile=False):
        self.model, self.detector, self.segmenter = ocr_model, detector, segmenter
        self.extractor, self.templates = HOGExtractor(hog), templates
        self.rerank_weight, self.top_k = rerank_weight, top_k
        self.decoder = decoder or decode_plate
        self.profile = profile

    def recognize_crop(self, crop):
        start = perf_counter() if self.profile else 0
        hypotheses = self.segmenter.hypotheses(crop) if hasattr(self.segmenter, 'hypotheses') else [self.segmenter.segment(crop)]
        stages = dict(segmentation=(perf_counter()-start)*1000, hog=0., classifier=0., decoder=0.) if self.profile else {}
        results = []
        for segmentation in hypotheses:
            chars = segmentation.characters
            start = perf_counter() if self.profile else 0
            X = self.extractor.transform(c.image for c in chars)
            if self.profile:
                stages['hog'] += (perf_counter()-start)*1000
                start = perf_counter()
            p = self.model.predict_proba(X) if len(X) else np.empty((0, len(self.model.classes_)))
            if self.profile:
                stages['classifier'] += (perf_counter()-start)*1000
                start = perf_counter()
            rows = tuple(sum(c.row == r for c in chars) for r in sorted({c.row for c in chars}))
            decoded = self.decoder(p, self.model.classes_, rows, self.templates)
            if self.profile:
                stages['decoder'] += (perf_counter()-start)*1000
            if len(p):
                logp = np.log(np.maximum(p.max(axis=1), 1e-9))
                heights = np.array([c.box[3] for c in chars], dtype=float)
                aspects = np.array([c.box[2]/c.box[3] for c in chars])
                geometry = float(1 - heights.std()/max(1, heights.mean())
                                 - .3*abs(np.median(aspects)-.55))
                quality = float(logp.mean() + .25*logp.min()
                    + .25*geometry
                    + .15*(decoded['template'] is not None)
                    - .20*min(abs(len(chars)-n) for n in (7, 8, 9)))
            else:
                quality = -10.0
            results.append(dict(segmentation=segmentation, probabilities=p, decoded=decoded, quality=quality))
        best = max(results, key=lambda r: r['quality'])
        if self.profile:
            best['stage_ms'] = stages
            best['hypotheses'] = len(hypotheses)
        return best

    def recognize(self, image):
        start = perf_counter() if self.profile else 0
        detected = self.detector.detect(image)
        stages = dict(detector=(perf_counter()-start)*1000) if self.profile else {}
        predictions = []
        for candidate in detected.candidates[:self.top_k]:
            result = self.recognize_crop(candidate.crop)
            if self.profile:
                for key, value in result['stage_ms'].items():
                    stages[key] = stages.get(key, 0.) + value
            result['candidate'] = candidate
            result['score'] = candidate.score + self.rerank_weight*result['quality']
            predictions.append(result)
        if self.profile:
            detected.diagnostics['stage_ms'] = stages
        return detected, sorted(predictions, key=lambda r: r['score'], reverse=True)


def sequence_report(rows):
    return {column: sequence_metrics([r['truth'] for r in rows], [r[column] for r in rows])
            for column in ('oracle_raw', 'oracle_decoded', 'pipeline_raw', 'pipeline_decoded')}
