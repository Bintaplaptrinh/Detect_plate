"""VNLP OCR data ablation using notebook 01's unchanged image pipeline.

Filename text supervises an ordered segmentation, not individual annotated boxes.
Even count/row agreement is only a weak-label filter; retain an audit trail.
"""
from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from .config import HOGConfig, SegmentationConfig
from .evaluation import sequence_metrics
from .features import HOGExtractor, normalize_glyph
from .heuristics import DEFAULT_TEMPLATES, decode_plate
from .imaging import read_image
from .segmentation import CharacterSegmenter
from .vnlp import crop_scene


@dataclass
class GlyphDataset:
    X: np.ndarray
    y: np.ndarray
    records: list[dict]
    plates: list[dict]

    def summary(self):
        accepted = sum(p['accepted'] for p in self.plates)
        return dict(plates=len(self.plates), accepted_plates=accepted,
                    acceptance_rate=accepted / max(1, len(self.plates)), glyphs=len(self.y),
                    classes=dict(sorted(Counter(self.y.tolist()).items())),
                    rejected_reasons=dict(Counter(p['reason'] for p in self.plates if not p['accepted'])),
                    label_status='weak_filename_alignment_not_manual_glyph_annotations')


def assert_group_disjoint(partitions):
    """Check before cutting any scene into character samples."""
    seen = {}
    for partition, scenes in partitions.items():
        for scene in scenes:
            for key in (('identity', scene.group), ('text', scene.text)):
                if key in seen and seen[key] != partition:
                    raise ValueError(f'Plate identity leakage: {key[1]}')
                seen[key] = partition


def remove_scene_duplicates(partitions):
    """Prefer holdouts to train for exact decoded-image duplicates.

    A validation/test collision is an invalid evaluation, so fail explicitly.
    Images remain on disk; only the experimental selection is deduplicated.
    """
    assert_group_disjoint(partitions)
    selected, seen, removed = {}, {}, []
    for partition in ('validation', 'test', 'train'):
        selected[partition] = []
        for scene in partitions[partition]:
            image = read_image(scene.path)
            digest = sha256(str(image.shape).encode() + image.tobytes()).hexdigest()
            if digest in seen:
                previous_partition, previous_text = seen[digest]
                if previous_text != scene.text:
                    raise ValueError(f'Identical scene has conflicting text: {scene.path}')
                if partition == 'test' and previous_partition == 'validation':
                    raise ValueError('Duplicate image across validation/test')
                removed.append(dict(path=scene.path, partition=partition, kept_in=previous_partition))
                continue
            seen[digest] = partition, scene.text
            selected[partition].append(scene)
    return selected, removed


def extract_vnlp_glyphs(scenes, segmentation=SegmentationConfig(), hog=HOGConfig(),
                        templates=DEFAULT_TEMPLATES):
    """No forced count, no truncating zip, no OCR-predicted training labels.

    Use exactly the inference segmenter + extractor. Reject unknown row layouts,
    ambiguous ordering, blank crops, and any count mismatch as whole plates.
    """
    segmenter, extractor = CharacterSegmenter(segmentation), HOGExtractor(hog)
    images, labels, records, plates = [], [], [], []
    for scene in scenes:
        result = segmenter.segment(crop_scene(scene))
        chars = result.characters
        rows = tuple(sum(c.row == r for c in chars) for r in sorted({c.row for c in chars}))
        reason = None
        if not scene.text or any(c not in '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ' for c in scene.text):
            reason = 'unsupported_text'
        elif len(chars) != len(scene.text):
            reason = 'count_mismatch'
        elif scene.layout == 'one_row':
            if rows != (len(scene.text),):
                reason = 'row_mismatch'
        else:
            valid_rows = {t.rows for t in templates if len(t.rows) == 2
                          and len(t.positions) == len(scene.text)
                          and all(c in allowed for c, allowed in zip(scene.text, t.positions))}
            if rows not in valid_rows:
                reason = 'row_or_format_mismatch'
        if reason is None:
            if any(a.row > b.row or (a.row == b.row and a.box[0] >= b.box[0])
                   for a, b in zip(chars, chars[1:])):
                reason = 'reading_order'
            elif any(not normalize_glyph(c.image, hog).any() for c in chars):
                reason = 'blank_glyph'
        plates.append(dict(path=scene.path, text=scene.text, group=scene.group, layout=scene.layout,
                           expected=len(scene.text), found=len(chars), rows=list(rows),
                           accepted=reason is None, reason=reason,
                           threshold=result.diagnostics['threshold']))
        if reason is not None:
            continue
        for position, (char, label) in enumerate(zip(chars, scene.text, strict=True)):
            normalized = normalize_glyph(char.image, hog)
            images.append(char.image)
            labels.append(label)
            records.append(dict(path=scene.path, group=scene.group, text=scene.text,
                                layout=scene.layout, position=position, label=label,
                                row=char.row, box=list(char.box),
                                normalized_sha256=sha256(normalized.tobytes()).hexdigest()))
    return GlyphDataset(extractor.transform(images), np.asarray(labels, dtype=str), records, plates)


def glyph_metrics(model, dataset):
    """Observed target classes include labels absent from a VNLP-only model."""
    if not len(dataset.y):
        return dict(samples=0, accuracy=None, macro_f1=None, unseen_truth_classes=[])
    prediction = model.classes_[model.predict_proba(dataset.X).argmax(axis=1)]
    labels = sorted(set(dataset.y))
    return dict(samples=len(dataset.y), accuracy=float(accuracy_score(dataset.y, prediction)),
                macro_f1=float(f1_score(dataset.y, prediction, labels=labels, average='macro', zero_division=0)),
                unseen_truth_classes=sorted(set(dataset.y) - set(model.classes_)))


def cache_scene_features(scenes, recognizer):
    """Freeze detector top-1 + segmentation for paired comparisons of OCR models."""
    cached = []
    for scene in scenes:
        detection = recognizer.detector.detect(read_image(scene.path))
        top = detection.candidates[0] if detection.candidates else None
        item = dict(path=scene.path, truth=scene.text, layout=scene.layout,
                    candidates=len(detection.candidates), top_box=list(top.box) if top else None)
        for name, crop in [('oracle', crop_scene(scene)), ('pipeline', top.crop if top else None)]:
            chars = recognizer.segmenter.segment(crop).characters if crop is not None else []
            item[name] = dict(X=recognizer.extractor.transform(c.image for c in chars),
                              rows=tuple(sum(c.row == r for c in chars) for r in sorted({c.row for c in chars})))
        cached.append(item)
    return cached


def evaluate_cached(model, cached, templates=DEFAULT_TEMPLATES):
    predictions = []
    for item in cached:
        row = {k: item[k] for k in ('path', 'truth', 'layout', 'candidates', 'top_box')}
        for name in ('oracle', 'pipeline'):
            X = item[name]['X']
            probabilities = model.predict_proba(X) if len(X) else np.empty((0, len(model.classes_)))
            decoded = decode_plate(probabilities, model.classes_, item[name]['rows'], templates)
            row[name + '_raw'], row[name + '_decoded'] = decoded['raw'], decoded['text']
        predictions.append(row)
    metrics = {key: sequence_metrics([r['truth'] for r in predictions], [r[key] for r in predictions])
               for key in ('oracle_raw', 'oracle_decoded', 'pipeline_raw', 'pipeline_decoded')}
    return metrics, predictions


def check_baseline_pipeline(recognizer):
    """Fail closed if this stops being notebook 01's canonical 24x40/HOG path."""
    if asdict(recognizer.extractor.config) != asdict(HOGConfig()):
        raise ValueError('Expected notebook 01 deskew + 24x40 glyph on 32x48 canvas + HOG')
