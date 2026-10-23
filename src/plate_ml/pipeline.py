from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
import numpy as np
from .config import DetectionConfig, HOGConfig, SegmentationConfig
from .detection import PlateDetector
from .features import HOGExtractor
from .heuristics import decode_plate, DEFAULT_TEMPLATES
from .segmentation import CharacterSegmenter


@dataclass
class Recognition:
    candidate: object
    segmentation: object
    probabilities: np.ndarray
    decoded: dict
    confidence: float


class PlateRecognizer:
    def __init__(self, ocr_model, detection=DetectionConfig(), segmentation=SegmentationConfig(),
                 hog=HOGConfig(), verifier=None, templates=DEFAULT_TEMPLATES):
        self.model = ocr_model
        self.detector = PlateDetector(detection, verifier)
        self.segmenter = CharacterSegmenter(segmentation)
        self.extractor = HOGExtractor(hog)
        self.templates = templates

    def to_bundle(self):
        cascade = self.detector.config.cascade_path
        return dict(ocr_model=self.model, verifier=self.detector.verifier,
                    detection=replace(self.detector.config, cascade_path=None),
                    segmentation=self.segmenter.config, hog=self.extractor.config,
                    templates=self.templates, cascade_xml=Path(cascade).read_bytes() if cascade else None)

    @classmethod
    def from_bundle(cls, bundle, cascade_directory: str | Path):
        config = bundle["detection"]
        if bundle.get("cascade_xml"):
            data = bundle["cascade_xml"]
            directory = Path(cascade_directory).resolve()
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{sha256(data).hexdigest()}.xml"
            if not path.is_file() or path.read_bytes() != data:
                path.write_bytes(data)
            config = replace(config, cascade_path=str(path))
        return cls(bundle["ocr_model"], config, bundle["segmentation"], bundle["hog"],
                   bundle["verifier"], bundle["templates"])

    def recognize_crop(self, crop, candidate=None):
        segmented = self.segmenter.segment(crop)
        if segmented.characters:
            X = self.extractor.transform([char.image for char in segmented.characters])
            probabilities = self.model.predict_proba(X)
            rows = tuple(sum(c.row == row for c in segmented.characters) for row in sorted({c.row for c in segmented.characters}))
            decoded = decode_plate(probabilities, self.model.classes_, rows, self.templates)
            confidence = float(probabilities.max(axis=1).mean())
        else:
            probabilities = np.empty((0, len(self.model.classes_)))
            decoded = decode_plate(probabilities, self.model.classes_, (), self.templates)
            confidence = 0.0
        return Recognition(candidate, segmented, probabilities, decoded, confidence)

    def recognize(self, image):
        detection = self.detector.detect(image)
        # Retain detector score order; never select against known plate text.
        results = [self.recognize_crop(candidate.crop, candidate) for candidate in detection.candidates]
        return detection, results
