from pathlib import Path
import cv2
import numpy as np
from .config import DetectionConfig
from .features import plate_features
from .imaging import gray_u8, nms, preprocess, rectify
from .types import DetectionResult, PlateCandidate


class PlateDetector:
    """Multi-scale Haar and morphology proposals, optional learned HOG verifier.

    Scores without a verifier are ranking heuristics, not probabilities.
    Haar models must be supplied explicitly; no unrelated cascade is selected.
    """

    def __init__(self, config: DetectionConfig = DetectionConfig(), verifier=None):
        self.config = config
        self.verifier = verifier
        self.cascade = None
        if config.cascade_path:
            path = Path(config.cascade_path)
            if not path.is_file():
                raise FileNotFoundError(path)
            self.cascade = cv2.CascadeClassifier(str(path))
            if self.cascade.empty():
                raise ValueError(f"Invalid Haar cascade: {path}")

    def detect(self, image: np.ndarray) -> DetectionResult:
        stages = preprocess(image)
        gray = stages["contrast"]
        height, width = gray.shape
        proposals = []
        if self.cascade is not None and "haar" in self.config.proposal_sources:
            for x, y, w, h in self.cascade.detectMultiScale(
                gray, scaleFactor=self.config.cascade_scale,
                minNeighbors=self.config.cascade_neighbors, minSize=(32, 16)
            ):
                proposals.append(((int(x), int(y), int(w), int(h)), None, "haar"))

        edges = cv2.Canny(gray, 60, 180)
        stages["edges"] = edges
        # Scaled kernels adapt to scene resolution and long / compact plates.
        gradient = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
        gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        gradient = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        stages["gradient"] = gradient
        kernels = ((0.015, 0.28), (0.03, 0.28), (0.055, 0.28), (0.035, 0.8), (0.065, 0.8))
        for scale, vertical in (kernels if "morphology" in self.config.proposal_sources else ()):
            kw, kh = max(3, round(width * scale)), max(3, round(height * scale * vertical))
            joined = cv2.morphologyEx(gradient, cv2.MORPH_CLOSE, np.ones((kh, kw), np.uint8))
            joined = cv2.morphologyEx(joined, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            stages[f"joined_{scale}_{vertical}"] = joined
            contours = cv2.findContours(joined, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
            for contour in contours:
                rect = cv2.minAreaRect(contour)
                (cx, cy), (rw, rh), angle = rect
                if min(rw, rh) < 12:
                    continue
                # Include a small margin around text bands before perspective rectification.
                quad = cv2.boxPoints(((cx, cy), (rw * 1.10, rh * 1.18), angle))
                x, y, w, h = cv2.boundingRect(quad)
                proposals.append(((x, y, w, h), quad, "morphology"))
        contours = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0] if "quadrilateral" in self.config.proposal_sources else []
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                proposals.append((cv2.boundingRect(approx), approx.reshape(4, 2).astype(np.float32), "quadrilateral"))

        candidates = []
        for (x, y, w, h), quad, source in proposals:
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(width, x + w), min(height, y + h)
            w, h = x1 - x0, y1 - y0
            if min(w, h) <= 0:
                continue
            aspect = w / h
            area = w * h / (width * height)
            if not (self.config.min_area_ratio <= area <= self.config.max_area_ratio):
                continue
            if not any(lo <= aspect <= hi for lo, hi in self.config.aspect_ranges):
                continue
            if quad is not None:
                quad = np.clip(quad, (0, 0), (width - 1, height - 1)).astype(np.float32)
                crop = rectify(image, quad)
            else:
                crop = image[y0:y1, x0:x1].copy()
            binary = cv2.threshold(gray_u8(crop), 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            density = np.count_nonzero(binary) / binary.size
            score = float(max(0, 1 - abs(density - 0.35) * 2))
            if source == "haar":
                score += 0.15
            if self.verifier is not None:
                classes = list(self.verifier.classes_)
                if 1 not in classes:
                    raise ValueError("Verifier must expose plate class 1")
                score = float(self.verifier.predict_proba([plate_features(crop)])[0, classes.index(1)])
                if score < self.config.verifier_threshold:
                    continue
            candidates.append(PlateCandidate((x0, y0, w, h), crop, score, source, quad))
        kept = nms(candidates, self.config.nms_iou)[:self.config.max_candidates]
        return DetectionResult(kept, stages, {
            "haar_enabled": self.cascade is not None, "raw_proposals": len(proposals),
            "filtered_proposals": len(candidates), "score_kind": "verifier_probability" if self.verifier else "heuristic",
        })
