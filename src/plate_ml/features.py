import cv2
import numpy as np
from .config import HOGConfig
from .imaging import gray_u8


def normalize_glyph(image: np.ndarray, config: HOGConfig = HOGConfig()) -> np.ndarray:
    gray = gray_u8(image)
    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    border = np.concatenate((binary[0], binary[-1], binary[:, 0], binary[:, -1]))
    if np.mean(border) > 127:  # always white foreground on black
        binary = cv2.bitwise_not(binary)
    points = cv2.findNonZero(binary)
    w, h = config.size
    if points is None:
        return np.zeros((h, w), np.uint8)
    x, y, bw, bh = cv2.boundingRect(points)
    glyph = binary[y:y + bh, x:x + bw]
    if config.deskew:
        margin = max(bw, bh) // 2
        glyph = cv2.copyMakeBorder(glyph, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=0)
        moment = cv2.moments(glyph)
        if abs(moment["mu02"]) > 1e-3:
            skew = moment["mu11"] / moment["mu02"]
            center_y = moment["m01"] / moment["m00"]
            matrix = np.float32([[1, skew, -center_y * skew], [0, 1, 0]])
            glyph = cv2.warpAffine(glyph, matrix, (glyph.shape[1], glyph.shape[0]), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
        gx, gy, bw, bh = cv2.boundingRect(cv2.findNonZero(glyph))
        glyph = glyph[gy:gy + bh, gx:gx + bw]
    scale = min((w - 8) / bw, (h - 8) / bh)
    nw, nh = (max(1, round(bw * scale)), max(1, round(bh * scale))) if config.preserve_aspect else (w - 8, h - 8)
    resized = cv2.resize(glyph, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.zeros((h, w), np.uint8)
    left, top = (w - nw) // 2, (h - nh) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


class HOGExtractor:
    def __init__(self, config: HOGConfig = HOGConfig()):
        self.config = config
        self._descriptor = cv2.HOGDescriptor(config.size, config.block, config.stride, config.cell, config.bins)

    def transform_one(self, image: np.ndarray) -> np.ndarray:
        return self._descriptor.compute(normalize_glyph(image, self.config)).ravel()

    def transform(self, images) -> np.ndarray:
        features = [self.transform_one(image) for image in images]
        if not features:
            return np.empty((0, self._descriptor.getDescriptorSize()), dtype=np.float32)
        return np.asarray(features, dtype=np.float32)


def plate_features(image: np.ndarray) -> np.ndarray:
    """Shape-normalized HOG for the optional plate/non-plate verifier."""
    gray = cv2.resize(gray_u8(image), (96, 48))
    descriptor = cv2.HOGDescriptor((96, 48), (16, 16), (8, 8), (8, 8), 9)
    return descriptor.compute(gray).ravel()
