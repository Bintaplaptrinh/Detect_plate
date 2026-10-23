from pathlib import Path
import cv2
import numpy as np


def read_image(path: str | Path) -> np.ndarray:
    """Unicode-safe image read on Windows."""
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def gray_u8(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0 or image.dtype != np.uint8:
        raise ValueError("Expected a nonempty uint8 image")
    if image.ndim == 2:
        return image.copy()
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("Expected grayscale, BGR or BGRA")
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY if image.shape[2] == 3 else cv2.COLOR_BGRA2GRAY)


def preprocess(image: np.ndarray) -> dict[str, np.ndarray]:
    gray = gray_u8(image)
    contrast = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    smooth = cv2.GaussianBlur(contrast, (3, 3), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    top = cv2.morphologyEx(smooth, cv2.MORPH_TOPHAT, kernel)
    black = cv2.morphologyEx(smooth, cv2.MORPH_BLACKHAT, kernel)
    enhanced = cv2.subtract(cv2.add(smooth, top), black)
    otsu = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    adaptive = cv2.adaptiveThreshold(smooth, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 25, 9)
    return dict(gray=gray, contrast=contrast, smooth=smooth, top_hat=top,
                black_hat=black, enhanced=enhanced, otsu=otsu, adaptive=adaptive)


def rectify(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Order minAreaRect corners cyclically; retain upright portrait/landscape layout."""
    points = np.asarray(quad, np.float32).reshape(4, 2)
    center = points.mean(axis=0)
    points = points[np.argsort(np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0]))]
    points = np.roll(points, -int(np.argmin(points.sum(axis=1))), axis=0)
    width = np.linalg.norm(points[1] - points[0])
    height = np.linalg.norm(points[2] - points[1])
    w, h = max(2, round(float(width))), max(2, round(float(height)))
    target = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    return cv2.warpPerspective(image, cv2.getPerspectiveTransform(points, target), (w, h))


def intersection_over_union(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    inter = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0, min(ay + ah, by + bh) - max(ay, by))
    return inter / max(aw * ah + bw * bh - inter, 1)


def nms(items: list, threshold: float) -> list:
    kept = []
    for item in sorted(items, key=lambda c: c.score, reverse=True):
        if all(intersection_over_union(item.box, k.box) < threshold for k in kept):
            kept.append(item)
    return kept
