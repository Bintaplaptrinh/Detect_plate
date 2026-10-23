from dataclasses import dataclass, asdict
from hashlib import sha256
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from .features import plate_features
from .imaging import read_image, intersection_over_union


@dataclass(frozen=True)
class Scene:
    path: str
    text: str
    box: tuple[int, int, int, int]
    layout: str
    group: str


def scan_vnlp(root: str | Path, max_per_layout: int | None = None, seed=42,
              box_format: str = "xyxy") -> tuple[list[Scene], dict]:
    """Parse VNLP filenames. Explicit xyxy default checked on repository images.

    The bundled VNLP readme calls the last values width/height, but the provided
    images use right/bottom coordinates. Never silently clip malformed labels.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if box_format not in {"xyxy", "xywh"}:
        raise ValueError("box_format must be xyxy or xywh")
    rng = np.random.default_rng(seed)
    scenes, digest, counts = [], sha256(), {}
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        paths = sorted(folder.glob("*.jpg"))
        if not paths:
            continue
        counts[folder.name] = len(paths)
        if max_per_layout is not None and len(paths) > max_per_layout:
            indices = sorted(rng.choice(len(paths), max_per_layout, replace=False).tolist())
            paths = [paths[i] for i in indices]
        for path in paths:
            parts = path.stem.split("_")
            if len(parts) != 8:
                raise ValueError(f"Unexpected VNLP filename: {path.name}")
            x, y, a, b = map(int, parts[-4:])
            w, h = (a - x, b - y) if box_format == "xyxy" else (a, b)
            image = read_image(path)
            ih, iw = image.shape[:2]
            if not (0 <= x < x + w <= iw and 0 <= y < y + h <= ih):
                raise ValueError(f"Invalid box for {path.name}; verify coordinate convention")
            text = parts[3].upper()
            digest.update(path.relative_to(root).as_posix().encode() + path.read_bytes())
            scenes.append(Scene(str(path), text, (x, y, w, h), folder.name, text))
    if not scenes:
        raise ValueError("No VNLP scenes found")
    return scenes, dict(root=str(root), raw_layout_counts=counts, selected=len(scenes), seed=seed,
                        selection_sha256=digest.hexdigest(), box_format=box_format, group="plate_identity")


def split_scenes(scenes: list[Scene], seed=42) -> dict[str, np.ndarray]:
    labels = np.array([s.layout for s in scenes])
    groups = np.array([s.group for s in scenes])
    for label in set(labels):
        if len(set(groups[labels == label])) < 5:
            raise ValueError(f"Need at least five plate identities for {label}")
    trainval, test = next(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(groups, labels, groups))
    train, val = next(StratifiedGroupKFold(4, shuffle=True, random_state=seed + 1).split(
        groups[trainval], labels[trainval], groups[trainval]))
    result = dict(train=trainval[train], validation=trainval[val], test=test)
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if set(groups[result[a]]) & set(groups[result[b]]):
            raise RuntimeError("Plate identity leakage")
    return result


def crop_scene(scene: Scene):
    image = read_image(scene.path)
    x, y, w, h = scene.box
    return image[y:y + h, x:x + w].copy()


def verifier_dataset(scenes: list[Scene], negatives_per_scene=3, seed=42):
    """Generate negatives away from labelled plate; no held-out scenes accepted implicitly."""
    rng = np.random.default_rng(seed)
    features, labels = [], []
    for scene in scenes:
        image = read_image(scene.path)
        ih, iw = image.shape[:2]
        x, y, w, h = scene.box
        features.append(plate_features(image[y:y + h, x:x + w]))
        labels.append(1)
        collected = 0
        for _ in range(negatives_per_scene * 100):
            nw = int(np.clip(round(w * rng.uniform(0.7, 1.5)), 24, iw))
            nh = int(np.clip(round(h * rng.uniform(0.7, 1.5)), 16, ih))
            nx, ny = int(rng.integers(0, iw - nw + 1)), int(rng.integers(0, ih - nh + 1))
            if intersection_over_union((nx, ny, nw, nh), scene.box) > 0.02:
                continue
            features.append(plate_features(image[ny:ny + nh, nx:nx + nw]))
            labels.append(0)
            collected += 1
            if collected >= negatives_per_scene:
                break
        if collected < negatives_per_scene:
            raise ValueError(f"Could not sample negatives for {scene.path}")
    return np.asarray(features, np.float32), np.asarray(labels)


def scene_manifest(scenes, splits):
    partition = {int(i): name for name, indices in splits.items() for i in indices}
    return [dict(**asdict(scene), partition=partition[i]) for i, scene in enumerate(scenes)]
