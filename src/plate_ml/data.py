from dataclasses import dataclass, asdict, replace
from hashlib import sha256
from pathlib import Path
import csv
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from .imaging import read_image, gray_u8


@dataclass(frozen=True)
class Sample:
    path: str
    label: str
    group: str
    digest: str


def scan_characters(root: str | Path, group_csv: str | Path | None = None,
                    max_per_class: int | None = None, seed: int = 42) -> tuple[list[Sample], dict]:
    """Read class folders, deduplicate decoded pixels, optionally group source families.

    group_csv columns: path (relative to root), group (original glyph/source ID).
    Pre-generated augmentations cannot be reconstructed safely from arbitrary filenames.
    Without a provenance map, metrics are explicitly exploratory.
    """
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if max_per_class is not None and max_per_class < 5:
        raise ValueError("Need at least five samples per class")
    groups = {}
    if group_csv is not None:
        with Path(group_csv).open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                key = Path(row["path"]).as_posix()
                if not row["group"] or key in groups:
                    raise ValueError(f"Empty or duplicate group mapping: {key}")
                groups[key] = row["group"]
    samples, seen, duplicates, missing = [], {}, 0, []
    parents, digest_groups = {}, {}

    def find(group):
        parents.setdefault(group, group)
        while parents[group] != group:
            parents[group] = parents[parents[group]]
            group = parents[group]
        return group
    dataset_hash = sha256()
    counts = {}
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        if len(folder.name) != 1 or folder.name not in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            continue
        counts[folder.name] = 0
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            gray = gray_u8(read_image(path))
            digest = sha256(str(gray.shape).encode() + gray.tobytes()).hexdigest()
            relative = path.relative_to(root).as_posix()
            dataset_hash.update(f"{relative}:{digest}\n".encode())
            counts[folder.name] += 1
            if relative not in groups:
                missing.append(relative)
            group = groups.get(relative, digest)
            if digest in digest_groups:
                parents[find(group)] = find(digest_groups[digest])
            else:
                digest_groups[digest] = group
                find(group)
            if digest in seen:
                if seen[digest] != folder.name:
                    raise ValueError(f"Identical pixels have conflicting labels: {relative}")
                duplicates += 1
                continue
            seen[digest] = folder.name
            samples.append(Sample(str(path), folder.name, group, digest))
    if not samples:
        raise ValueError(f"No class images in {root}")
    if group_csv is not None and missing:
        raise ValueError(f"Group map missing {len(missing)} images, e.g. {missing[0]}")
    samples = [replace(s, group=find(s.group)) for s in samples]
    rng = np.random.default_rng(seed)
    if max_per_class is not None:
        selected = []
        for label in sorted({s.label for s in samples}):
            items = [s for s in samples if s.label == label]
            indices = sorted(rng.choice(len(items), min(max_per_class, len(items)), replace=False).tolist())
            selected.extend(items[i] for i in indices)
        samples = selected
    selected_hash = sha256("\n".join(f"{s.digest}:{s.label}:{s.group}" for s in samples).encode()).hexdigest()
    return samples, dict(root=str(root), raw_counts=counts, selected=len(samples),
                         exact_duplicates_removed=duplicates, dataset_sha256=dataset_hash.hexdigest(),
                         selection_sha256=selected_hash, source_groups_verified=group_csv is not None,
                         metric_status="source_grouped" if group_csv else "exploratory_unknown_augmentation_lineage",
                         seed=seed, max_per_class=max_per_class)


def split_samples(samples: list[Sample], seed: int = 42, require_provenance: bool = False,
                  audit: dict | None = None) -> dict[str, np.ndarray]:
    if require_provenance and not (audit or {}).get("source_groups_verified"):
        raise ValueError("Source group CSV required for publication-grade evaluation")
    y = np.array([s.label for s in samples])
    groups = np.array([s.group for s in samples])
    for label in np.unique(y):
        if len(np.unique(groups[y == label])) < 5:
            raise ValueError(f"Class {label} needs at least five independent groups")
    # Approximately 60/20/20, same groups never cross partitions.
    outer = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    development, test = next(outer.split(np.zeros(len(y)), y, groups))
    inner = StratifiedGroupKFold(4, shuffle=True, random_state=seed + 1)
    train_local, val_local = next(inner.split(np.zeros(len(development)), y[development], groups[development]))
    splits = dict(train=development[train_local], validation=development[val_local], test=test)
    classes = set(y)
    for name, indices in splits.items():
        if set(y[indices]) != classes:
            raise ValueError(f"Partition {name} lacks a class; add groups or change seed")
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if set(groups[splits[a]]) & set(groups[splits[b]]):
            raise RuntimeError("Group leakage")
    return splits


def sample_manifest(samples, splits):
    partition = {int(i): name for name, indices in splits.items() for i in indices}
    return [dict(**asdict(sample), partition=partition[i]) for i, sample in enumerate(samples)]
