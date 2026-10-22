from dataclasses import dataclass


@dataclass(frozen=True)
class HOGConfig:
    size: tuple[int, int] = (32, 48)  # width, height
    block: tuple[int, int] = (16, 16)
    stride: tuple[int, int] = (8, 8)
    cell: tuple[int, int] = (8, 8)
    bins: int = 9
    deskew: bool = True
    preserve_aspect: bool = False  # canonical glyph box handles anisotropic camera/dataset scaling


@dataclass(frozen=True)
class DetectionConfig:
    cascade_path: str | None = None
    cascade_scale: float = 1.08
    cascade_neighbors: int = 3
    min_area_ratio: float = 0.0008
    max_area_ratio: float = 0.35
    aspect_ranges: tuple[tuple[float, float], ...] = ((0.6, 2.5), (2.5, 12.0))
    nms_iou: float = 0.35
    max_candidates: int = 30
    verifier_threshold: float = 0.5
    proposal_sources: tuple[str, ...] = ("haar", "morphology", "quadrilateral")


@dataclass(frozen=True)
class SegmentationConfig:
    method: str = "cca_plus"  # cca, projection, cca_plus
    threshold: str = "auto"  # auto, otsu, adaptive
    min_height_ratio: float = 0.14
    max_height_ratio: float = 0.92
    min_aspect: float = 0.08  # keep narrow I / 1
    max_aspect: float = 1.65
    min_ink: float = 0.08
    min_score: float = 0.50
    sigmoid_gain: float = 12.0
    row_gap_ratio: float = 0.45
    split_width_ratio: float = 1.8


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    svm_c: tuple[float, ...] = (0.3, 1.0, 3.0)
    svm_kernels: tuple[str, ...] = ("linear", "rbf")
    ada_depths: tuple[int, ...] = (2, 4)
    ada_learning_rates: tuple[float, ...] = (0.5, 1.0)
    ada_max_estimators: int = 80
    patience: int = 10
    min_delta: float = 0.0001
    expert_groups: tuple[str, ...] = ("Z2", "A4", "I1", "3B8", "0DQ", "S5", "G6")
