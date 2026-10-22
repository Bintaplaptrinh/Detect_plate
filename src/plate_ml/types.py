from dataclasses import dataclass, field
import numpy as np

Box = tuple[int, int, int, int]  # x, y, width, height, half-open bounds


@dataclass
class PlateCandidate:
    box: Box
    crop: np.ndarray
    score: float
    source: str
    quad: np.ndarray | None = None


@dataclass
class DetectionResult:
    candidates: list[PlateCandidate]
    stages: dict[str, np.ndarray]
    diagnostics: dict = field(default_factory=dict)


@dataclass
class Character:
    box: Box
    image: np.ndarray
    row: int
    score: float


@dataclass
class SegmentationResult:
    characters: list[Character]
    binary: np.ndarray
    horizontal_projection: np.ndarray
    vertical_projections: list[np.ndarray]
    components: list[dict]
    diagnostics: dict = field(default_factory=dict)
