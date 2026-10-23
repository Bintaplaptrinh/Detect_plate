from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PlateTemplate:
    name: str
    positions: tuple[str, ...]
    rows: tuple[int, ...]


DIGITS = "0123456789"
LETTERS = "ABCDEFGHJKLMNPRSTUVXYZ"
# Research templates, not an exhaustive statement of Vietnamese registration law.
# Geometry is inferred from segmentation, independent of vehicle/front/rear labels.
DEFAULT_TEMPLATES = (
    PlateTemplate("car_5_digit", (DIGITS, DIGITS, LETTERS) + (DIGITS,) * 5, (8,)),
    PlateTemplate("car_4_digit", (DIGITS, DIGITS, LETTERS) + (DIGITS,) * 4, (7,)),
    PlateTemplate("compact_car_5_digit", (DIGITS, DIGITS, LETTERS) + (DIGITS,) * 5, (3, 5)),
    PlateTemplate("moto_series_digit_5", (DIGITS, DIGITS, LETTERS, DIGITS) + (DIGITS,) * 5, (4, 5)),
    PlateTemplate("moto_series_digit_4", (DIGITS, DIGITS, LETTERS, DIGITS) + (DIGITS,) * 4, (4, 4)),
    PlateTemplate("moto_two_letters_5", (DIGITS, DIGITS, LETTERS, LETTERS) + (DIGITS,) * 5, (4, 5)),
)


def decode_plate(probabilities: np.ndarray, classes, row_lengths: tuple[int, ...],
                 templates=DEFAULT_TEMPLATES, max_log_penalty=0.35) -> dict:
    """Choose a positional template only when evidence is close to raw OCR.

    Unknown formats retain raw text. Never globally replace Z/2 or B/8.
    No province-code validity or electric / MD format assumptions are made.
    """
    classes = np.asarray(classes)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(classes):
        raise ValueError("Probability dimensions do not match classes")
    if sum(row_lengths) != len(probabilities):
        raise ValueError("Row lengths do not match characters")
    if len(probabilities) == 0:
        return dict(raw="", text="", template=None, changed=False, mean_log_penalty=0.0)
    raw_indices = probabilities.argmax(axis=1)
    raw = "".join(classes[raw_indices])
    raw_score = float(np.log(np.maximum(probabilities[np.arange(len(raw)), raw_indices], 1e-12)).sum())
    options = []
    for template in templates:
        if template.rows != row_lengths or len(template.positions) != len(probabilities):
            continue
        indices = []
        for row, allowed in zip(probabilities, template.positions):
            valid = np.flatnonzero(np.isin(classes, list(allowed)))
            if not len(valid):
                break
            indices.append(int(valid[np.argmax(row[valid])]))
        if len(indices) != len(probabilities):
            continue
        score = float(np.log(np.maximum(probabilities[np.arange(len(indices)), indices], 1e-12)).sum())
        options.append((score, template.name, "".join(classes[indices])))
    if not options:
        return dict(raw=raw, text=raw, template=None, changed=False, mean_log_penalty=0.0)
    score, name, text = max(options)
    penalty = (raw_score - score) / len(probabilities)
    accepted = penalty <= max_log_penalty
    return dict(raw=raw, text=text if accepted else raw, template=name if accepted else None,
                changed=accepted and text != raw, mean_log_penalty=penalty)
