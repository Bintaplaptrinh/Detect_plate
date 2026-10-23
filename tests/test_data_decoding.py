import csv
import cv2
import numpy as np
import pytest
from plate_ml.data import scan_characters, split_samples, Sample
from plate_ml.heuristics import decode_plate
from plate_ml.vnlp import scan_vnlp


def write_image(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((32, 24, 3), 255, np.uint8)
    image[4:28, 6:12 + value] = 0
    assert cv2.imwrite(str(path), image)


def test_pixel_duplicates_and_conflicting_labels(tmp_path):
    write_image(tmp_path / "A" / "one.png", 1)
    write_image(tmp_path / "A" / "copy.png", 1)
    samples, audit = scan_characters(tmp_path)
    assert len(samples) == 1 and audit["exact_duplicates_removed"] == 1
    assert audit["source_groups_verified"] is False
    write_image(tmp_path / "B" / "wrong.png", 1)
    with pytest.raises(ValueError, match="conflicting"):
        scan_characters(tmp_path)


def test_duplicates_join_source_families(tmp_path):
    for name, value in (("a", 0), ("b", 0), ("c", 1), ("d", 2)):
        write_image(tmp_path / "A" / f"{name}.png", value)
    mapping = tmp_path / "groups.csv"
    mapping.write_text("path,group\nA/a.png,source1\nA/b.png,source2\nA/c.png,source1\nA/d.png,source2\n")
    samples, audit = scan_characters(tmp_path, mapping)
    assert len({s.group for s in samples}) == 1
    assert audit["source_groups_verified"]


def test_group_split_and_strict_provenance():
    samples = [Sample(f"{label}/{i}.png", label, f"{label}-{i // 2}", f"{label}-{i}")
               for label in "AB" for i in range(40)]
    splits = split_samples(samples)
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert not {samples[i].group for i in splits[a]} & {samples[i].group for i in splits[b]}
    with pytest.raises(ValueError, match="Source group"):
        split_samples(samples, require_provenance=True)


def test_vnlp_coordinate_contract(tmp_path):
    folder = tmp_path / "one_row"
    folder.mkdir()
    cv2.imwrite(str(folder / "1_1_1_29A12345_20_30_80_60.jpg"), np.full((70, 100, 3), 255, np.uint8))
    scenes, _ = scan_vnlp(tmp_path, box_format="xyxy")
    assert scenes[0].box == (20, 30, 60, 30)
    with pytest.raises(ValueError, match="coordinate"):
        scan_vnlp(tmp_path, box_format="xywh")


def test_decoder_positional_evidence_and_unknown_layout():
    classes = np.asarray(list("0123456789AZ"))
    probabilities = np.full((8, len(classes)), 0.001)
    raw = "Z9A12345"
    for i, c in enumerate(raw):
        probabilities[i, list(classes).index(c)] = 0.9
    probabilities[0, list(classes).index("2")] = 0.8
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    result = decode_plate(probabilities, classes, (8,))
    assert result["raw"] == raw and result["text"] == "29A12345"
    assert decode_plate(probabilities, classes, (4, 4), templates=())["text"] == raw
    assert decode_plate(probabilities, classes, (8,), max_log_penalty=0)["text"] == raw
