from types import SimpleNamespace

import numpy as np
import pytest

from plate_ml.demo import (DemoEngine, acceptable_multi, annotate, group_candidates,
                           observation_confidence, same_region, select_distinct, tile_origins)
from plate_ml.types import DetectionResult, PlateCandidate


@pytest.mark.parametrize('length', [300, 960, 961, 1456, 2184, 4000])
def test_tiles_cover_both_edges_without_gaps_or_duplicate_origins(length):
    origins = tile_origins(length)
    assert origins == sorted(set(origins))
    assert origins[0] == 0
    assert origins[-1] == max(0, length-960)
    assert all(b-a <= 960 for a, b in zip(origins, origins[1:]))


def test_region_grouping_does_not_merge_nearby_plates_or_large_parent_boxes():
    assert same_region((10, 10, 100, 50), (12, 12, 95, 49))
    assert not same_region((10, 10, 100, 50), (120, 10, 100, 50))
    assert not same_region((0, 0, 1000, 1000), (20, 20, 100, 50))
    candidates = [PlateCandidate((x, 10, 100, 50), None, score, 'test')
                  for x, score in [(10, 3.), (12, 2.), (120, 1.)]]
    groups = group_candidates(candidates)
    assert [len(g) for g in groups] == [2, 1]


def test_distinct_selection_keeps_equal_text_at_different_positions():
    detections = [dict(box=box, text='29A12345', rank_score=score) for box, score in
                  [([10, 10, 100, 50], 3), ([12, 12, 95, 48], 2), ([200, 10, 100, 50], 1)]]
    assert [r['box'][0] for r in select_distinct(detections)] == [10, 200]


def test_tile_boxes_are_mapped_to_original_image_coordinates():
    class Detector:
        def detect(self, image):
            return DetectionResult([PlateCandidate((12, 18, 40, 30), np.zeros((30,40,3), np.uint8),
                2., 'white', np.float32([[12,18],[52,18],[52,48],[12,48]]))], {}, {'raw_proposals': 1})

    engine = DemoEngine.__new__(DemoEngine)
    engine.multi_detector = SimpleNamespace(detect=lambda image: DetectionResult([], {}, {'raw_proposals': 0}))
    engine.tile_detector = Detector()
    groups, diagnostics = engine.multi_candidates(np.zeros((1100,1200,3),np.uint8))
    assert {g[0].box for g in groups} == {(12,18,40,30), (252,18,40,30), (12,158,40,30), (252,158,40,30)}
    assert diagnostics['views'] == 5
    assert diagnostics['raw_proposals'] == 4
    assert any(np.all(g[0].quad[0] == [252,158]) for g in groups)


def test_confidence_uses_decoded_character_probabilities_and_filters_out_of_format():
    result = dict(decoded={'text': 'AB'}, probabilities=np.array([[.99,.01],[.99,.01]]))
    assert observation_confidence(result, ['A','B']) == pytest.approx(np.sqrt(.99*.01))
    valid = dict(decoded={'text':'59LA17006'}, quality=.2)
    assert acceptable_multi(valid, .8)
    assert not acceptable_multi(valid, .2)
    assert not acceptable_multi(dict(decoded={'text':'159LA17006'},quality=.2), .8)


def test_single_mode_selects_requested_model_and_preserves_top_result():
    class Recognizer:
        def __init__(self, text):
            self.text = text
            self.model = SimpleNamespace(classes_=np.asarray(sorted(set(text))))

        def recognize(self, image):
            p=np.asarray([[float(c==label) for label in self.model.classes_] for c in self.text])
            return DetectionResult([],{},{}), [dict(candidate=PlateCandidate((2,2,20,16),image,2.,'single'),
                decoded=dict(text=self.text,raw=self.text), probabilities=p, quality=.5)]

    engine = DemoEngine.__new__(DemoEngine)
    engine.recognizers = {'svm':Recognizer('29A12345'),'pca_fld_hmm':Recognizer('54T39182')}
    for algorithm, expected in [('svm','29A12345'),('pca_fld_hmm','54T39182')]:
        result,image=engine.recognize(np.zeros((40,60,3),np.uint8),algorithm,False)
        assert [r['text'] for r in result['detections']] == [expected]
        assert result['processing_seconds'] >= 0
        assert image.shape == (40,60,3)
    with pytest.raises(ValueError, match='Thuật toán'):
        engine.recognize(np.zeros((40,60,3),np.uint8),'unknown',False)
    with pytest.raises(ValueError, match='bật hoặc tắt'):
        engine.recognize(np.zeros((40,60,3),np.uint8),'svm','false')


def test_annotation_handles_tiny_images_and_boxes_at_edges():
    image=np.zeros((32,32,3),np.uint8)
    output=annotate(image,[dict(id=1,text='59LA17006',box=[0,0,31,31])])
    assert output.shape == image.shape
    assert output.any()
    assert not image.any()
