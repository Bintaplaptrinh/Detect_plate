import cv2
import numpy as np
import pytest

from plate_ml.features import HOGExtractor
from plate_ml.imaging import intersection_over_union
from plate_ml.research_pipeline import PanelProposer, RowProjectionSegmenter


def plate(rows):
    image = np.full((140 if len(rows) == 2 else 80, 300, 3), 255, np.uint8)
    for row, text in enumerate(rows):
        for i, char in enumerate(text):
            cv2.putText(image, char, (12+i*34, 52+row*62), cv2.FONT_HERSHEY_SIMPLEX,
                        1.1, (0,0,0), 2)
    return image


@pytest.mark.parametrize('rows', [('29A12345',), ('29A', '12345')])
def test_row_projection_recovers_reading_order_without_count_or_layout_input(rows):
    result = RowProjectionSegmenter(deskew=False).segment(plate(rows))
    assert len(result.characters) == 8
    assert [c.row for c in result.characters] == ([0]*8 if len(rows)==1 else [0]*3+[1]*5)
    assert HOGExtractor().transform(c.image for c in result.characters).shape == (8,540)


def test_white_panel_localization_without_filename_or_ground_truth():
    image = np.full((240,480,3), 30, np.uint8)
    image[100:180,90:390] = plate(('29A12345',))
    proposals = PanelProposer().propose(image)
    assert max(intersection_over_union(p.box, (90,100,300,80)) for p in proposals) > .9
    assert all(p.crop is not None and p.crop.size for p in proposals)


def test_blank_research_segmentation_has_no_forced_characters():
    result = RowProjectionSegmenter().segment(np.full((70,200,3),255,np.uint8))
    assert not result.characters
