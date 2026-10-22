import cv2
import numpy as np
import pytest

from plate_ml.config import SegmentationConfig
from plate_ml.contrast_proposals import ContrastProposer,DualPolaritySegmenter
from plate_ml.features import HOGExtractor
from plate_ml.imaging import intersection_over_union
from plate_ml.segmentation import CharacterSegmenter


def card(background=(240,240,240),ink=(10,10,10)):
    image=np.full((80,300,3),background,np.uint8)
    cv2.rectangle(image,(1,1),(298,78),ink,2)
    for i,c in enumerate('29A12345'):
        cv2.putText(image,c,(12+i*34,53),cv2.FONT_HERSHEY_SIMPLEX,1.1,ink,2)
    return image


@pytest.mark.parametrize('body',[(0,220,240),(230,230,230),(150,40,20)])
@pytest.mark.parametrize('background,ink',[
    ((240,240,240),(10,10,10)),((0,220,240),(10,10,10)),
    ((150,40,20),(245,245,245)),((20,20,170),(245,245,245)),
    ((40,100,20),(245,245,245))])
def test_structural_proposals_cover_both_text_polarities_and_colored_bodies(body,background,ink):
    image=np.full((240,480,3),body,np.uint8)
    image[100:180,90:390]=card(background,ink)
    proposals=ContrastProposer().propose(image)
    assert max((intersection_over_union(p.box,(90,100,300,80)) for p in proposals),default=0)>.8
    assert all(p.source.startswith(('strokes','quadrilateral')) for p in proposals)


def test_inverse_branch_supplies_identical_canonical_glyphs_for_inverted_input():
    image=card()
    base=CharacterSegmenter(SegmentationConfig(method='cca',threshold='otsu'))
    expected=base.segment(image)
    candidates=DualPolaritySegmenter(base).hypotheses(255-image)
    recovered=next(s for s in candidates if s.diagnostics['polarity']=='light_on_dark')
    assert len(expected.characters)==len(recovered.characters)==8
    extractor=HOGExtractor()
    np.testing.assert_array_equal(extractor.transform(c.image for c in expected.characters),
                                  extractor.transform(c.image for c in recovered.characters))
