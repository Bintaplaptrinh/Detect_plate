from dataclasses import replace

import numpy as np

from plate_ml.glyph_quality import filter_glyphs
from plate_ml.types import Character, SegmentationResult


def glyph(x, y=10, w=20, h=40, row=0):
    image = np.zeros((h,w),np.uint8)
    image[:,max(0,w//3):max(1,2*w//3)] = 255
    return Character((x,y,w,h),image,row,1)


def segmentation(chars, shape=(80,230)):
    return SegmentationResult(chars,np.zeros(shape,np.uint8),np.zeros(shape[0]),[],[],{})


def test_short_bumper_blob_removed_before_classification_without_mutating_input():
    chars = [glyph(x) for x in (20,50,80,110)] + [glyph(143,y=40,w=3,h=9)]
    original = segmentation(chars)
    cleaned = filter_glyphs(original)
    assert len(cleaned.characters) == 4
    assert len(original.characters) == 5
    assert cleaned.diagnostics['glyph_quality']['decisions'][-1]['reasons'] == ['too_short_for_row']
    assert all(a is b for a,b in zip(chars[:4],cleaned.characters))


def test_aligned_narrow_one_at_edge_is_kept_even_with_large_gap():
    one = Character((0,10,3,40),np.full((40,3),255,np.uint8),0,1)
    chars = [one] + [glyph(x) for x in (40,70,100)]
    assert filter_glyphs(segmentation(chars)).characters == chars


def test_displaced_isolated_solid_border_is_removed():
    frame = Character((227,3,3,40),np.full((40,3),255,np.uint8),0,1)
    chars = [glyph(x) for x in (20,50,80,110)] + [frame]
    cleaned = filter_glyphs(segmentation(chars))
    assert cleaned.characters == chars[:-1]
    assert 'isolated_solid_edge_stroke' in cleaned.diagnostics['glyph_quality']['decisions'][-1]['reasons']


def test_reference_is_per_row_and_tolerates_slope():
    top = [glyph(x,y=10+i*3,h=40) for i,x in enumerate((20,50,80))]
    bottom = [glyph(x,y=70,h=22,row=1) for x in (20,50,80,110)]
    original = segmentation(top+bottom,shape=(110,230))
    assert filter_glyphs(original).characters == top+bottom


def test_fragmentation_alone_does_not_remove_broken_interior_character():
    broken = glyph(80)
    broken.image[:] = 0
    broken.image[:15,5:10] = 255
    broken.image[25:,5:10] = 255
    chars = [glyph(20),glyph(50),broken,glyph(110)]
    assert filter_glyphs(segmentation(chars)).characters == chars


def test_insufficient_row_support_does_not_infer_noise_from_height():
    chars = [glyph(20),glyph(50,h=8)]
    assert filter_glyphs(segmentation(chars)).characters == chars
