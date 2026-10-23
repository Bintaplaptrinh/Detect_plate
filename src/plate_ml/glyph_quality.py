"""Conservative, image-only glyph rejection between segmentation and OCR.

The filter uses relative row geometry and multiple signs of edge contamination.
It never uses the desired plate text, a predicted digit, or a forced plate count.
"""
from dataclasses import asdict, dataclass, replace

import cv2
import numpy as np


@dataclass(frozen=True)
class GlyphQualityConfig:
    min_row_support: int = 3
    min_relative_height: float = 0.50
    edge_margin: float = 0.025
    fragmented_fraction: float = 0.65
    fragmented_row_offset: float = 0.18
    solid_stroke_ink: float = 0.75
    solid_stroke_aspect: float = 0.18
    isolated_edge_gap: float = 0.35
    solid_stroke_row_offset: float = 0.08
    filter_fragmented_edges: bool = True
    filter_solid_edges: bool = True


def _row_reference(chars):
    height = float(np.median([c.box[3] for c in chars]))
    stable = [c for c in chars if .85*height <= c.box[3] <= 1.25*height
              and c.box[2]/c.box[3] >= .2]
    anchors = stable if len(stable) >= 3 else chars
    centers = np.array([(c.box[0]+c.box[2]/2, c.box[1]+c.box[3]/2) for c in anchors])
    # A median pairwise slope tolerates tilted rows and an isolated bumper blob.
    slopes = [(b[1]-a[1])/(b[0]-a[0]) for i,a in enumerate(centers)
              for b in centers[i+1:] if abs(b[0]-a[0]) > height*.5]
    slope = float(np.median(slopes)) if slopes else 0.0
    intercept = float(np.median(centers[:,1]-slope*centers[:,0]))
    return height, slope, intercept


def filter_glyphs(segmentation, config=GlyphQualityConfig()):
    """Return a new segmentation plus per-glyph evidence; leave input untouched.

    Full-height narrow 1/I remains valid. Fragmentation alone does not remove a
    broken real glyph: edge position and row inconsistency must also support it.
    """
    image_h, image_w = segmentation.binary.shape
    chars = segmentation.characters
    keep, decisions = [], []
    rows = {row: [c for c in chars if c.row == row] for row in {c.row for c in chars}}
    references = {row: _row_reference(items) for row,items in rows.items() if len(items) >= config.min_row_support}
    for index, char in enumerate(chars):
        x,y,w,h = char.box
        foreground = (char.image > 0).astype(np.uint8)
        _, _, stats, _ = cv2.connectedComponentsWithStats(foreground)
        areas = stats[1:, cv2.CC_STAT_AREA]
        ink = int(areas.sum())
        fraction = float(areas.max()/max(1,ink)) if len(areas) else 0.0
        significant = int(np.count_nonzero(areas >= max(2, ink*.015)))
        fill = ink/max(1,w*h)
        edge = x <= max(2,image_w*config.edge_margin) or x+w >= image_w-max(2,image_w*config.edge_margin)
        relative_height, offset, gap = None, None, 0.0
        reasons = []
        if not ink:
            reasons.append('empty_glyph')
        if char.row in references:
            reference,slope,intercept = references[char.row]
            relative_height = h/max(1,reference)
            offset = abs(y+h/2-(slope*(x+w/2)+intercept))/max(1,reference)
            row = sorted(rows[char.row],key=lambda c:c.box[0])
            position = next(i for i,c in enumerate(row) if c is char)
            if position == 0 and len(row)>1:
                gap = max(0,row[1].box[0]-(x+w))/max(1,reference)
            elif position == len(row)-1 and position:
                previous = row[position-1].box
                gap = max(0,x-(previous[0]+previous[2]))/max(1,reference)
            if relative_height < config.min_relative_height:
                reasons.append('too_short_for_row')
            if (config.filter_fragmented_edges and edge and significant >= 2 and fraction < config.fragmented_fraction
                    and offset > config.fragmented_row_offset):
                reasons.append('fragmented_edge_outside_row')
            if (config.filter_solid_edges and edge and w/max(1,h) < config.solid_stroke_aspect
                    and fill > config.solid_stroke_ink and gap > config.isolated_edge_gap
                    and offset > config.solid_stroke_row_offset):
                reasons.append('isolated_solid_edge_stroke')
        accepted = not reasons
        if accepted:
            keep.append(char)
        decisions.append(dict(index=index, box=list(char.box), row=char.row,
            accepted=accepted, reasons=reasons, ink_fraction=float(fill),
            largest_component_fraction=fraction, significant_components=significant,
            near_edge=bool(edge), relative_height=relative_height, row_offset=offset,
            outside_gap=gap))
    diagnostics = dict(segmentation.diagnostics, glyph_quality=dict(
        stage='after_segmentation_before_HOG_and_OCR', config=asdict(config),
        input_count=len(chars), kept_count=len(keep), rejected_count=len(chars)-len(keep),
        decisions=decisions))
    # Character crops and their row/box metadata remain a single ordered list.
    return replace(segmentation, characters=keep, diagnostics=diagnostics)


class QualityFilteredSegmenter:
    """Filter every existing hypothesis, including the row-projection fallback.

    Keeping the alternative thresholds matters: a glyph connected to a frame in
    CCA may already be present in projection, with only edge junk lowering its
    previous OCR score. No text-specific recovery/insertion is performed.
    """
    def __init__(self, base, config=GlyphQualityConfig()):
        self.base, self.config = base, config

    def hypotheses(self, image):
        candidates = self.base.hypotheses(image) if hasattr(self.base,'hypotheses') else [self.base.segment(image)]
        return [filter_glyphs(s,self.config) for s in candidates]

    def segment(self, image):
        # Production hypothesis choice remains in ResearchRecognizer using OCR.
        return max(self.hypotheses(image),key=lambda s:s.diagnostics.get('geometry',0))
