"""Experimental plate proposals based on contrast/shape, with both text polarities.

No hue palette, registration-category lookup, filename, or GT enters inference.
The earlier white/yellow proposal implementation remains available as a baseline.
"""
from dataclasses import replace

import cv2
import numpy as np

from .imaging import gray_u8, nms, rectify
from .research_pipeline import RegionConfig, ResearchRecognizer, TwoStageDetector
from .types import PlateCandidate


class ContrastProposer:
    """Vertical stroke bands and quadrilateral edges; no HSV/color filtering."""
    def __init__(self, config=RegionConfig()):
        self.config=config

    def propose(self,image):
        oh,ow=image.shape[:2]
        scale=min(1.0,self.config.max_side/max(oh,ow))
        gray=gray_u8(image)
        if scale<1:
            gray=cv2.resize(gray,None,fx=scale,fy=scale)
        h,w=gray.shape
        contrast=cv2.createCLAHE(2.0,(8,8)).apply(gray)
        gradient=cv2.convertScaleAbs(cv2.Sobel(contrast,cv2.CV_16S,1,0,ksize=3))
        gradient=cv2.threshold(gradient,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
        sources=[]
        for kw,kh in ((9,3),(17,3),(27,5),(41,7),(13,13),(23,21)):
            joined=cv2.morphologyEx(gradient,cv2.MORPH_CLOSE,np.ones((kh,kw),np.uint8))
            sources.append((f'strokes_{kw}_{kh}',joined))
        sources.append(('quadrilateral',cv2.Canny(contrast,50,150)))
        groups={'strokes':[],'quadrilateral':[]}
        for source,mask in sources:
            for contour in cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0]:
                area=cv2.contourArea(contour)
                if not .0007*w*h<=area<=.45*w*h:
                    continue
                if source=='quadrilateral':
                    approx=cv2.approxPolyDP(contour,.025*cv2.arcLength(contour,True),True)
                    if len(approx)!=4 or not cv2.isContourConvex(approx):
                        continue
                    quad=approx.reshape(4,2).astype(np.float32)
                    (cx,cy),(rw,rh),angle=cv2.minAreaRect(quad)
                else:
                    (cx,cy),(rw,rh),angle=cv2.minAreaRect(contour)
                    quad=cv2.boxPoints(((cx,cy),(rw*1.12,rh*1.20),angle))
                if min(rw,rh)<13 or max(rw,rh)/max(1,min(rw,rh))>11:
                    continue
                quad=np.clip(quad/scale,(0,0),(ow-1,oh-1)).astype(np.float32)
                x,y,bw,bh=cv2.boundingRect(quad)
                bw,bh=min(bw,ow-x),min(bh,oh-y)
                if not .4<=bw/max(1,bh)<=12:
                    continue
                fill=min(1.,area/max(1,rw*rh))
                p=PlateCandidate((x,y,bw,bh),None,float(fill),source,quad)
                groups['quadrilateral' if source=='quadrilateral' else 'strokes'].append(p)
        # Reserve space for both structural branches before global deduplication:
        # one class of large vehicle contours must not consume the entire budget.
        budget=self.config.max_proposals//2
        diverse=[p for group in groups.values() for p in nms(group,.85)[:budget]]
        kept=nms(diverse,.85)[:self.config.max_proposals]
        for p in kept:
            p.crop=rectify(image,p.quad)
        return kept


class DualPolaritySegmenter:
    """Try dark-on-light and light-on-dark before the unchanged glyph pipeline."""
    def __init__(self,base):
        self.base=base

    def hypotheses(self,image):
        results=[]
        for polarity,prepared in [('dark_on_light',image),('light_on_dark',255-image)]:
            hypotheses=(self.base.hypotheses(prepared) if hasattr(self.base,'hypotheses')
                        else [self.base.segment(prepared)])
            for result in hypotheses:
                results.append(replace(result,diagnostics=dict(result.diagnostics,
                    polarity=polarity)))
        return results

    def segment(self,image):
        return max(self.hypotheses(image),key=lambda s:s.diagnostics.get('geometry',0))


def contrast_recognizer(base, structural=True, dual_polarity=True):
    """Reuse frozen VNLP verifier/OCR; this factory does not imply retraining."""
    detector=base.detector
    if structural:
        detector=TwoStageDetector(base.detector.verifier,base.detector.config)
        detector.proposer=ContrastProposer(base.detector.config)
    segmenter=DualPolaritySegmenter(base.segmenter) if dual_polarity else base.segmenter
    return ResearchRecognizer(base.model,detector,segmenter,base.extractor.config,
        base.templates,base.rerank_weight,base.top_k)
