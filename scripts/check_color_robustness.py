"""Controlled color/body stress tests plus VNLP validation regression.

Palette names describe rendered colors, not legal registration categories.
No new model is trained, and these controls are not a road-image benchmark.
"""
from datetime import datetime,timezone
from pathlib import Path
import json,sys

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from run_glyph_filter_experiment import load_filter_release
from plate_ml.contrast_proposals import contrast_recognizer
from plate_ml.evaluation import sequence_metrics
from plate_ml.imaging import read_image,intersection_over_union
from plate_ml.registry import ModelRegistry,code_info
from plate_ml.research_experiment import write_json


STYLES={
    'white_black':((240,240,240),(10,10,10)),
    'yellow_black':((0,220,240),(10,10,10)),
    'blue_white':((150,40,20),(245,245,245)),
    'red_white':((20,20,170),(245,245,245)),
    'green_white':((40,100,20),(245,245,245)),
}
BODIES={'yellow':(0,220,240),'white':(230,230,230),'blue':(160,50,20),
        'red':(20,20,160),'green':(30,110,30)}


def render(body,style=None,two_rows=False,grille=False):
    image=np.full((360,640,3),50,np.uint8)
    cv2.rectangle(image,(35,35),(605,325),body,-1)
    cv2.rectangle(image,(85,60),(555,125),(35,35,35),-1)
    cv2.rectangle(image,(60,155),(145,195),(230,230,230),-1)
    cv2.rectangle(image,(495,155),(580,195),(230,230,230),-1)
    if grille:
        for y in range(225,285,9):cv2.line(image,(170,y),(470,y),(15,15,15),3)
    if style is None:return image,None,''
    background,ink=STYLES[style]
    rows=('29A1','12345') if two_rows else ('29A12345',)
    h=140 if two_rows else 80
    panel=np.full((h,300,3),background,np.uint8)
    cv2.rectangle(panel,(1,1),(298,h-2),ink,2)
    for row,text in enumerate(rows):
        for i,c in enumerate(text):
            cv2.putText(panel,c,(12+i*34,52+row*62),cv2.FONT_HERSHEY_SIMPLEX,1.1,ink,2)
    x,y=170,200 if two_rows else 230
    image[y:y+h,x:x+300]=panel
    return image,(x,y,300,h),''.join(rows)


def evaluate(image,arms,truth,box=None):
    detections={}
    rows={}
    for name,recognizer in arms.items():
        key=id(recognizer.detector)
        if key not in detections:detections[key]=recognizer.detector.detect(image)
        detection=detections[key]
        options=[]
        for candidate in detection.candidates[:recognizer.top_k]:
            result=recognizer.recognize_crop(candidate.crop)
            result['candidate']=candidate
            result['score']=candidate.score+recognizer.rerank_weight*result['quality']
            options.append(result)
        best=max(options,key=lambda r:r['score'],default=None)
        crop_result=None
        if box is not None:
            x,y,w,h=box
            crop_result=recognizer.recognize_crop(image[y:y+h,x:x+w])
        rows[name]=dict(truth=truth,text=best['decoded']['text'] if best else '',
            raw=best['decoded']['raw'] if best else '',candidates=len(detection.candidates),
            selected_box=list(best['candidate'].box) if best else None,
            selected_source=best['candidate'].source if best else None,
            selected_polarity=best['segmentation'].diagnostics.get('polarity') if best else None,
            crop_text=crop_result['decoded']['text'] if crop_result else None,
            top1_iou=intersection_over_union(best['candidate'].box,box) if box and best else 0)
    return rows


def main():
    cv2.setNumThreads(1)
    run=ROOT/'reports'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-color-robustness')
    run.mkdir(parents=True,exist_ok=False)
    base,receipt=load_filter_release(ModelRegistry(ROOT/'Registration'),'v1')
    arms=dict(baseline_04=base,dual_polarity=contrast_recognizer(base,structural=False),
              structural_dual=contrast_recognizer(base))
    synthetic=[]
    for body,color in BODIES.items():
        for style in STYLES:
            for two_rows in (False,True):
                image,box,truth=render(color,style,two_rows)
                rows=evaluate(image,arms,truth,box)
                for arm,row in rows.items():
                    synthetic.append(dict(arm=arm,body=body,style=style,two_rows=two_rows,**row))
                print(f'controlled body={body}, plate={style}, rows={2 if two_rows else 1}',flush=True)
    negatives=[]
    for body,color in BODIES.items():
        for grille in (False,True):
            image,box,truth=render(color,grille=grille)
            rows=evaluate(image,arms,truth)
            negatives.extend(dict(arm=arm,body=body,grille=grille,**row) for arm,row in rows.items())
    write_json(run/'synthetic_predictions.json',synthetic)
    write_json(run/'negative_controls.json',negatives)
    # A known validation set only: no claim of a new independent test result.
    manifest=json.loads((ROOT/'reports/20261022T091902580679Z-research/scene_manifest.json').read_text(encoding='utf-8'))
    scenes=[r for r in manifest if r['partition']=='validation']
    validation=[]
    for i,scene in enumerate(scenes):
        rows=evaluate(read_image(scene['path']),arms,scene['text'])
        validation.extend(dict(arm=arm,path=scene['path'],layout=scene['layout'],**row) for arm,row in rows.items())
        if (i+1)%10==0:print(f'VNLP validation {i+1}/{len(scenes)}',flush=True)
    synthetic_metrics={}
    for arm in arms:
        synthetic_metrics[arm]={}
        for style in STYLES:
            group=[r for r in synthetic if r['arm']==arm and r['style']==style]
            synthetic_metrics[arm][style]=dict(samples=len(group),
                crop_exact=sum(r['crop_text']==r['truth'] for r in group),
                full_exact=sum(r['text']==r['truth'] for r in group),
                selected_iou_hits=sum(r['top1_iou']>=.5 for r in group))
    val_metrics={arm:sequence_metrics([r['truth'] for r in validation if r['arm']==arm],
        [r['text'] for r in validation if r['arm']==arm]) for arm in arms}
    summary=dict(run_path=str(run),source=code_info(ROOT),base_artifacts=receipt['artifact_sha256'],
        synthetic=synthetic_metrics,validation=val_metrics,
        no_plate_nonempty_outputs={arm:sum(bool(r['text']) for r in negatives if r['arm']==arm) for arm in arms},
        no_plate_samples=10,status='controlled_color_rendering_and_reused_validation_not_real_color_benchmark',
        trained=False,promoted=False)
    write_json(run/'validation_predictions.json',validation)
    write_json(run/'summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)
    print('RUN: '+str(run),flush=True)
    return summary


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
