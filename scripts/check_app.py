"""End-to-end local Rust API check with both frozen models and both modes."""
import argparse
import base64
import json
import math
from pathlib import Path
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def request_image(url, content, algorithm='svm', multi='false', image=True):
    boundary = 'PlateDetectTestBoundary'
    chunks=[]
    for name,value in [('algorithm',algorithm),('multi_detection',multi)]:
        chunks.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    if image:
        chunks.append(f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="sample.jpg"\r\nContent-Type: image/jpeg\r\n\r\n'.encode()+content+b'\r\n')
    chunks.append(f'--{boundary}--\r\n'.encode())
    request=Request(url+'/api/recognize',data=b''.join(chunks),headers={'Content-Type':f'multipart/form-data; boundary={boundary}'})
    try:
        with urlopen(request,timeout=70) as response:
            return response.status,json.load(response)
    except HTTPError as exc:
        return exc.code,json.loads(exc.read().decode('utf-8'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',default='http://127.0.0.1:8787')
    parser.add_argument('--image',type=Path,default=ROOT/'multi_dectec_sample.jpg')
    args=parser.parse_args()
    destination=ROOT/'reports/app-demo'
    destination.mkdir(parents=True,exist_ok=True)
    with urlopen(args.url+'/api/health') as response:
        health=json.load(response)
    assert health['ready'] and health['algorithms']==['svm','pca_fld_hmm']
    content=args.image.read_bytes()
    runs=[]
    for algorithm in ['svm','pca_fld_hmm']:
        for multi in [False,True]:
            status,data=request_image(args.url,content,algorithm,str(multi).lower())
            assert status==200,data
            assert data['algorithm']==algorithm and data['multi_detection']==multi
            assert math.isfinite(data['processing_seconds']) and data['processing_seconds']>=0
            output=base64.b64decode(data.pop('image_base64'),validate=True)
            decoded=cv2.imdecode(np.frombuffer(output,np.uint8),cv2.IMREAD_COLOR)
            assert decoded.shape[:2]==(data['image_height'],data['image_width'])
            if not multi:assert len(data['detections'])<=1
            for d in data['detections']:
                x,y,w,h=d['box']
                assert x>=0 and y>=0 and w>0 and h>0
                assert x+w<=decoded.shape[1] and y+h<=decoded.shape[0]
                assert d['text'] and d['id']>=1
            if multi: assert data['diagnostics']['ocr_calls']<=96
            tag=f"{algorithm}-{'multi' if multi else 'single'}"
            (destination/f'{tag}.jpg').write_bytes(output)
            runs.append(data)
            print(tag,round(data['processing_seconds'],3),'seconds',[(d['text'],d['box']) for d in data['detections']],flush=True)
    for fields in [dict(algorithm='unknown'),dict(multi='False'),dict(image=False),dict(content=b'not an image')]:
        options=dict(content=content);options.update(fields)
        status,data=request_image(args.url,**options)
        assert status==400 and data.get('error'),(fields,status,data)
    # An invalid image must not desynchronize JSON-lines replies for later images.
    status,data=request_image(args.url,content,'svm','false')
    assert status==200 and data['algorithm']=='svm'
    (destination/'api-verification.json').write_text(json.dumps(dict(health=health,runs=runs,
        invalid_requests_checked=4,recovery_after_invalid_request=True),ensure_ascii=False,indent=2),encoding='utf-8')
    print('API checks passed. Outputs:',destination)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
