"""Persistent JSON-lines inference worker owned by the local Rust application."""
import os
from pathlib import Path
import sys

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '1')
os.environ.setdefault('OPENCV_IO_MAX_IMAGE_PIXELS', '16000000')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import base64
import binascii
import json
import cv2
import numpy as np
from threadpoolctl import threadpool_limits
from plate_ml.demo import DemoEngine


def write(value):
    print(json.dumps(value, ensure_ascii=False, allow_nan=False), flush=True)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    cv2.setNumThreads(1)
    with threadpool_limits(limits=1):
        try:
            engine = DemoEngine(ROOT)
        except Exception as exc:
            write(dict(ready=False, error=str(exc)))
            return 1
        write(dict(ready=True, algorithms=['svm', 'pca_fld_hmm']))
        for line in sys.stdin:
            try:
                request = json.loads(line)
                encoded = request.get('image_base64', '')
                if not isinstance(encoded, str) or len(encoded) > 28_000_000:
                    raise ValueError('Ảnh tải lên vượt giới hạn dung lượng.')
                content = base64.b64decode(encoded, validate=True)
                if not content or len(content) > 20*1024*1024:
                    raise ValueError('Dung lượng ảnh không hợp lệ.')
                image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError('Không giải mã được ảnh. Vui lòng chọn ảnh JPG, PNG, WEBP hoặc BMP hợp lệ.')
                result, annotated = engine.recognize(image, request.get('algorithm'), request.get('multi_detection'))
                ok, data = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 93])
                if not ok:
                    raise ValueError('Không xuất được ảnh kết quả.')
                result.update(image_mime='image/jpeg', image_base64=base64.b64encode(data.tobytes()).decode('ascii'))
                write(result)
            except (ValueError, TypeError, KeyError, binascii.Error, cv2.error) as exc:
                write(dict(error=str(exc)))
            except Exception as exc:
                # Keep one response per request and preserve the worker protocol.
                print(f'Inference error: {exc!r}', file=sys.stderr, flush=True)
                write(dict(error='Không hoàn tất được xử lý ảnh. Vui lòng thử lại.'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
