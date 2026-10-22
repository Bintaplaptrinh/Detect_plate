"""Presentation cells for the main notebook's independently runnable inference demo."""
import nbformat as nbf


def inference_cells():
    cells = []

    def md(source):
        cells.append(nbf.v4.new_markdown_cell(source.strip(), metadata={"tags": ["inference-demo"]}))

    def code(source):
        cells.append(nbf.v4.new_code_cell(source.strip(), metadata={"tags": ["inference-demo"]}))

    md("""
## 0. Demo nhận diện biển số — ảnh đầu vào → khung biển → chuỗi nhận diện

**Có thể chạy riêng mục này, không cần train lại.** Chọn kernel Python 3.14,
chạy ba cell bên dưới để nạp `pipeline/v3` từ Registry và xem kết quả thực tế.
Nếu chưa có Registry trên máy mới, chạy các mục training/registration phía dưới trước.

Ảnh được chạy từ đầu qua detector → segmentation → HOG/OCR → heuristic.
Khung xanh là candidate đứng đầu theo score detector, không bảo đảm nhận diện đúng.
Crop hiển thị lấy từ detector, không lấy bounding box ground truth. Các ảnh không
detect được vẫn xuất hiện cùng thông báo; không dùng nhãn thật để chọn ảnh thành công.
""")
    code("""
from pathlib import Path
import json, sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

DEMO_ROOT = Path.cwd().resolve()
if not (DEMO_ROOT / 'pyproject.toml').exists():
    DEMO_ROOT = DEMO_ROOT.parent
sys.path.insert(0, str(DEMO_ROOT / 'src'))
from plate_ml.imaging import read_image
from plate_ml.pipeline import PlateRecognizer
from plate_ml.registry import ModelRegistry

DEMO_VERSION = 'v3'  # Đổi rõ phiên bản muốn sử dụng, không tự chọn model mới nhất.
demo_registry = ModelRegistry(DEMO_ROOT / 'Registration')
demo_bundle, demo_receipt = demo_registry.load('pipeline', DEMO_VERSION, trusted=True)
demo_recognizer = PlateRecognizer.from_bundle(demo_bundle, demo_registry.root / 'RuntimeAssets')
DEMO_OUTPUT = DEMO_ROOT / 'reports' / 'inference'
DEMO_OUTPUT.mkdir(parents=True, exist_ok=True)

# Sửa list này để chạy ảnh của bạn. Mặc định: hai ảnh cũ + toàn bộ test scenes
# đã lưu trong manifest, giữ thứ tự manifest, không lọc theo kết quả/ground truth.
DEMO_IMAGE_PATHS = [DEMO_ROOT / '3.jpg', DEMO_ROOT / 'carlong_0004.png']
demo_detector_version = demo_receipt['metadata']['detector_version']
demo_manifest_path = (demo_registry.root / 'Registry' / 'detector' /
                      demo_detector_version / 'split_manifest.json')
demo_manifest = json.loads(demo_manifest_path.read_text(encoding='utf-8'))
DEMO_IMAGE_PATHS += [Path(row['path']) for row in demo_manifest if row['partition'] == 'test']

def infer_demo_image(image_path):
    image_path = Path(image_path)
    image = read_image(image_path)
    detection, predictions = demo_recognizer.recognize(image)
    top = predictions[0] if predictions else None
    annotated = image.copy()
    if top is not None:
        x, y, w, h = top.candidate.box
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 220, 0), 2)
        # ASCII overlay works with OpenCV's built-in font; Vietnamese labels use matplotlib.
        caption = top.decoded['text'] or 'NO CHARACTERS'
        cv2.putText(annotated, caption, (x, max(18, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 180, 0), 2)
    return dict(path=image_path, image=image, annotated=annotated, detection=detection, top=top)

def show_demo_detail(result):
    top = result['top']
    raw = top.decoded['raw'] if top else ''
    text = top.decoded['text'] if top else ''
    status = 'Không phát hiện vùng biển số' if top is None else (
        'Không tách được ký tự' if not text else text)
    display(Markdown(f"**Ảnh:** `{result['path'].name}` — **Biển số nhận diện: {status}**"))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].imshow(cv2.cvtColor(result['image'], cv2.COLOR_BGR2RGB))
    axes[0].set_title('Ảnh đầu vào')
    axes[1].imshow(cv2.cvtColor(result['annotated'], cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'Khung phát hiện / kết quả: {status}')
    if top is not None:
        axes[2].imshow(cv2.cvtColor(top.candidate.crop, cv2.COLOR_BGR2RGB))
        axes[2].set_title(f'Crop từ detector\\nOCR gốc: {raw or "(rỗng)"}\\nSau heuristic: {text or "(rỗng)"}')
    else:
        axes[2].text(.5, .5, status, ha='center', va='center', wrap=True)
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    if top is not None:
        print('Detector score:', round(top.candidate.score, 4), '| nguồn proposal:', top.candidate.source)
        print('OCR mean max-probability:', round(top.confidence, 4), '(không phải xác suất đúng cả biển)')
        chars = top.segmentation.characters
        if chars:
            fig, axes = plt.subplots(1, len(chars), figsize=(max(6, len(chars) * 1.1), 2.5), squeeze=False)
            for index, (ax, char) in enumerate(zip(axes.ravel(), chars)):
                row = top.probabilities[index]
                predicted = str(demo_recognizer.model.classes_[int(row.argmax())])
                ax.imshow(char.image, cmap='gray')
                ax.set_title(f'{predicted} / {row.max():.2f}\\nDòng {char.row + 1}')
                ax.axis('off')
            plt.tight_layout()
            plt.show()

print('Pipeline:', DEMO_VERSION, '| OCR:', demo_receipt['metadata']['ocr_version'],
      '| Detector:', demo_detector_version)
print('Số ảnh demo:', len(DEMO_IMAGE_PATHS))
""")
    code("""
# Chạy đúng pipeline trên từng ảnh, hiển thị cả ca không tìm thấy biển.
demo_results, demo_summary = [], []
for index, image_path in enumerate(DEMO_IMAGE_PATHS):
    result = infer_demo_image(image_path)
    demo_results.append(result)
    top = result['top']
    text = top.decoded['text'] if top else ''
    status = 'không phát hiện biển' if top is None else ('không tách được ký tự' if not text else 'đã nhận diện')
    demo_summary.append(dict(image=image_path.name, raw=top.decoded['raw'] if top else '',
        recognized_plate=text, status=status, detector_score=top.candidate.score if top else None))
    cv2.imencode('.png', result['annotated'])[1].tofile(DEMO_OUTPUT / f'{index:02d}_annotated.png')

columns = 3
rows = (len(demo_results) + columns - 1) // columns
fig, axes = plt.subplots(rows, columns, figsize=(18, 4.3 * rows), squeeze=False)
for ax, result, summary in zip(axes.ravel(), demo_results, demo_summary):
    ax.imshow(cv2.cvtColor(result['annotated'], cv2.COLOR_BGR2RGB))
    label = summary['recognized_plate'] or summary['status']
    ax.set_title(f'{result["path"].name[:40]}\\nKẾT QUẢ: {label}', fontsize=10)
    ax.axis('off')
for ax in axes.ravel()[len(demo_results):]:
    ax.axis('off')
fig.suptitle(f'Kết quả nhận diện từ ảnh đầy đủ — pipeline/{DEMO_VERSION}', fontsize=16)
plt.tight_layout(rect=(0, 0, 1, .985))
fig.savefig(DEMO_OUTPUT / 'recognition_gallery.png', dpi=120, bbox_inches='tight')
plt.show()
demo_table = pd.DataFrame(demo_summary)
display(demo_table)
demo_table.to_csv(DEMO_OUTPUT / 'recognition_results.csv', index=False, encoding='utf-8-sig')
print('Ảnh gắn kết quả và CSV được lưu tại:', DEMO_OUTPUT)
""")
    code("""
# Chi tiết hai ảnh ví dụ cũ và tối đa ba ảnh có candidate theo thứ tự đầu vào.
# Lọc có candidate chỉ để minh họa crop/ký tự; gallery phía trên vẫn giữ mọi ca lỗi.
detail_indices = list(range(min(2, len(demo_results))))
for index, result in enumerate(demo_results):
    if result['top'] is not None and index not in detail_indices:
        detail_indices.append(index)
    if len(detail_indices) >= 5:
        break
for index in detail_indices:
    show_demo_detail(demo_results[index])
""")
    md("""
### Chạy một ảnh tùy chọn

Đổi `MY_IMAGE` thành đường dẫn ảnh muốn thử và chạy cell dưới sau cell nạp model.
Kết quả gồm ảnh đầu vào, khung phát hiện, crop detector, chuỗi OCR gốc/sau heuristic
và các ký tự đã tách. Ảnh sai/không detect được được báo ngay trong output.
""")
    code("""
MY_IMAGE = None  # Ví dụ: DEMO_ROOT / '3.jpg', hoặc Path(r'D:\\images\\my_car.jpg')
if MY_IMAGE is not None:
    show_demo_detail(infer_demo_image(MY_IMAGE))
""")
    return cells
