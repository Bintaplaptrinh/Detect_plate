"""Focused notebook: pre-OCR glyph quality filter and its regression evidence."""
from pathlib import Path
from textwrap import dedent
import nbformat as nbf


def build():
    cells=[]
    def md(s):cells.append(nbf.v4.new_markdown_cell(dedent(s).strip()))
    def code(s):cells.append(nbf.v4.new_code_cell(dedent(s).strip()))
    md('''
    # Thí nghiệm 04 — loại crop ký tự nhiễu trước HOG/OCR

    Hai ca người dùng báo: `36A406691` có phần ngoại thất bị đọc thành `1`;
    `29X254` thiếu `3` dù crop biển vẫn chứa chữ số đó. Đây là **ca phát triển**,
    không dùng độ đúng trên hai ảnh này làm benchmark độc lập.

    Pipeline: detector/crop → các giả thuyết CCA/histogram → **lọc chất lượng từng
    crop ký tự** → deskew, 24×40 trên canvas 32×48, HOG 540 chiều → OCR đã đóng băng
    → chọn giả thuyết/rerank. Không xóa ký tự chỉ vì confidence thấp, không chèn
    số theo filename hoặc ép độ dài đúng nhãn.
    ''')
    code('''
    from pathlib import Path
    import os, sys, json
    import cv2
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display
    ROOT=Path.cwd().resolve()
    if not (ROOT/'pyproject.toml').exists():ROOT=ROOT.parent
    sys.path.insert(0,str(ROOT/'src'))
    sys.path.insert(0,str(ROOT/'scripts'))
    from plate_ml.registry import ModelRegistry
    from plate_ml.imaging import read_image
    from plate_ml.features import normalize_glyph
    from run_research_experiment import load_research_release
    from run_glyph_filter_experiment import load_filter_release
    cv2.setNumThreads(1)
    RUN=Path(os.environ.get('PLATE_GLYPH_FILTER_RUN',str(ROOT/'reports/20261022T104205345901Z-glyph-filter')))
    def read_json(name):return json.loads((RUN/(name+'.json')).read_text(encoding='utf-8'))
    summary=read_json('summary')
    registry=ModelRegistry(ROOT/'Registration')
    before,_=load_research_release(registry,summary['baseline_version'])
    after,_=load_filter_release(registry,summary['release']['version'])
    display(summary['selection']['quality_config'])
    print('Source/HOG/model cũ giữ nguyên. Filter trước HOG; chỉ giữ crop qua kiểm tra.')
    ''')
    md('''
    ## Vì sao không chỉ đặt ngưỡng confidence?

    SVM vẫn phải chọn một class cho crop ngoài miền. Bộ lọc dùng ảnh và hình học:

    - Có ít nhất 3 ký tự hỗ trợ trong hàng mới so chiều cao; loại blob thấp hơn
      một nửa median chiều cao hàng. Ước lượng đường hàng bằng slope median để
      chịu được biển nghiêng.
    - Chỉ loại crop phân mảnh khi đồng thời ở mép và lệch hàng.
    - Chỉ loại nét đặc giống viền khi đồng thời hẹp, sát mép, cách cụm chữ xa và
      lệch hàng. Số `1` hẹp nhưng đầy đủ chiều cao và cùng hàng vẫn được giữ.
    - Không blur/erode ký tự thật và không thay chuẩn HOG của OCR.

    Với `29X2534`, CCA gộp `3` vào viền rồi loại khối. Nhánh histogram đã có `3`;
    sau khi bỏ hai crop mép nhiễu, nhánh đó thắng bằng bằng chứng OCR/hình học.
    Không dựng lại một số `3` không tồn tại trong ảnh.
    ''')
    code('''
    cases=read_json('reported_cases')
    display(pd.DataFrame([dict(arm=arm,truth=r['truth'],raw=r['raw'],text=r['text'],method=r['method'])
                         for arm,rows in cases['predictions'].items() for r in rows]))
    validation=summary['selection']['metrics']
    display(pd.DataFrame([dict(arm=arm,correct=m['exact_count'],samples=m['decoded']['samples'],
        raw_cer=m['raw']['character_error_rate'],cer=m['decoded']['character_error_rate'])
        for arm,m in validation.items()]))
    print('Validation regression:',len(summary['selection']['changes']['regressed']))
    print('Khóa trước test mới:',summary['selection']['selected'])
    ''')
    md('''
    ## Regression cũ và holdout mới phải đọc riêng

    180 test scenes của thí nghiệm 03 nay dùng làm **regression audit**, vì đã thấy
    và dùng hai lỗi của tập đó để phát triển filter. Holdout mới gồm 180 identity
    khác (60/layout), loại toàn bộ identity đã xuất hiện ở thí nghiệm 02/03.
    Detector, OCR, ranking weights giữ nguyên; chỉ thay filter crop ký tự.
    Đây vẫn là cùng dataset/camera, không phải benchmark ngoài miền.
    ''')
    code('''
    def metric_table(data):
        return pd.DataFrame([dict(arm=arm,correct=m['exact_count'],samples=m['decoded']['samples'],
            exact=m['decoded']['exact_match'],raw_cer=m['raw']['character_error_rate'],
            cer=m['decoded']['character_error_rate']) for arm,m in data.items()])
    print('Regression audit 180 ảnh cũ:')
    display(metric_table(summary['old_regression_metrics']))
    print('Sửa được:',len(summary['old_regression_changes']['repaired']),
          '| Đang đúng thành sai:',len(summary['old_regression_changes']['regressed']))
    display(pd.DataFrame(summary['old_regression_changes']['regressed']))
    print('180 ảnh mới, chưa dùng để phát triển filter:')
    fresh_table=metric_table(summary['fresh_test'])
    display(fresh_table)
    display(pd.DataFrame([dict(arm=arm,layout=layout,**m)
        for arm,layouts in summary['fresh_by_layout'].items() for layout,m in layouts.items()]))
    print('Fresh repaired:',len(summary['fresh_changes']['repaired']),
          '| Fresh regressed:',len(summary['fresh_changes']['regressed']))
    display(pd.DataFrame(summary['fresh_changes']['regressed']))
    fresh_table.to_csv(RUN/'fresh_metrics.csv',index=False)
    ''')
    md('''
    ## So sánh trực quan hai ca báo lỗi

    Xanh: crop được giữ. Đỏ: crop bị loại trước HOG/OCR. Ảnh detector trước/sau có
    thể khác nhau vì filter tác động đến xếp hạng cuối; detector proposals không đổi.
    Các số dự đoán đều được suy ra trước khi dùng GT để hiển thị.
    ''')
    code('''
    def show_case(path,truth=None):
        image=read_image(path)
        fig,axes=plt.subplots(2,3,figsize=(17,7))
        result_rows=[]
        for row,(name,recognizer) in enumerate([('Trước filter',before),('Sau filter',after)]):
            _,results=recognizer.recognize(image)
            best=results[0] if results else None
            if best:
                seg=best['segmentation']
                axes[row,0].imshow(cv2.cvtColor(best['candidate'].crop,cv2.COLOR_BGR2RGB))
                axes[row,0].set_title(name+' — crop detector')
                overlay=cv2.cvtColor(seg.binary,cv2.COLOR_GRAY2BGR)
                quality=seg.diagnostics.get('glyph_quality')
                decisions=(quality['decisions'] if quality else
                    [dict(index=i,box=list(c.box),accepted=True) for i,c in enumerate(seg.characters)])
                for d in decisions:
                    x,y,w,h=d['box']
                    cv2.rectangle(overlay,(x,y),(x+w,y+h),(0,200,0) if d['accepted'] else (0,0,255),2)
                axes[row,1].imshow(cv2.cvtColor(overlay,cv2.COLOR_BGR2RGB))
                axes[row,1].set_title(f"{seg.diagnostics.get('method')} / {seg.diagnostics.get('threshold')}")
                glyphs=[normalize_glyph(c.image,recognizer.extractor.config) for c in seg.characters]
                if glyphs:axes[row,2].imshow(np.hstack(glyphs),cmap='gray')
                axes[row,2].set_title(f"Raw: {best['decoded']['raw']} | text: {best['decoded']['text']}")
                result_rows.append(dict(arm=name,truth=truth,raw=best['decoded']['raw'],text=best['decoded']['text']))
                if name=='Sau filter' and quality:
                    rejected=[d for d in quality['decisions'] if not d['accepted']]
                    if rejected:display(pd.DataFrame(rejected))
        for ax in axes.ravel():ax.axis('off')
        fig.suptitle('GT chỉ để đối chiếu: '+str(truth),y=.98)
        plt.tight_layout(rect=(0,0,1,.95))
        if truth:plt.savefig(RUN/(truth+'_before_after.png'),dpi=140)
        plt.show()
        return result_rows
    demo=[]
    for row in cases['predictions']['baseline_v3']:
        demo.extend(show_case(row['path'],row['truth']))
    display(pd.DataFrame(demo))
    ''')
    code('''
    # Thử ảnh riêng: không yêu cầu tên file có nhãn VNLP.
    MY_IMAGE=None
    if MY_IMAGE is not None:
        display(pd.DataFrame(show_case(Path(MY_IMAGE))))
    print('Reload verified:',summary['reload_verified'],'| Release:',summary['release'])
    ''')
    md('''
    Chi tiết chỉ số và giới hạn: [glyph quality experiment](../docs/glyph-quality-experiment.md).
    Module chính: `src/plate_ml/glyph_quality.py`; mọi quyết định loại đều có reason,
    box, relative height, row offset và thành phần liên thông trong trace.

    Source notebook 01/02/03 và model research_pipeline/v3 không bị ghi đè. Model
    có filter ở namespace riêng `research_glyph_filter`. Việc sửa được hai ảnh
    không bảo đảm mọi glyph nhiễu hoặc mọi số `1` đều được xử lý đúng.
    ''')
    notebook=nbf.v4.new_notebook(cells=cells,metadata=dict(
        kernelspec=dict(name='plate-ml-314',display_name='Python 3.14 (PlateDetect)',language='python'),
        language_info=dict(name='python',version='3.14.2')))
    path=Path(__file__).resolve().parents[1]/'notebooks/04_glyph_quality_filter.ipynb'
    nbf.write(notebook,path)
    return path


if __name__=='__main__':print(build())
