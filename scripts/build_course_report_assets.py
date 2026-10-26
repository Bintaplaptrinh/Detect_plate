"""Build report figures and tables from the saved classical-model experiments."""
import json
from pathlib import Path
import sys

import cv2
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from plate_ml.features import normalize_glyph
from plate_ml.imaging import read_image
from plate_ml.inference import load_recognizer
from plate_ml.research_pipeline import ResearchRecognizer

REPORT = ROOT / 'docs/report'
ASSETS = REPORT / 'assets'
RUN = ROOT / 'reports/pca-fld-hmm-no-svm-180-20261026'
REFERENCE = ROOT / 'reports/pca-fld-hmm-180-20261026-v2'
NAMES = {'current_svm': 'SVM', 'pca_fld_svm_hmm': 'PCA, FLD, SVM, HMM',
         'pca_fld_hmm': 'PCA, FLD, Gaussian, HMM'}
COLORS = ['#263f65', '#6e8796', '#9b6041']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(fig, name):
    fig.savefig(ASSETS / name, dpi=220, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def pct(value):
    return f'{value * 100:.2f}'.replace('.', ',')


def num(value):
    return f'{value:.2f}'.replace('.', ',')


def row(values):
    return ' & '.join(map(str, values)) + r' \\' + '\n'


def main():
    ASSETS.mkdir(exist_ok=True)
    for font in ('times.ttf', 'timesbd.ttf', 'timesi.ttf'):
        path = Path('C:/Windows/Fonts') / font
        if path.exists():
            font_manager.fontManager.addfont(str(path))
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 15,
                         'axes.spines.top': False, 'axes.spines.right': False})
    summary = read(RUN / 'summary.json')
    training = read(RUN / 'training_audit.json')
    compact = read(ROOT / 'docs/results/pca-fld-hmm-no-svm-180-20261026.json')
    assert compact['summary']['metrics'] == summary['metrics']
    tables = REPORT / 'tables'
    tables.mkdir(exist_ok=True)
    classes, counts = zip(*training['classes'].items())
    fig, ax = plt.subplots(figsize=(8, 3))
    bars = ax.bar(classes, counts, color=['#263f65' if c.isdigit() else '#9b6041' for c in classes])
    ax.set(xlabel='Lớp ký tự', ylabel='Số mẫu huấn luyện', ylim=(0, 550))
    ax.grid(axis='y', alpha=.18)
    save(fig, 'phan_bo_lop.png')
    chunks = [list(training['classes'].items())[i:i+12] for i in (0,12,24)]
    countrows = []
    for i in range(12):
        vals = []
        for chunk in chunks:
            vals.extend(chunk[i] if i < len(chunk) else ('', ''))
        countrows.append(row(vals))
    (tables / 'class_counts.tex').write_text(''.join(countrows), encoding='utf-8')

    metrics = summary['metrics']
    labels = ['A', 'B', 'C']
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.3))
    x = np.arange(3)
    for j,(key,legend) in enumerate([('exact_match','Đúng toàn biển'),('character_error_rate','CER')]):
        vals = [100*m['decoded'][key] for m in metrics.values()]
        bars = axes[0].bar(x+(j-.5)*.34, vals, width=.34, color=[COLORS[0],COLORS[2]][j], label=legend)
        axes[0].bar_label(bars, fmt='%.2f', fontsize=12)
    axes[0].set(xticks=x, xticklabels=labels, ylabel='Tỷ lệ (%)', ylim=(0,65))
    axes[0].legend(loc='upper right', fontsize=12)
    vals = [m['latency']['mean_ms'] for m in metrics.values()]
    bars = axes[1].bar(x, vals, color=COLORS)
    axes[1].bar_label(bars, fmt='%.2f', padding=3)
    axes[1].set(xticks=x, xticklabels=labels, ylabel='Thời gian (ms/ảnh)', ylim=(0,205))
    for ax in axes: ax.grid(axis='y', alpha=.18)
    fig.tight_layout(w_pad=3)
    save(fig, 'so_sanh_chinh.png')
    stage_keys = ['detector','segmentation','hog','classifier','decoder']
    stage_names = ['Phát hiện','Phân đoạn','HOG','Mô hình quan sát','Giải mã']
    fig, ax = plt.subplots(figsize=(8,3.3))
    bottom = np.zeros(3)
    for key,label,color in zip(stage_keys,stage_names,['#263f65','#6e8796','#c0c7c9','#9b6041','#d4ac83']):
        vals = [m['stages_mean_ms'][key] for m in metrics.values()]
        ax.bar(x,vals,bottom=bottom,color=color,label=label)
        bottom += vals
    ax.set(xticks=x,xticklabels=labels,ylabel='Thời gian công đoạn (ms/ảnh)')
    ax.legend(loc='upper left',bbox_to_anchor=(1.01,1),fontsize=12)
    save(fig, 'thoi_gian_cong_doan.png')

    mainrows, glyphrows, layoutrows, stagerows, errorrows = [], [], [], [], []
    for key,m in metrics.items():
        mainrows.append(row([NAMES[key],f"{m['exact_count']}/180",pct(m['decoded']['exact_match']),pct(m['decoded']['character_error_rate']),num(m['latency']['mean_ms'])]))
        stagerows.append(row([NAMES[key]] + [num(m['stages_mean_ms'][s]) for s in stage_keys]))
        align=m['aligned_text']['alignment']
        errorrows.append(row([NAMES[key],align['substitutions'],align['deletions'],align['insertions'],align['edit_distance']]))
        g = summary['glyph_diagnostic']['models'][key]
        glyphrows.append(row([NAMES[key],pct(g['accuracy']),pct(g['macro_f1']),num(g['batch_median_ms'])]))
        cm=np.asarray(g['confusion_matrix'])
        fig,ax=plt.subplots(figsize=(7,6.8))
        im=ax.imshow(np.log1p(cm),cmap='Blues',vmin=0,vmax=np.log1p(65))
        for i,j in zip(*np.nonzero(cm)):
            ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=11,color='white' if cm[i,j]>10 else '#172636')
        ax.set(xticks=range(35),yticks=range(35),xticklabels=g['classes'],yticklabels=g['classes'],
               xlabel='Lớp dự đoán',ylabel='Lớp thực tế')
        ax.tick_params(labelsize=13,length=0)
        ax.set_xticks(np.arange(-.5,35,1),minor=True)
        ax.set_yticks(np.arange(-.5,35,1),minor=True)
        ax.grid(which='minor',color='#d9e0e6',linewidth=.3)
        ax.tick_params(which='minor',length=0)
        save(fig, f'nham_lan_{key}.png')
        errors=sorted([(int(cm[i,j]),g['classes'][i],g['classes'][j]) for i in range(35) for j in range(35) if i!=j and cm[i,j]],reverse=True)
        print(key,'top confusion',errors[:8])
    for layout,label in [('one_row','Một hàng'),('two_rows','Hai hàng ô tô'),('two_rows_label_xe_may','Hai hàng xe máy')]:
        layoutrows.append(row([label]+[pct(metrics[k]['by_layout'][layout]['exact_match']) for k in NAMES]))
    for name,content in [('main_results',mainrows),('glyph_results',glyphrows),('layout_results',layoutrows),('stage_results',stagerows),('error_results',errorrows)]:
        (tables/f'{name}.tex').write_text(''.join(content),encoding='utf-8')

    rows={k:read(RUN/f'{k}_predictions.json') for k in NAMES}
    a,b=rows['current_svm'],rows['pca_fld_hmm']
    cases=['36A43014','36A40043','89G103316','35G102993']
    case_rows=[]
    fig,axes=plt.subplots(2,2,figsize=(9,6.4))
    for ax,truth in zip(axes.flat,cases):
        index=next(i for i,r in enumerate(a) if r['truth']==truth)
        r,t=a[index],b[index]
        image=read_image(r['path'])
        ax.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        for box,color,style in [(r['gt_box'],'#23833e','-'),(r['box'],'#2a67b0','--'),(t['box'],'#bc4d26',':')]:
            if box:
                xx,yy,ww,hh=box
                ax.add_patch(Rectangle((xx,yy),ww,hh,fill=False,color=color,linestyle=style,linewidth=1.8))
        ax.set_title(f"Nhãn: {truth}\nSVM: {r['text']}\nGaussian: {t['text']}",fontsize=16)
        ax.axis('off')
        case_rows.append(row([truth,r['text'],t['text']]))
    fig.tight_layout(h_pad=1.8)
    save(fig,'vi_du_loi.png')
    (tables/'case_results.tex').write_text(''.join(case_rows),encoding='utf-8')

    cv2.setNumThreads(1)
    with threadpool_limits(limits=1):
        current=load_recognizer(ROOT/'Registration')
        saved=joblib.load(RUN/'no_svm_candidate.joblib')
        candidate=ResearchRecognizer(saved['model'],current.detector,current.segmenter,current.extractor.config,
                                    current.templates,current.rerank_weight,current.top_k,decoder=saved['hmm'])
        sample=next(r for r in a if r['truth']=='36C28038')
        image=read_image(sample['path'])
        _,pred=current.recognize(image)
        _,other=candidate.recognize(image)
        best=pred[0]
        assert best['decoded']['text']==sample['text']==other[0]['decoded']['text']=='36C28038'
        fig,axes=plt.subplots(1,3,figsize=(9,3.1),gridspec_kw={'width_ratios':[2,1.5,1.6]})
        axes[0].imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        xx,yy,ww,hh=best['candidate'].box
        axes[0].add_patch(Rectangle((xx,yy),ww,hh,fill=False,color='#23833e',linewidth=2))
        axes[0].set_title('Ảnh và vùng được chọn')
        axes[1].imshow(cv2.cvtColor(best['candidate'].crop,cv2.COLOR_BGR2RGB))
        axes[1].set_title('Vùng biển sau hiệu chỉnh')
        glyphs=[normalize_glyph(c.image,current.extractor.config) for c in best['segmentation'].characters]
        axes[2].imshow(np.hstack(glyphs),cmap='gray')
        axes[2].set_title('Ký tự chuẩn hóa\nKết quả: 36C28038')
        for ax in axes:ax.axis('off')
        fig.tight_layout()
        save(fig,'minh_hoa_nhan_dang.png')
        fig,axes=plt.subplots(2,8,figsize=(11,3.5))
        for i,c in enumerate(best['segmentation'].characters[:8]):
            axes[0,i].imshow(c.image,cmap='gray');axes[1,i].imshow(glyphs[i],cmap='gray')
            axes[0,i].set_title('36C28038'[i],fontsize=21)
        for ax in axes.flat:ax.axis('off')
        fig.tight_layout()
        save(fig,'chuan_hoa_ky_tu.png')
    audit={'experiment_summary_sha256':compact['summary_sha256'],
           'training_samples':sum(counts),'test_scenes':180,'diagnostic_glyphs':434,
           'demo_truth':'36C28038','demo_svm':best['decoded']['text'],'demo_gaussian_hmm':other[0]['decoded']['text'],
           'report_figures_from_saved_results':True}
    (REPORT/'asset_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Report assets ready:',ASSETS)


if __name__=='__main__':
    main()
